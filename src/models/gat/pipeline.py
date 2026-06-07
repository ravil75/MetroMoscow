# src/models/gat/pipeline.py

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ...backtest import make_rolling_folds, summarize_results
from ...baseline_models import forecast_metrics
from ...synthesis import synthesize_from_train, validate_synthetic, get_synth_days
from ...windowing import make_time_covariates, make_correlation_adjacency
from ... import config

# ── Константы ─────────────────────────────────────────────────────────────────
NIGHT_HOURS   = frozenset({0, 1, 2, 3, 4, 5})
MIN_DAILY_PAX = 50

# ── Вспомогательные функции ───────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)

def compute_object_scales(df: pd.DataFrame) -> dict:
    scales = {}
    for col in df.columns:
        vals = df[col].values.astype(float)
        positive = vals[vals > 0]
        scales[col] = max(float(np.mean(positive)) if len(positive) else 1.0, 1.0)
    return scales

def get_active_objects(train_slice: pd.DataFrame, object_ids: list, min_daily_pax: float = MIN_DAILY_PAX) -> list:
    n_days = max(len(train_slice) / 24, 1.0)
    active = [oid for oid in object_ids if train_slice[oid].sum() / n_days >= min_daily_pax]
    return active if len(active) >= 10 else list(object_ids)

def get_day_mask(target_index) -> np.ndarray:
    return np.array([ts.hour not in NIGHT_HOURS for ts in target_index])

# ── Dataset ───────────────────────────────────────────────────────────────────

class GATWindowDataset(Dataset):
    """
    В отличие от TFT, один сэмпл GAT содержит ВСЕ объекты сразу (весь граф).
      x       : [past_window, num_nodes, features]
      dec_cov : [horizon,     cov_features]
      y       : [horizon,     num_nodes]
    """
    def __init__(self, frames, object_ids, scales, past_window, horizon, stride=1, is_synthetic=0):
        self.object_ids   = list(object_ids)
        self.past_window  = past_window
        self.horizon      = horizon
        self.stride       = max(int(stride), 1)
        self.samples      = []
        
        self.scales_log = np.log1p(
            np.array([scales[oid] for oid in self.object_ids], dtype=np.float32)
        ) # [num_nodes]

        for frame in frames:
            if len(frame) == 0: continue
            frame = frame.sort_index()
            data = frame[self.object_ids].values.astype(np.float32) # [T, N]
            cov = make_time_covariates(frame.index, is_synthetic=is_synthetic).values.astype(np.float32) # [T, 9]
            
            max_start = len(data) - past_window - horizon
            for s in range(0, max_start + 1, self.stride):
                self.samples.append((data, cov, s))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, cov, start = self.samples[idx]
        
        h_s, h_e = start, start + self.past_window
        t_s, t_e = h_e, h_e + self.horizon

        history = data[h_s:h_e]   # [pw, N]
        target  = data[t_s:t_e]   # [h, N]

        # Нормализация
        hist_norm   = np.log1p(history) / self.scales_log # [pw, N]
        target_norm = np.log1p(target)  / self.scales_log # [h, N]

        enc_cov = cov[h_s:h_e] # [pw, 9]
        dec_cov = cov[t_s:t_e] # [h, 9]

        # Подготовка X: [pw, N, 1 + 9] = [pw, N, 10]
        hist_norm_exp = hist_norm[..., np.newaxis] # [pw, N, 1]
        enc_cov_exp   = np.broadcast_to(enc_cov[:, np.newaxis, :], (self.past_window, len(self.object_ids), enc_cov.shape[-1]))
        
        x = np.concatenate([hist_norm_exp, enc_cov_exp], axis=-1) # [pw, N, 10]

        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(dec_cov.astype(np.float32)),
            torch.from_numpy(target_norm.astype(np.float32)),
        )

# ── GAT Строительные блоки ────────────────────────────────────────────────────

class DenseGATLayer(nn.Module):
    """GAT слой, работающий с плотной матрицей смежности (удобно для N < 2000)."""
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1, alpha=0.2):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // n_heads

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, 1, n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(1, 1, n_heads, self.head_dim))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        # h: [B, N, in_dim], adj: [N, N]
        B, N, _ = h.size()
        Wh = self.W(h).view(B, N, self.n_heads, self.head_dim) # [B, N, H, D]

        attn_src = (Wh * self.a_src).sum(dim=-1) # [B, N, H]
        attn_dst = (Wh * self.a_dst).sum(dim=-1) # [B, N, H]

        # e_{ij} = LeakyReLU( a^T [Wh_i || Wh_j] )
        e = attn_src.unsqueeze(2) + attn_dst.unsqueeze(1) # [B, N, N, H]
        e = self.leakyrelu(e)

        # Маскирование отсутствующих ребер
        adj_mask = (adj == 0).unsqueeze(0).unsqueeze(-1) # [1, N, N, 1]
        e = e.masked_fill(adj_mask, float('-inf'))

        alpha = F.softmax(e, dim=2) # [B, N, N, H]
        alpha = self.dropout(alpha)

        # Агрегация сообщений: sum_j (alpha_{ij} * Wh_j)
        out = torch.einsum('bijh,bjhd->bihd', alpha, Wh)
        return out.reshape(B, N, self.out_dim)

class STGATForecaster(nn.Module):
    """Spatio-Temporal GAT: LSTM для времени + GAT для пространства."""
    def __init__(self, num_nodes, in_features=10, dec_cov_features=9, hidden_dim=64, 
                 horizon=24, n_heads=4, lstm_layers=2, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        # Temporal Encoder
        self.lstm = nn.LSTM(in_features, hidden_dim, lstm_layers, batch_first=True, 
                            dropout=dropout if lstm_layers > 1 else 0.0)
        
        # Spatial Graph Attention
        self.gat1 = DenseGATLayer(hidden_dim, hidden_dim, n_heads, dropout)
        self.gat2 = DenseGATLayer(hidden_dim, hidden_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Decoder (MLP для каждого узла)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + dec_cov_features * horizon, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon)
        )

    def forward(self, x, dec_cov, adj):
        # x: [B, pw, N, F], dec_cov: [B, h, C], adj: [N, N]
        B, pw, N, F = x.size()

        # 1. Извлекаем временные фичи через LSTM независимо для каждого узла
        x_flat = x.transpose(1, 2).reshape(B * N, pw, F) 
        lstm_out, _ = self.lstm(x_flat) 
        h_nodes = lstm_out[:, -1, :].view(B, N, -1) # берем последний hidden state -> [B, N, hidden]

        # 2. Пространственный обмен (GAT)
        g1 = self.gat1(h_nodes, adj)
        h_nodes = self.norm1(h_nodes + F.elu(g1)) # residual + norm

        g2 = self.gat2(h_nodes, adj)
        h_nodes = self.norm2(h_nodes + F.elu(g2)) # [B, N, hidden]

        # 3. Декодер с учетом будущих ковариат (праздники, выходные в горизонте)
        h, C = dec_cov.size(1), dec_cov.size(2)
        dec_cov_flat = dec_cov.reshape(B, h * C).unsqueeze(1).expand(B, N, h * C)
        
        out_feat = torch.cat([h_nodes, dec_cov_flat], dim=-1) # [B, N, hidden + h*C]
        pred = self.decoder(out_feat) # [B, N, horizon]

        return pred.transpose(1, 2) # Возвращаем [B, horizon, N]

# ── Конфигурация ──────────────────────────────────────────────────────────────

@dataclass
class GATTrainConfig:
    past_window:   int   = 72
    hidden_dim:    int   = 64    # Для графов 64-128 оптимально, чтобы избежать OOM
    n_heads:       int   = 4
    lstm_layers:   int   = 2
    dropout:       float = 0.1
    epochs:        int   = 20
    batch_size:    int   = 16    # ВАЖНО: 1 сэмпл = весь граф, поэтому bs=16 эквивалентно bs=~16000 в TFT
    learning_rate: float = 1e-3
    weight_decay:  float = 1e-4
    window_stride: int   = 1
    synthetic_window_stride: int = 24
    num_workers:   int   = 0
    device:        str   = "auto"
    seed:          int   = 42
    amp:           bool  = True

# ── Обучение ──────────────────────────────────────────────────────────────────

def train_gat(train_real, synthetic_train, horizon, cfg, scale_source=None):
    set_seed(cfg.seed)
    object_ids = list(train_real.columns)
    scales     = compute_object_scales(scale_source if scale_source is not None else train_real)

    # Строим граф смежности на основе корреляций (только по реальным данным)
    adj_df = make_correlation_adjacency(train_real, top_k=8, min_corr=0.30)
    adj_matrix = torch.from_numpy(adj_df.values).float()

    frames  = [train_real]
    strides = [cfg.window_stride]
    if synthetic_train is not None:
        frames.append(synthetic_train[object_ids])
        strides.append(cfg.synthetic_window_stride)

    datasets = [
        GATWindowDataset([f], object_ids, scales, cfg.past_window, horizon, stride=s, is_synthetic=int(i > 0))
        for i, (f, s) in enumerate(zip(frames, strides))
    ]
    usable = [d for d in datasets if len(d)]
    if not usable:
        raise ValueError("No GAT training windows.")

    train_ds = usable[0] if len(usable) == 1 else torch.utils.data.ConcatDataset(usable)
    loader   = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                          num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())

    device = resolve_device(cfg.device)
    adj_matrix = adj_matrix.to(device)

    model = STGATForecaster(
        num_nodes        = len(object_ids),
        in_features      = 10,
        dec_cov_features = 9,
        hidden_dim       = cfg.hidden_dim,
        horizon          = horizon,
        n_heads          = cfg.n_heads,
        lstm_layers      = cfg.lstm_layers,
        dropout          = cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.1)
    loss_fn = nn.HuberLoss()
    scaler  = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    model.train()
    history = []
    for epoch in range(1, cfg.epochs + 1):
        total_loss, n_seen = 0.0, 0
        for x, dec_cov, y in loader:
            x       = x.to(device=device)
            dec_cov = dec_cov.to(device=device)
            y       = y.to(device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                pred = model(x, dec_cov, adj_matrix)
                loss = loss_fn(pred, y)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * x.size(0)
            n_seen     += x.size(0)

        scheduler.step()
        epoch_loss = total_loss / max(n_seen, 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"GAT h={horizon} epoch={epoch}/{cfg.epochs} loss={epoch_loss:.5f}")

    return model, scales, adj_matrix, history

# ── Инференс ──────────────────────────────────────────────────────────────────

def predict_gat_batch(model, history_frame, target_index, adj_matrix, scales, cfg):
    """Инференс GAT (сразу для всех объектов графа)."""
    history_frame = history_frame.sort_index()
    object_ids    = list(history_frame.columns)
    device        = resolve_device(cfg.device)
    model.eval()

    if len(history_frame) >= cfg.past_window:
        hist_slice = history_frame.iloc[-cfg.past_window:]
        values     = hist_slice.values.astype(np.float32) # [pw, N]
        hist_index = hist_slice.index
    else:
        missing    = cfg.past_window - len(history_frame)
        hist_index = pd.date_range(end=target_index[0] - pd.Timedelta(hours=1), periods=cfg.past_window, freq="h")
        sv = history_frame.values.astype(np.float32)
        pv = sv.mean(axis=0) if len(sv) else np.zeros(len(object_ids))
        values = np.pad(sv, ((missing, 0), (0, 0)), constant_values=pv)[-cfg.past_window:]

    horizon    = len(target_index)
    scale_logs = np.log1p(np.asarray([scales[oid] for oid in object_ids], dtype=np.float32)) # [N]

    # Encoder features
    enc_cov = make_time_covariates(hist_index, is_synthetic=0).values.astype(np.float32) # [pw, 9]
    hist_norm = np.log1p(values) / scale_logs # [pw, N]
    
    hist_norm_exp = hist_norm[..., np.newaxis] # [pw, N, 1]
    enc_cov_exp   = np.broadcast_to(enc_cov[:, np.newaxis, :], (cfg.past_window, len(object_ids), enc_cov.shape[-1]))
    x_input       = np.concatenate([hist_norm_exp, enc_cov_exp], axis=-1) # [pw, N, 10]
    x_tensor      = torch.from_numpy(x_input).unsqueeze(0).to(device) # [1, pw, N, 10]

    # Decoder features
    dec_cov = make_time_covariates(target_index, is_synthetic=0).values.astype(np.float32)
    dec_tensor = torch.from_numpy(dec_cov).unsqueeze(0).to(device) # [1, h, 9]

    with torch.no_grad():
        pred_scaled = model(x_tensor, dec_tensor, adj_matrix.to(device)).cpu().numpy()[0] # [h, N]

    # Денормализация
    pred = np.expm1(np.clip(pred_scaled, 0.0, 4.0) * scale_logs)
    max_pred = np.expm1(scale_logs) * 10
    pred = np.maximum(np.minimum(pred, max_pred), 0.0)

    return pd.DataFrame(pred, index=target_index, columns=object_ids)

# ── Rolling backtest ──────────────────────────────────────────────────────────

def run_gat_backtest(pivot_df, horizon, train_modes=("real_only", "real_plus_synth"), 
                     synth_days=30, min_train_hours=None, step_hours=None, max_folds=None, max_objects=None, cfg=None):
    cfg        = cfg or GATTrainConfig()
    pivot_df   = pivot_df.sort_index()
    object_ids = list(pivot_df.columns[:max_objects]) if max_objects else list(pivot_df.columns)
    pivot_df   = pivot_df[object_ids]

    prepended_synth = None
    if "real_plus_synth" in train_modes:
        from ...synthesis import prepend_synthetic_week
        prepended_synth = prepend_synthetic_week(pivot_df, seed=cfg.seed)

    folds = make_rolling_folds(len(pivot_df), horizon, min_train_hours, step_hours)
    if max_folds is not None: folds = folds[-max_folds:]
    if not folds: raise ValueError(f"Not enough data for GAT horizon={horizon}.")

    rows, histories, synth_val_rows = [], [], []

    for fold in folds:
        real_train_slice = pivot_df.iloc[fold["train_start"]:fold["train_end"]]
        test_real        = pivot_df.iloc[fold["test_start"]:fold["test_end"]]
        target_index     = test_real.index

        if fold["fold"] == 0 and prepended_synth is not None:
            val = validate_synthetic(real_train_slice, prepended_synth)
            val.update({"fold": fold["fold"], "horizon": horizon})
            synth_val_rows.append(val)

        active_objects = get_active_objects(real_train_slice, object_ids)

        for train_mode in train_modes:
            if train_mode == "real_plus_synth" and prepended_synth is not None:
                train_real = pd.concat([prepended_synth, real_train_slice]).sort_index()
                fold_synth = None
                if len(real_train_slice) < 72:
                    dyn_days   = get_synth_days(len(real_train_slice), base_synth_days=synth_days)
                    fold_synth = synthesize_from_train(real_train_slice, gen_days=dyn_days, seed=cfg.seed + fold["fold"] + horizon * 1000)
            else:
                train_real = real_train_slice
                fold_synth = None

            print(f"GAT h={horizon} fold={fold['fold']} mode={train_mode} train={len(train_real)}h test={len(test_real)}h")

            model, scales, adj_matrix, history = train_gat(
                train_real, fold_synth, horizon, cfg,
                scale_source=real_train_slice if train_mode == "real_plus_synth" else None,
            )

            for item in history:
                histories.append({**item, "horizon": horizon, "fold": fold["fold"], "train_mode": train_mode, "model": "GAT"})

            pred_df  = predict_gat_batch(model, real_train_slice, target_index, adj_matrix, scales, cfg)
            day_mask = get_day_mask(target_index)

            for oid in active_objects:
                y_true = test_real[oid].values[day_mask]
                y_pred = pred_df[oid].values[day_mask]
                if len(y_true) == 0: continue
                row = forecast_metrics(y_true, y_pred, model_name="GAT", horizon=horizon, train_mode=train_mode, object_id=oid, fold=fold["fold"])
                row.update({"test_start": target_index[0], "test_end": target_index[-1], "train_hours": len(real_train_slice), "test_data": "real"})
                rows.append(row)

    results      = pd.DataFrame(rows)
    summary      = summarize_results(results)
    histories_df = pd.DataFrame(histories)
    synth_val_df = pd.DataFrame(synth_val_rows)
    return results, summary, histories_df, synth_val_df