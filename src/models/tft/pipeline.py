import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from ...backtest import make_rolling_folds, summarize_results
from ...baseline_models import forecast_metrics
from ...synthesis import synthesize_from_train, validate_synthetic, get_synth_days
from ...windowing import make_time_covariates
from ... import config


# ── Константы ─────────────────────────────────────────────────────────────────
NIGHT_HOURS   = frozenset({0, 1, 2, 3, 4, 5})
MIN_DAILY_PAX = 50


# ── Вспомогательные функции (общие с TCN/N-BEATS) ────────────────────────────

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
        vals     = df[col].values.astype(float)
        positive = vals[vals > 0]
        scales[col] = max(float(np.mean(positive)) if len(positive) else 1.0, 1.0)
    return scales


def get_active_objects(train_slice: pd.DataFrame, object_ids: list,
                       min_daily_pax: float = MIN_DAILY_PAX) -> list:
    n_days = max(len(train_slice) / 24, 1.0)
    active = [oid for oid in object_ids
              if train_slice[oid].sum() / n_days >= min_daily_pax]
    return active if len(active) >= 10 else list(object_ids)


def get_day_mask(target_index) -> np.ndarray:
    return np.array([ts.hour not in NIGHT_HOURS for ts in target_index])


# ── Dataset ───────────────────────────────────────────────────────────────────

class TFTWindowDataset(Dataset):
    """
    Каждый сэмпл:
      encoder_x  : [past_window, n_enc]  — лог-нормализованный поток + 9 time covariates
      decoder_x  : [horizon,     n_dec]  — seasonal hint + 9 time covariates
      y          : [horizon]             — лог-нормализованный целевой поток

    n_enc = 1 + 9 = 10
    n_dec = 1 + 9 = 10  (seasonal hint + covariates)
    """

    def __init__(self, frames, object_ids, scales, past_window, horizon,
                 stride=1, is_synthetic=0):
        self.object_ids   = list(object_ids)
        self.past_window  = past_window
        self.horizon      = horizon
        self.stride       = max(int(stride), 1)
        self.samples      = []
        self.frames_np    = []
        self.covariates_np = []

        self.scales_log = np.log1p(
            np.array([scales[oid] for oid in self.object_ids], dtype=np.float32)
        )  # [N_objects]

        for frame in frames:
            if len(frame) == 0:
                continue
            frame = frame.sort_index()
            self.frames_np.append(frame[self.object_ids].values.astype(np.float32))
            cov = make_time_covariates(frame.index, is_synthetic=is_synthetic)
            self.covariates_np.append(cov.values.astype(np.float32))

        for fi, fnp in enumerate(self.frames_np):
            max_start = len(fnp) - past_window - horizon
            if max_start < 0:
                continue
            for ci in range(len(self.object_ids)):
                for s in range(0, max_start + 1, self.stride):
                    self.samples.append((fi, ci, s))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fi, ci, start = self.samples[idx]
        data = self.frames_np[fi]
        cov  = self.covariates_np[fi]
        sl   = self.scales_log[ci]

        h_s, h_e = start, start + self.past_window
        t_s, t_e = h_e, h_e + self.horizon

        history = data[h_s:h_e, ci]   # [past_window]
        target  = data[t_s:t_e, ci]   # [horizon]

        # Seasonal hint: те же часы вчера
        ss, se = t_s - 24, t_e - 24
        if ss >= 0:
            seasonal = data[ss:se, ci]
        else:
            fb = float(np.mean(history))
            avail = se
            if avail > 0:
                seasonal = np.concatenate([np.full(self.horizon - avail, fb),
                                           data[0:avail, ci]])
            else:
                seasonal = np.full(self.horizon, fb)

        # Нормализация
        x_val  = (np.log1p(history)  / sl).reshape(-1, 1).astype(np.float32)  # [pw,1]
        y_norm = (np.log1p(target)   / sl).astype(np.float32)                  # [h]
        sh     = (np.log1p(seasonal) / sl).reshape(-1, 1).astype(np.float32)  # [h,1]

        enc_cov  = cov[h_s:h_e]   # [pw, 9]
        dec_cov  = cov[t_s:t_e]   # [h,  9]

        encoder_x = np.concatenate([x_val, enc_cov], axis=1)          # [pw, 10]
        decoder_x = np.concatenate([sh,    dec_cov], axis=1)          # [h,  10]

        return (
            torch.from_numpy(encoder_x),
            torch.from_numpy(decoder_x),
            torch.from_numpy(y_norm),
        )


# ── TFT строительные блоки ───────────────────────────────────────────────────

class GRN(nn.Module):
    """Gated Residual Network — ключевой блок TFT."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1, context_dim: int = 0):
        super().__init__()
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        self.fc1  = nn.Linear(input_dim + context_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, context=None):
        residual = x if self.skip is None else self.skip(x)
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        h     = F.elu(self.fc1(x))
        h     = self.drop(h)
        out   = self.fc2(h)
        gate  = torch.sigmoid(self.gate(h))
        out   = gate * out + (1 - gate) * residual
        return self.norm(out)


class VariableSelectionNetwork(nn.Module):
    """VSN — взвешивает вклад каждой входной переменной."""

    def __init__(self, n_vars: int, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.grn_vars = nn.ModuleList([
            GRN(input_dim, hidden_dim, hidden_dim, dropout) for _ in range(n_vars)
        ])
        self.grn_flat = GRN(n_vars * input_dim, hidden_dim, n_vars, dropout)
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [B, T, n_vars, input_dim]  or  [B, n_vars, input_dim]
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(1)                         # [B,1,n_vars,d]

        B, T, V, D = x.shape
        flat   = x.reshape(B, T, V * D)
        weight = self.softmax(self.grn_flat(flat))     # [B,T,V]
        weight = weight.unsqueeze(-1)                  # [B,T,V,1]

        processed = torch.stack([self.grn_vars[i](x[:, :, i, :])
                                 for i in range(V)], dim=2)  # [B,T,V,hidden]
        out = (weight * processed).sum(dim=2)          # [B,T,hidden]

        if squeeze:
            out = out.squeeze(1)
        return out, weight.squeeze(-1)


class TFTForecaster(nn.Module):
    """
    Упрощённый TFT для глобального прогнозирования:
      • VSN для encoder и decoder входов
      • LSTM encoder + decoder
      • Multi-head self-attention на encoder выходах
      • Gated skip connection + LayerNorm перед head
    """

    def __init__(
        self,
        n_enc_vars:    int   = 10,   # число признаков encoder (лог-поток + 9 ковариат)
        n_dec_vars:    int   = 10,   # число признаков decoder (seasonal + 9 ковариат)
        d_model:       int   = 128,
        n_heads:       int   = 4,
        lstm_layers:   int   = 2,
        horizon:       int   = 24,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.d_model    = d_model
        self.horizon    = horizon
        self.n_enc_vars = n_enc_vars
        self.n_dec_vars = n_dec_vars

        # VSN — каждая переменная dim=1, total input_dim = 1 per var
        self.enc_vsn = VariableSelectionNetwork(n_enc_vars, 1, d_model, dropout)
        self.dec_vsn = VariableSelectionNetwork(n_dec_vars, 1, d_model, dropout)

        # LSTM encoder
        self.enc_lstm = nn.LSTM(d_model, d_model, lstm_layers,
                                batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        # LSTM decoder
        self.dec_lstm = nn.LSTM(d_model, d_model, lstm_layers,
                                batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)

        # Gated skip после LSTM
        self.enc_gate = GRN(d_model, d_model, d_model, dropout)
        self.dec_gate = GRN(d_model, d_model, d_model, dropout)

        # Interpretable Multi-head Attention на encoder
        self.attn      = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_grn  = GRN(d_model, d_model, d_model, dropout)

        # Output head
        self.output_grn  = GRN(d_model, d_model, d_model, dropout)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, enc_x, dec_x):
        """
        enc_x : [B, past_window, n_enc_vars]
        dec_x : [B, horizon,     n_dec_vars]
        returns: [B, horizon]
        """
        B = enc_x.size(0)

        # ── VSN ───────────────────────────────────────────────────────────────
        # Раскладываем каждую переменную в отдельное измерение [B,T,V,1]
        enc_in = enc_x.unsqueeze(-1)          # [B,pw,V_enc,1]
        dec_in = dec_x.unsqueeze(-1)          # [B,h, V_dec,1]

        enc_vsn, _ = self.enc_vsn(enc_in)    # [B,pw,d]
        dec_vsn, _ = self.dec_vsn(dec_in)    # [B,h, d]

        # ── LSTM encoder ──────────────────────────────────────────────────────
        enc_out, (h_n, c_n) = self.enc_lstm(enc_vsn)   # [B,pw,d], hidden
        enc_out = self.enc_gate(enc_out)                # gated skip

        # ── LSTM decoder (инициализируем состоянием encoder) ──────────────────
        dec_out, _ = self.dec_lstm(dec_vsn, (h_n, c_n))  # [B,h,d]
        dec_out    = self.dec_gate(dec_out)               # gated skip

        # ── Multi-head Attention: decoder запрашивает у encoder ───────────────
        attn_out, _ = self.attn(
            query = dec_out,    # [B,h, d]
            key   = enc_out,    # [B,pw,d]
            value = enc_out,
        )
        attn_out = self.attn_norm(attn_out + dec_out)   # residual + norm
        attn_out = self.attn_grn(attn_out)

        # ── Output ────────────────────────────────────────────────────────────
        out = self.output_grn(attn_out)                 # [B,h,d]
        out = self.output_proj(out).squeeze(-1)         # [B,h]
        return out


# ── Конфигурация ──────────────────────────────────────────────────────────────

@dataclass
class TFTTrainConfig:
    past_window:   int   = 72     # 3 дня истории
    d_model:       int   = 128
    n_heads:       int   = 4
    lstm_layers:   int   = 2
    dropout:       float = 0.1
    epochs:        int   = 20
    batch_size:    int   = 1024
    learning_rate: float = 1e-3
    weight_decay:  float = 1e-4
    window_stride: int   = 1
    synthetic_window_stride: int = 24
    num_workers:   int   = 0
    device:        str   = "auto"
    seed:          int   = 42
    amp:           bool  = True


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_tft_checkpoint(model, scales, cfg, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "scales":           scales,
        "cfg":              cfg.__dict__,
        "n_enc_vars":       model.n_enc_vars,
        "n_dec_vars":       model.n_dec_vars,
        "d_model":          model.d_model,
        "horizon":          model.horizon,
    }, filepath)
    print(f"TFT модель сохранена: {filepath}")


def load_tft_checkpoint(filepath, device="auto"):
    device = resolve_device(device)
    ckpt   = torch.load(filepath, map_location=device)
    cfg    = TFTTrainConfig(**ckpt["cfg"])
    model  = TFTForecaster(
        n_enc_vars  = ckpt["n_enc_vars"],
        n_dec_vars  = ckpt["n_dec_vars"],
        d_model     = ckpt["d_model"],
        n_heads     = cfg.n_heads,
        lstm_layers = cfg.lstm_layers,
        horizon     = ckpt["horizon"],
        dropout     = cfg.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["scales"], cfg


# ── Обучение ──────────────────────────────────────────────────────────────────

def train_tft(train_real, synthetic_train, horizon, cfg, scale_source=None):
    set_seed(cfg.seed)
    object_ids = list(train_real.columns)
    scales     = compute_object_scales(scale_source if scale_source is not None else train_real)

    frames  = [train_real]
    strides = [cfg.window_stride]
    if synthetic_train is not None:
        frames.append(synthetic_train[object_ids])
        strides.append(cfg.synthetic_window_stride)

    datasets = [
        TFTWindowDataset([f], object_ids, scales, cfg.past_window, horizon,
                         stride=s, is_synthetic=int(i > 0))
        for i, (f, s) in enumerate(zip(frames, strides))
    ]
    usable = [d for d in datasets if len(d)]
    if not usable:
        raise ValueError("No TFT training windows. Increase train size or reduce past_window.")

    train_ds = usable[0] if len(usable) == 1 else torch.utils.data.ConcatDataset(usable)
    loader   = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())

    # Определяем размерности из первого сэмпла
    sample_enc, sample_dec, _ = usable[0][0]
    n_enc = sample_enc.shape[-1]   # 10
    n_dec = sample_dec.shape[-1]   # 10

    device = resolve_device(cfg.device)
    model  = TFTForecaster(
        n_enc_vars  = n_enc,
        n_dec_vars  = n_dec,
        d_model     = cfg.d_model,
        n_heads     = cfg.n_heads,
        lstm_layers = cfg.lstm_layers,
        horizon     = horizon,
        dropout     = cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    # Планировщик: cosine annealing улучшает сходимость TFT
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.1
    )
    loss_fn = nn.HuberLoss()
    scaler  = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    model.train()
    history = []
    for epoch in range(1, cfg.epochs + 1):
        total_loss, n_seen = 0.0, 0
        for enc_x, dec_x, y in loader:
            enc_x = enc_x.to(device=device, dtype=torch.float32)
            dec_x = dec_x.to(device=device, dtype=torch.float32)
            y     = y.to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                pred = model(enc_x, dec_x)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * enc_x.shape[0]
            n_seen     += enc_x.shape[0]

        scheduler.step()
        epoch_loss = total_loss / max(n_seen, 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"TFT h={horizon} epoch={epoch}/{cfg.epochs} loss={epoch_loss:.5f}")

    return model, scales, history


# ── Инференс ──────────────────────────────────────────────────────────────────

def predict_tft_batch(model, history_frame, target_index, scales, cfg, batch_size=None):
    """Батчевый инференс TFT для всех объектов."""
    history_frame = history_frame.sort_index()
    object_ids    = list(history_frame.columns)
    batch_size    = batch_size or cfg.batch_size
    device        = resolve_device(cfg.device)
    model.eval()

    # ── История ───────────────────────────────────────────────────────────────
    if len(history_frame) >= cfg.past_window:
        hist_slice = history_frame.iloc[-cfg.past_window:]
        values     = hist_slice.values.astype(np.float32).T          # [N, pw]
        hist_index = hist_slice.index
    else:
        missing    = cfg.past_window - len(history_frame)
        hist_index = pd.date_range(
            end=target_index[0] - pd.Timedelta(hours=1),
            periods=cfg.past_window, freq="h",
        )
        values = []
        for oid in object_ids:
            sv = history_frame[oid].values.astype(np.float32)
            pv = float(sv.mean()) if len(sv) else 0.0
            values.append(np.pad(sv, (missing, 0), constant_values=pv)[-cfg.past_window:])
        values = np.asarray(values, dtype=np.float32)

    horizon    = len(target_index)
    scale_logs = np.log1p(
        np.asarray([scales[oid] for oid in object_ids], dtype=np.float32)
    ).reshape(-1, 1)   # [N, 1]

    # ── Encoder features ──────────────────────────────────────────────────────
    enc_cov  = make_time_covariates(hist_index, is_synthetic=0).values.astype(np.float32)
    # [N, pw, 1] + [N, pw, 9] = [N, pw, 10]
    x_val    = (np.log1p(values) / scale_logs)[:, :, None]           # [N,pw,1]
    enc_cov_b = np.broadcast_to(enc_cov[None], (len(object_ids), cfg.past_window, enc_cov.shape[1]))
    enc_input = np.concatenate([x_val, enc_cov_b], axis=2).astype(np.float32)   # [N,pw,10]

    # ── Decoder features ──────────────────────────────────────────────────────
    dec_cov  = make_time_covariates(target_index, is_synthetic=0).values.astype(np.float32)
    s_start  = cfg.past_window - 24
    seasonal = values[:, s_start: s_start + horizon]                  # [N, h]
    sh       = (np.log1p(seasonal) / scale_logs)[:, :, None]          # [N,h,1]
    dec_cov_b = np.broadcast_to(dec_cov[None], (len(object_ids), horizon, dec_cov.shape[1]))
    dec_input = np.concatenate([sh, dec_cov_b], axis=2).astype(np.float32)      # [N,h,10]

    # ── Батчевый прогноз ──────────────────────────────────────────────────────
    preds = []
    with torch.no_grad():
        for start in range(0, len(object_ids), batch_size):
            stop  = start + batch_size
            enc_b = torch.from_numpy(enc_input[start:stop]).to(device=device, dtype=torch.float32)
            dec_b = torch.from_numpy(dec_input[start:stop]).to(device=device, dtype=torch.float32)
            pred_scaled = model(enc_b, dec_b).cpu().numpy()           # [B, h]

            # Клиппинг + денормализация (аналогично TCN/N-BEATS)
            pred     = np.expm1(np.clip(pred_scaled, 0.0, 4.0) * scale_logs[start:stop])
            max_pred = np.expm1(scale_logs[start:stop]) * 10
            preds.append(np.maximum(np.minimum(pred, max_pred), 0.0))

    pred_matrix = np.vstack(preds)   # [N, h]
    return pd.DataFrame(pred_matrix.T, index=target_index, columns=object_ids)


# ── Rolling backtest ──────────────────────────────────────────────────────────

def run_tft_backtest(
    pivot_df,
    horizon,
    train_modes     = ("real_only", "real_plus_synth"),
    synth_days      = 30,
    min_train_hours = None,
    step_hours      = None,
    max_folds       = None,
    max_objects     = None,
    cfg             = None,
):
    cfg        = cfg or TFTTrainConfig()
    pivot_df   = pivot_df.sort_index()
    object_ids = list(pivot_df.columns[:max_objects]) if max_objects else list(pivot_df.columns)
    pivot_df   = pivot_df[object_ids]

    # Синтетическая неделя до реальных данных
    prepended_synth = None
    if "real_plus_synth" in train_modes:
        from ...synthesis import prepend_synthetic_week
        prepended_synth = prepend_synthetic_week(pivot_df, seed=cfg.seed)
        print(f"TFT synthetic week: {prepended_synth.index[0]} → {prepended_synth.index[-1]}")

    folds = make_rolling_folds(len(pivot_df), horizon, min_train_hours, step_hours)
    if max_folds is not None:
        folds = folds[-max_folds:]
    if not folds:
        raise ValueError(f"Not enough data for TFT horizon={horizon}.")

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
                    fold_synth = synthesize_from_train(
                        real_train_slice, gen_days=dyn_days,
                        seed=cfg.seed + fold["fold"] + horizon * 1000,
                    )
            else:
                train_real = real_train_slice
                fold_synth = None

            print(
                f"TFT h={horizon} fold={fold['fold']} mode={train_mode} "
                f"train={len(train_real)}h (real={len(real_train_slice)}h) "
                f"test={len(test_real)}h active={len(active_objects)}/{len(object_ids)}"
            )

            model, scales, history = train_tft(
                train_real, fold_synth, horizon, cfg,
                scale_source=real_train_slice if train_mode == "real_plus_synth" else None,
            )

            # Сохраняем модель последнего фолда
            if fold["fold"] == folds[-1]["fold"]:
                model_path = config.OUTPUT_DIR / f"tft_model_rolling_{train_mode}_h{horizon}.pt"
                save_tft_checkpoint(model, scales, cfg, model_path)

            for item in history:
                histories.append({**item, "horizon": horizon,
                                   "fold": fold["fold"], "train_mode": train_mode, "model": "TFT"})

            pred_df  = predict_tft_batch(model, real_train_slice, target_index, scales, cfg)
            day_mask = get_day_mask(target_index)

            for oid in active_objects:
                y_true = test_real[oid].values[day_mask]
                y_pred = pred_df[oid].values[day_mask]
                if len(y_true) == 0:
                    continue
                row = forecast_metrics(y_true, y_pred, model_name="TFT",
                                       horizon=horizon, train_mode=train_mode,
                                       object_id=oid, fold=fold["fold"])
                row.update({
                    "test_start":  target_index[0],
                    "test_end":    target_index[-1],
                    "train_hours": len(real_train_slice),
                    "test_data":   "real",
                })
                rows.append(row)

    results      = pd.DataFrame(rows)
    summary      = summarize_results(results)
    histories_df = pd.DataFrame(histories)
    synth_val_df = pd.DataFrame(synth_val_rows)
    return results, summary, histories_df, synth_val_df