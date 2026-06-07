import random
from dataclasses import dataclass, replace

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
from ...windowing import make_time_covariates, make_correlation_adjacency
from ... import config


# ── Константы ─────────────────────────────────────────────────────────────────
NIGHT_HOURS   = frozenset({0, 1, 2, 3, 4, 5})
MIN_DAILY_PAX = 100


# ── Вспомогательные функции ──────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name="auto"):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def compute_object_scales(train_real):
    scales = {}
    for object_id in train_real.columns:
        values = train_real[object_id].values.astype(float)
        positive = values[values > 0]
        scale = float(np.mean(positive)) if len(positive) else 1.0
        scales[object_id] = max(scale, 1.0)
    return scales


def get_active_objects(train_slice, object_ids, min_daily_pax=MIN_DAILY_PAX):
    n_days = max(len(train_slice) / 24, 1.0)
    active = [
        oid for oid in object_ids
        if train_slice[oid].sum() / n_days >= min_daily_pax
    ]
    return active if len(active) >= 10 else list(object_ids)


def get_day_mask(target_index):
    return np.array([ts.hour not in NIGHT_HOURS for ts in target_index])


def build_adjacency_mask(pivot_df, top_k=8, min_corr=0.30):
    """Строит бинарную маску смежности [1, 1, N, N] с self-loops."""
    adj = make_correlation_adjacency(
        pivot_df, top_k=top_k, min_corr=min_corr, include_self=True,
    )
    mask = (adj.values > 0).astype(np.float32)
    np.fill_diagonal(mask, 1.0)
    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]


# ── Чекпойнты ────────────────────────────────────────────────────────────────

def save_gat_checkpoint(model, scales, cfg, adj_mask_np, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "scales": scales,
        "cfg": cfg.__dict__,
        "in_features": model.in_features,
        "future_cov_dim": model.future_cov_dim,
        "d_model": model.d_model,
        "horizon": model.horizon,
        "n_gat_layers": model.n_gat_layers,
        "adj_mask": adj_mask_np,
    }
    torch.save(checkpoint, filepath)
    print(f"GAT модель и скейлеры сохранены в: {filepath}")


def load_gat_checkpoint(filepath, device="auto"):
    device = resolve_device(device)
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    cfg = GATTrainConfig(**checkpoint["cfg"])
    model = GATForecaster(
        in_features=checkpoint["in_features"],
        future_cov_dim=checkpoint["future_cov_dim"],
        d_model=checkpoint["d_model"],
        n_heads=cfg.n_heads,
        n_gat_layers=checkpoint["n_gat_layers"],
        horizon=checkpoint["horizon"],
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    adj_mask = torch.from_numpy(checkpoint["adj_mask"]).unsqueeze(0).unsqueeze(0).to(device)
    return model, checkpoint["scales"], cfg, adj_mask


# ── Dataset ───────────────────────────────────────────────────────────────────

class GATWindowDataset(Dataset):
    """Один сэмпл = ВСЕ объекты для одного временного окна.

    Возвращает:
        encoder_x  : [N, past_window, n_features]   (1 value + 9 covariates)
        decoder_x  : [N, horizon,     cov_dim]       (9 covariates + 1 seasonal hint)
        y          : [N, horizon]
    """

    def __init__(
        self,
        frames,
        object_ids,
        scales,
        past_window,
        horizon,
        stride=1,
        is_synthetic=0,
    ):
        self.object_ids = list(object_ids)
        self.past_window = past_window
        self.horizon = horizon
        self.stride = max(int(stride), 1)
        self.samples = []          # (frame_idx, start)
        self.frames_np = []
        self.covariates_np = []

        self.scales_log = np.log1p(
            np.array([scales[oid] for oid in self.object_ids], dtype=np.float32)
        )  # [N]

        for frame in frames:
            if len(frame) == 0:
                continue
            frame = frame.sort_index()
            self.frames_np.append(frame[self.object_ids].values.astype(np.float32))  # [T, N]
            cov = make_time_covariates(frame.index, is_synthetic=is_synthetic)
            self.covariates_np.append(cov.values.astype(np.float32))  # [T, 9]

        for fi, fnp in enumerate(self.frames_np):
            max_start = len(fnp) - past_window - horizon
            if max_start < 0:
                continue
            for s in range(0, max_start + 1, self.stride):
                self.samples.append((fi, s))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fi, start = self.samples[idx]
        data = self.frames_np[fi]   # [T, N]
        cov = self.covariates_np[fi]  # [T, 9]
        sl = self.scales_log          # [N]

        h_s, h_e = start, start + self.past_window
        t_s, t_e = h_e, h_e + self.horizon

        history = data[h_s:h_e]     # [pw, N]
        target = data[t_s:t_e]      # [h,  N]

        # Сезонная подсказка — 24 ч назад
        s_s, s_e = t_s - 24, t_e - 24
        if s_s >= 0:
            seasonal = data[s_s:s_e]  # [h, N]
        else:
            fb = np.mean(history, axis=0)  # [N]
            avail = s_e
            if avail > 0:
                missing = self.horizon - avail
                seasonal = np.concatenate(
                    [np.tile(fb, (missing, 1)), data[0:avail]], axis=0
                )
            else:
                seasonal = np.tile(fb, (self.horizon, 1))

        # Нормализация
        x_val = (np.log1p(history) / sl[None, :]).T   # [N, pw]
        y_norm = (np.log1p(target) / sl[None, :]).T    # [N, h]
        sh = (np.log1p(seasonal) / sl[None, :]).T      # [N, h]

        # Encoder: [N, pw, 10]
        enc_cov = np.broadcast_to(
            cov[h_s:h_e][None, :, :],
            (len(self.object_ids), self.past_window, 9),
        )
        encoder_x = np.concatenate([x_val[:, :, None], enc_cov], axis=2)

        # Decoder: [N, h, 10]
        dec_cov = np.broadcast_to(
            cov[t_s:t_e][None, :, :],
            (len(self.object_ids), self.horizon, 9),
        )
        decoder_x = np.concatenate([sh[:, :, None], dec_cov], axis=2)

        return (
            torch.from_numpy(encoder_x.astype(np.float32)),
            torch.from_numpy(decoder_x.astype(np.float32)),
            torch.from_numpy(y_norm.astype(np.float32)),
        )


# ── Temporal Encoder ─────────────────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    """Conv1d с dilation + attention pooling по временной оси."""

    def __init__(self, in_features, d_model, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, d_model, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(d_model, d_model, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(d_model, d_model, 3, padding=4, dilation=4)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.pool_w = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x : [B*N, F, T]
        → : [B*N, d_model]
        """
        h = F.gelu(self.conv1(x))                    # [B*N, d, T]
        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        h = self.drop(h)

        h = F.gelu(self.conv2(h))
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        h = self.drop(h)

        h = F.gelu(self.conv3(h))
        h = self.norm3(h.transpose(1, 2)).transpose(1, 2)

        h = h.transpose(1, 2)                        # [B*N, T, d]
        w = torch.softmax(self.pool_w(h), dim=1)     # [B*N, T, 1]
        return (w * h).sum(dim=1)                     # [B*N, d]


# ── GAT Layer ────────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """Multi-head Graph Attention Layer (additive, оригинальный стиль GAT).

    Включает residual connection + LayerNorm + FFN.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.W = nn.Linear(d_model, d_model, bias=False)
        self.a_src = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.a_dst = nn.Parameter(torch.zeros(n_heads, self.d_head))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x, adj_mask):
        """
        x        : [B, N, D]
        adj_mask : [1, 1, N, N]  — бинарная маска смежности (1 = ребро)
        """
        B, N, D = x.shape
        residual = x

        h = self.W(x).reshape(B, N, self.n_heads, self.d_head)  # [B, N, H, d_h]

        # Additive attention scores
        e_src = torch.einsum("bnhd,hd->bnh", h, self.a_src)     # [B, N, H]
        e_dst = torch.einsum("bnhd,hd->bnh", h, self.a_dst)     # [B, N, H]
        e = e_src.unsqueeze(2) + e_dst.unsqueeze(1)              # [B, N, N, H]
        e = F.leaky_relu(e, negative_slope=0.2)

        # Переставляем головы на dim=1 → [B, H, N, N], чтобы совпало с adj_mask [1, 1, N, N]
        e = e.permute(0, 3, 1, 2)                                 # [B, H, N, N]

        # Маскируем несмежные пары
        e = e.masked_fill(adj_mask == 0, float("-inf"))           # [B, H, N, N]

        attn = torch.softmax(e, dim=-1)                            # [B, H, N, N]
        attn = attn.nan_to_num(0.0)                                # изолированные узлы
        attn = self.drop(attn)

        h_t = h.permute(0, 2, 1, 3)                               # [B, H, N, d_h]
        out = torch.matmul(attn, h_t)                              # [B, H, N, d_h]
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)            # [B, N, D]
        out = self.out_proj(out)

        x = self.norm1(residual + self.drop(out))
        x = self.norm2(x + self.ffn(x))
        return x


# ── GAT Forecaster ───────────────────────────────────────────────────────────

class GATForecaster(nn.Module):
    """GAT для глобального прогнозирования пассажиропотока.

    Пайплайн:
      1. Temporal Encoder (dilated Conv1d + attention pooling) → [B, N, d]
      2. GAT слои (графовое внимание на корреляционном графе) → [B, N, d]
      3. Output head с future covariate conditioning → [B, N, horizon]
    """

    def __init__(
        self,
        in_features: int = 10,
        future_cov_dim: int = 10,
        d_model: int = 128,
        n_heads: int = 4,
        n_gat_layers: int = 2,
        horizon: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.future_cov_dim = future_cov_dim
        self.d_model = d_model
        self.horizon = horizon
        self.n_gat_layers = n_gat_layers

        self.temporal_encoder = TemporalEncoder(in_features, d_model, dropout)

        self.gat_layers = nn.ModuleList([
            GATLayer(d_model, n_heads, dropout)
            for _ in range(n_gat_layers)
        ])

        self.output_head = nn.Sequential(
            nn.Linear(d_model + future_cov_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, encoder_x, adj_mask, decoder_x):
        """
        encoder_x : [B, N, pw, F]
        adj_mask  : [1, 1, N, N]
        decoder_x : [B, N, h,  cov_dim]
        returns   : [B, N, h]
        """
        B, N, T, F = encoder_x.shape

        # Temporal encoding: [B*N, F, T] → [B*N, d] → [B, N, d]
        x = encoder_x.reshape(B * N, T, F).permute(0, 2, 1)      # [B*N, F, T]
        x = self.temporal_encoder(x)                                # [B*N, d]
        x = x.reshape(B, N, -1)                                    # [B, N, d]

        # Graph Attention
        for gat in self.gat_layers:
            x = gat(x, adj_mask)                                    # [B, N, d]

        # Output head с future covariates
        x = x.unsqueeze(2).expand(-1, -1, self.horizon, -1)       # [B, N, h, d]
        head_in = torch.cat([x, decoder_x], dim=-1)                # [B, N, h, d+cov]
        out = self.output_head(head_in).squeeze(-1)                 # [B, N, h]
        return out


# ── Конфигурация ──────────────────────────────────────────────────────────────

@dataclass
class GATTrainConfig:
    past_window: int = 72
    d_model: int = 128
    n_heads: int = 4
    n_gat_layers: int = 2
    top_k_neighbors: int = 8
    min_corr: float = 0.30
    dropout: float = 0.1
    epochs: int = 16
    batch_size: int = 4         # число временных окон в батче (НЕ объектов!)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    window_stride: int = 1
    synthetic_window_stride: int = 12
    num_workers: int = 0
    device: str = "auto"
    seed: int = 42
    amp: bool = True


# ── Обучение ──────────────────────────────────────────────────────────────────

def train_gat(
    train_real, synthetic_train, horizon, cfg, adj_mask,
    pretrained_model=None, scale_source=None,
):
    """Обучает GAT. Если передан pretrained_model — дообучается поверх него."""
    set_seed(cfg.seed)
    object_ids = list(train_real.columns)
    scales = compute_object_scales(
        scale_source if scale_source is not None else train_real
    )

    frames = [train_real]
    strides = [cfg.window_stride]
    if synthetic_train is not None:
        frames.append(synthetic_train[object_ids])
        strides.append(cfg.synthetic_window_stride)

    datasets = [
        GATWindowDataset(
            [frame], object_ids, scales, cfg.past_window, horizon,
            stride=stride, is_synthetic=int(i > 0),
        )
        for i, (frame, stride) in enumerate(zip(frames, strides))
    ]
    usable = [d for d in datasets if len(d)]
    if not usable:
        raise ValueError(
            "No GAT training windows. Increase train size or reduce past_window."
        )
    train_dataset = (
        usable[0] if len(usable) == 1 else torch.utils.data.ConcatDataset(usable)
    )

    loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    sample_enc, sample_dec, _ = usable[0][0]
    device = resolve_device(cfg.device)
    adj_mask_dev = adj_mask.to(device)

    model = GATForecaster(
        in_features=sample_enc.shape[-1],
        future_cov_dim=sample_dec.shape[-1],
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_gat_layers=cfg.n_gat_layers,
        horizon=horizon,
        dropout=cfg.dropout,
    ).to(device)

    if pretrained_model is not None:
        model.load_state_dict(pretrained_model.state_dict())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.1,
    )
    loss_fn = nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    model.train()
    history = []
    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0.0
        n_seen = 0
        for enc_x, dec_x, y in loader:
            enc_x = enc_x.to(device=device, dtype=torch.float32)
            dec_x = dec_x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                pred = model(enc_x, adj_mask_dev, dec_x)  # [B, N, h]
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch = enc_x.shape[0]
            total_loss += float(loss.item()) * batch
            n_seen += batch

        scheduler.step()
        epoch_loss = total_loss / max(n_seen, 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"GAT h={horizon} epoch={epoch}/{cfg.epochs} loss={epoch_loss:.5f}")

    return model, scales, history


def train_gat_two_phase(
    train_real, synthetic_train, horizon, cfg, adj_mask, scale_source=None,
):
    """Фаза 1: предобучение на синтетике. Фаза 2: дообучение на реальных."""
    if synthetic_train is None:
        return train_gat(train_real, None, horizon, cfg, adj_mask,
                         scale_source=scale_source)

    synth_cfg = replace(cfg, window_stride=24)
    model, scales, history_phase1 = train_gat(
        synthetic_train[train_real.columns], None, horizon, synth_cfg, adj_mask,
        scale_source=scale_source,
    )

    finetune_epochs = max(3, cfg.epochs // 3)
    real_cfg = replace(cfg, epochs=finetune_epochs, window_stride=1)
    model, scales, history_phase2 = train_gat(
        train_real, None, horizon, real_cfg, adj_mask,
        pretrained_model=model, scale_source=scale_source,
    )

    for item in history_phase1:
        item["phase"] = "synth_pretrain"
    for item in history_phase2:
        item["phase"] = "real_finetune"

    return model, scales, history_phase1 + history_phase2


# ── Инференс ──────────────────────────────────────────────────────────────────

def predict_gat_batch(
    model, history_frame, target_index, scales, cfg, adj_mask, batch_size=None,
):
    """Прогноз всех объектов за один прямой проход GAT."""
    history_frame = history_frame.sort_index()
    object_ids = list(history_frame.columns)
    device = resolve_device(cfg.device)
    model.eval()
    adj_mask_dev = adj_mask.to(device)

    if len(history_frame) >= cfg.past_window:
        hist_slice = history_frame.iloc[-cfg.past_window:]
        values = hist_slice.values.astype(np.float32).T   # [N, pw]
        hist_index = hist_slice.index
    else:
        missing = cfg.past_window - len(history_frame)
        hist_index = pd.date_range(
            end=target_index[0] - pd.Timedelta(hours=1),
            periods=cfg.past_window, freq="h",
        )
        values = []
        for oid in object_ids:
            sv = history_frame[oid].values.astype(np.float32)
            pv = float(sv.mean()) if len(sv) else 0.0
            values.append(np.pad(sv, (missing, 0), constant_values=pv)[-cfg.past_window:])
        values = np.asarray(values, dtype=np.float32)      # [N, pw]

    horizon = len(target_index)
    N = len(object_ids)
    scale_logs = np.log1p(
        np.asarray([scales[oid] for oid in object_ids], dtype=np.float32)
    )  # [N]

    # Encoder: [N, pw, 10]
    x_val = (np.log1p(values) / scale_logs[:, None]).astype(np.float32)  # [N, pw]
    enc_cov = make_time_covariates(hist_index, is_synthetic=0).values.astype(np.float32)
    enc_cov_b = np.broadcast_to(enc_cov[None, :, :], (N, cfg.past_window, 9))
    encoder_x = np.concatenate([x_val[:, :, None], enc_cov_b], axis=2).astype(np.float32)

    # Seasonal hint
    if cfg.past_window >= 24:
        s_start = cfg.past_window - 24
        s_end = min(s_start + horizon, cfg.past_window)
        seasonal_pax = values[:, s_start:s_end]
        if seasonal_pax.shape[1] < horizon:
            reps = (horizon // seasonal_pax.shape[1]) + 1
            seasonal_pax = np.tile(seasonal_pax, (1, reps))[:, :horizon]
    else:
        mean_vals = values.mean(axis=1, keepdims=True)
        seasonal_pax = np.broadcast_to(mean_vals, (N, horizon)).copy()

    seasonal_hint = (np.log1p(seasonal_pax) / scale_logs[:, None]).astype(np.float32)

    # Decoder: [N, h, 10]
    dec_cov = make_time_covariates(target_index, is_synthetic=0).values.astype(np.float32)
    dec_cov_b = np.broadcast_to(dec_cov[None, :, :], (N, horizon, 9))
    decoder_x = np.concatenate([seasonal_hint[:, :, None], dec_cov_b], axis=2).astype(np.float32)

    # Прямой проход (все объекты за один раз)
    with torch.no_grad():
        enc_t = torch.from_numpy(encoder_x).unsqueeze(0).to(device=device, dtype=torch.float32)
        dec_t = torch.from_numpy(decoder_x).unsqueeze(0).to(device=device, dtype=torch.float32)
        pred_scaled = model(enc_t, adj_mask_dev, dec_t).cpu().numpy()[0]  # [N, h]

    # Денормализация
    pred = np.expm1(np.clip(pred_scaled, 0.0, 4.0) * scale_logs[:, None])
    max_pred = np.expm1(scale_logs[:, None]) * 10
    pred = np.maximum(np.minimum(pred, max_pred), 0.0)

    return pd.DataFrame(pred.T, index=target_index, columns=object_ids)


# ── Fast experiment ───────────────────────────────────────────────────────────

def _eval_starts(n_hours, train_hours, horizon, step_hours, max_eval_windows=None):
    starts = list(range(train_hours, n_hours - horizon + 1, step_hours))
    if max_eval_windows is not None and len(starts) > max_eval_windows:
        starts = starts[-max_eval_windows:]
    return starts


def run_gat_fast_experiment(
    pivot_df,
    horizons=(1, 24),
    train_modes=("real_only", "real_plus_synth"),
    synth_days=30,
    train_hours=96,
    eval_step_1h=1,
    eval_step_24h=24,
    max_eval_windows_1h=None,
    max_eval_windows_24h=None,
    max_objects=None,
    cfg=None,
):
    """Train once per mode, затем оценка на множестве реальных origin."""
    cfg = cfg or GATTrainConfig()
    horizons = sorted(set(int(h) for h in horizons))
    max_horizon = max(horizons)
    if train_hours < cfg.past_window + max_horizon:
        raise ValueError(
            f"train_hours={train_hours} too small for past_window={cfg.past_window} "
            f"and max_horizon={max_horizon}."
        )

    pivot_df = pivot_df.sort_index()
    object_ids = (
        list(pivot_df.columns[:max_objects]) if max_objects else list(pivot_df.columns)
    )
    pivot_df = pivot_df[object_ids]

    # Граф строим один раз по полным данным
    adj_mask = build_adjacency_mask(pivot_df, cfg.top_k_neighbors, cfg.min_corr)

    train_real = pivot_df.iloc[:train_hours]
    rows, histories, synth_validation_rows = [], [], []

    synthetic_train = None
    if "real_plus_synth" in train_modes:
        dynamic_days = get_synth_days(len(train_real), base_synth_days=synth_days)
        print(f"GAT fast: dynamic synth_days={dynamic_days} (train={len(train_real)}h)")
        synthetic_train = synthesize_from_train(
            train_real, gen_days=dynamic_days, seed=cfg.seed + max_horizon * 1000,
        )
        validation = validate_synthetic(train_real, synthetic_train)
        validation.update({"horizon": max_horizon, "protocol": "fast_single_train"})
        synth_validation_rows.append(validation)

    for train_mode in train_modes:
        active_synth = synthetic_train if train_mode == "real_plus_synth" else None
        print(
            f"GAT fast mode={train_mode} train={len(train_real)}h "
            f"max_horizon={max_horizon} objects={len(object_ids)}"
        )

        if train_mode == "real_plus_synth":
            model, scales, history = train_gat_two_phase(
                train_real, active_synth, max_horizon, cfg, adj_mask,
            )
        else:
            model, scales, history = train_gat(
                train_real, None, max_horizon, cfg, adj_mask,
            )

        adj_np = adj_mask.squeeze().numpy()
        model_path = config.OUTPUT_DIR / f"gat_model_fast_{train_mode}_h{max_horizon}.pt"
        save_gat_checkpoint(model, scales, cfg, adj_np, model_path)

        for item in history:
            histories.append({
                **item,
                "horizon": max_horizon,
                "train_mode": train_mode,
                "model": "GAT",
                "protocol": "fast_single_train",
                "train_hours": train_hours,
            })

        active_objects = get_active_objects(train_real, object_ids)
        print(f"  Активных объектов для метрик: {len(active_objects)} / {len(object_ids)}")

        for horizon in horizons:
            step_hours = (
                eval_step_1h if horizon == 1
                else (eval_step_24h if horizon == 24 else horizon)
            )
            max_eval = (
                max_eval_windows_1h if horizon == 1
                else (max_eval_windows_24h if horizon == 24 else None)
            )

            starts = _eval_starts(
                len(pivot_df), train_hours, horizon, step_hours, max_eval,
            )
            for eval_idx, start in enumerate(starts):
                forecast_index = pd.date_range(
                    start=pivot_df.index[start], periods=max_horizon, freq="h",
                )
                y_true = pivot_df.iloc[start: start + horizon]
                history_frame = pivot_df.iloc[:start]
                pred = predict_gat_batch(
                    model, history_frame, forecast_index, scales, cfg, adj_mask,
                ).iloc[:horizon]

                day_mask = get_day_mask(y_true.index)
                for oid in active_objects:
                    yt = y_true[oid].values[day_mask]
                    yp = pred[oid].values[day_mask]
                    if len(yt) == 0:
                        continue
                    row = forecast_metrics(
                        yt, yp,
                        model_name="GAT", horizon=horizon, train_mode=train_mode,
                        object_id=oid, fold=eval_idx,
                    )
                    row.update({
                        "test_start": y_true.index[0],
                        "test_end": y_true.index[-1],
                        "train_hours": train_hours,
                        "test_data": "real",
                        "protocol": "fast_single_train",
                    })
                    rows.append(row)

            print(
                f"GAT fast mode={train_mode} h={horizon}: "
                f"evaluated {len(starts)} real windows"
            )

    results = pd.DataFrame(rows)
    summary = summarize_results(results)
    histories = pd.DataFrame(histories)
    synth_validation = pd.DataFrame(synth_validation_rows)
    return results, summary, histories, synth_validation


# ── Rolling backtest ──────────────────────────────────────────────────────────

def run_gat_backtest(
    pivot_df,
    horizon,
    train_modes=("real_only", "real_plus_synth"),
    synth_days=30,
    min_train_hours=None,
    step_hours=None,
    max_folds=None,
    max_objects=None,
    cfg=None,
):
    """Rolling-origin backtest для GAT."""
    cfg = cfg or GATTrainConfig()
    pivot_df = pivot_df.sort_index()
    object_ids = (
        list(pivot_df.columns[:max_objects]) if max_objects else list(pivot_df.columns)
    )
    pivot_df = pivot_df[object_ids]

    # Граф строим один раз по полным данным (статическая структура)
    adj_mask = build_adjacency_mask(pivot_df, cfg.top_k_neighbors, cfg.min_corr)
    adj_np = adj_mask.squeeze().numpy()

    prepended_synth = None
    if "real_plus_synth" in train_modes:
        from ...synthesis import prepend_synthetic_week
        prepended_synth = prepend_synthetic_week(pivot_df, seed=cfg.seed)
        print(f"GAT synthetic week: {prepended_synth.index[0]} → {prepended_synth.index[-1]}")

    folds = make_rolling_folds(len(pivot_df), horizon, min_train_hours, step_hours)
    if max_folds is not None:
        folds = folds[-max_folds:]
    if not folds:
        raise ValueError(f"Not enough folds for GAT horizon={horizon}.")

    rows, histories, synth_validation_rows = [], [], []

    for fold in folds:
        real_train_slice = pivot_df.iloc[fold["train_start"]:fold["train_end"]]
        test_real = pivot_df.iloc[fold["test_start"]:fold["test_end"]]
        target_index = test_real.index

        if fold["fold"] == 0 and prepended_synth is not None:
            val = validate_synthetic(real_train_slice, prepended_synth)
            val.update({"fold": fold["fold"], "horizon": horizon})
            synth_validation_rows.append(val)

        active_objects = get_active_objects(real_train_slice, object_ids)

        for train_mode in train_modes:
            if train_mode == "real_plus_synth" and prepended_synth is not None:
                train_real = pd.concat([prepended_synth, real_train_slice]).sort_index()
                fold_synth = None
                if len(real_train_slice) < 72:
                    dyn_days = get_synth_days(
                        len(real_train_slice), base_synth_days=synth_days,
                    )
                    fold_synth = synthesize_from_train(
                        real_train_slice,
                        gen_days=dyn_days,
                        seed=cfg.seed + fold["fold"] + horizon * 1000,
                    )
            else:
                train_real = real_train_slice
                fold_synth = None

            print(
                f"GAT h={horizon} fold={fold['fold']} mode={train_mode} "
                f"train={len(train_real)}h (real={len(real_train_slice)}h) "
                f"test={len(test_real)}h active={len(active_objects)}/{len(object_ids)}"
            )

            if train_mode == "real_plus_synth":
                model, scales, history = train_gat_two_phase(
                    train_real, fold_synth, horizon, cfg, adj_mask,
                    scale_source=real_train_slice,
                )
            else:
                model, scales, history = train_gat(
                    real_train_slice, None, horizon, cfg, adj_mask,
                )

            if fold["fold"] == folds[-1]["fold"]:
                model_path = (
                    config.OUTPUT_DIR / f"gat_model_rolling_{train_mode}_h{horizon}.pt"
                )
                save_gat_checkpoint(model, scales, cfg, adj_np, model_path)

            for item in history:
                histories.append({
                    **item,
                    "horizon": horizon,
                    "fold": fold["fold"],
                    "train_mode": train_mode,
                    "model": "GAT",
                })

            pred_df = predict_gat_batch(
                model, real_train_slice, target_index, scales, cfg, adj_mask,
            )
            day_mask = get_day_mask(target_index)

            for oid in active_objects:
                yt = test_real[oid].values[day_mask]
                yp = pred_df[oid].values[day_mask]
                if len(yt) == 0:
                    continue
                row = forecast_metrics(
                    yt, yp,
                    model_name="GAT",
                    horizon=horizon,
                    train_mode=train_mode,
                    object_id=oid,
                    fold=fold["fold"],
                )
                row.update({
                    "test_start": target_index[0],
                    "test_end": target_index[-1],
                    "train_hours": len(real_train_slice),
                    "test_data": "real",
                })
                rows.append(row)

    results = pd.DataFrame(rows)
    summary = summarize_results(results)
    histories = pd.DataFrame(histories)
    synth_validation = pd.DataFrame(synth_validation_rows)
    return results, summary, histories, synth_validation