import random
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from ...backtest import make_rolling_folds, summarize_results
from ...baseline_models import forecast_metrics
from ...synthesis import synthesize_from_train, validate_synthetic, get_synth_days
from ...windowing import make_time_covariates
from ... import config


NIGHT_HOURS = frozenset({0, 1, 2, 3, 4, 5})
MIN_DAILY_PAX = 100


# ──────────────────────── утилиты ────────────────────────


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


# ──────────────────────── чекпойнты ────────────────────────


def save_nbeats_checkpoint(model, scales, cfg, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "scales": scales,
        "cfg": {
            k: v for k, v in cfg.__dict__.items()
            if k != "num_harmonics" or v is not None
        },
        "past_window": model.past_window,
        "horizon": model.horizon,
        "future_cov_dim": model.future_cov_dim,
        "architecture": model.architecture,
    }
    torch.save(checkpoint, filepath)
    print(f"Модель и скейлеры сохранены в: {filepath}")


def load_nbeats_checkpoint(filepath, device="auto"):
    device = resolve_device(device)
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    raw_cfg = checkpoint["cfg"]
    if "num_harmonics" not in raw_cfg:
        raw_cfg["num_harmonics"] = None
    cfg = NBEATSTrainConfig(**raw_cfg)

    num_harmonics = cfg.num_harmonics if cfg.num_harmonics and cfg.num_harmonics > 0 else None

    model = NBEATSForecaster(
        past_window=checkpoint["past_window"],
        horizon=checkpoint["horizon"],
        future_cov_dim=checkpoint["future_cov_dim"],
        architecture=checkpoint["architecture"],
        num_stacks=cfg.num_stacks,
        num_blocks_per_stack=cfg.num_blocks_per_stack,
        layer_sizes=cfg.layer_sizes,
        polynomial_degree=cfg.polynomial_degree,
        num_harmonics=num_harmonics,
        use_covariates=cfg.use_covariates,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["scales"], cfg


# ──────────────────────── датасет ────────────────────────


class NBEATSWindowDataset(Dataset):
    """Один сэмпл = (x_past, future_cov, y) для одного объекта."""

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
        self.samples = []
        self.frames_np = []
        self.covariates_np = []

        for frame in frames:
            if len(frame) == 0:
                continue
            frame = frame.sort_index()
            self.frames_np.append(frame[self.object_ids].values.astype(np.float32))
            cov = make_time_covariates(frame.index, is_synthetic=is_synthetic)
            self.covariates_np.append(cov.values.astype(np.float32))

        self.scales_log_np = np.log1p(
            np.array([scales[oid] for oid in self.object_ids], dtype=np.float32)
        )

        for frame_idx, frame_np in enumerate(self.frames_np):
            max_start = len(frame_np) - past_window - horizon
            if max_start < 0:
                continue
            for col_idx in range(len(self.object_ids)):
                for start in range(0, max_start + 1, self.stride):
                    self.samples.append((frame_idx, col_idx, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_idx, col_idx, start = self.samples[idx]
        data_np = self.frames_np[frame_idx]
        cov_np = self.covariates_np[frame_idx]
        scale_log = self.scales_log_np[col_idx]

        h_start, h_end = start, start + self.past_window
        t_start, t_end = h_end, h_end + self.horizon

        history = data_np[h_start:h_end, col_idx]
        target = data_np[t_start:t_end, col_idx]

        # сезонная подсказка — 24 ч назад от целевого окна
        s_start = t_start - 24
        s_end = t_end - 24
        if s_start >= 0:
            seasonal_values = data_np[s_start:s_end, col_idx]
        else:
            fallback_mean = np.mean(history)
            available_len = s_end
            if available_len > 0:
                missing_len = self.horizon - available_len
                seasonal_values = np.concatenate([
                    np.full(missing_len, fallback_mean),
                    data_np[0:available_len, col_idx],
                ])
            else:
                seasonal_values = np.full(self.horizon, fallback_mean)

        seasonal_hint = (np.log1p(seasonal_values) / scale_log).reshape(-1, 1)

        x_past = (np.log1p(history) / scale_log).astype(np.float32)
        future_cov = cov_np[t_start:t_end]
        future_cov_with_hint = np.concatenate([future_cov, seasonal_hint], axis=1)

        return (
            torch.from_numpy(x_past),
            torch.from_numpy(future_cov_with_hint.astype(np.float32)),
            torch.from_numpy((np.log1p(target) / scale_log).astype(np.float32)),
        )


# ──────────────────────── блоки N-BEATS ────────────────────────


class GenericBlock(nn.Module):
    """Универсальный блок: FC → θ → линейный бэккаст/форкаст."""

    def __init__(self, past_window, horizon, layer_sizes, dropout=0.0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(past_window, layer_sizes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_sizes, layer_sizes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_sizes, layer_sizes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_sizes, layer_sizes),
            nn.ReLU(),
        )
        self.backcast_linear = nn.Linear(layer_sizes, past_window)
        self.forecast_linear = nn.Linear(layer_sizes, horizon)

    def forward(self, x):
        h = self.fc(x)
        return self.backcast_linear(h), self.forecast_linear(h)


class TrendBlock(nn.Module):
    """Блок тренда: FC → θ → полиномиальный базис."""

    def __init__(self, past_window, horizon, layer_sizes, polynomial_degree=3, dropout=0.0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(past_window, layer_sizes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_sizes, layer_sizes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_sizes, layer_sizes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_sizes, layer_sizes),
            nn.ReLU(),
        )
        self.backcast_theta = nn.Linear(layer_sizes, polynomial_degree + 1)
        self.forecast_theta = nn.Linear(layer_sizes, polynomial_degree + 1)

        t_back = torch.arange(past_window, dtype=torch.float32) / past_window
        t_for = torch.arange(past_window, past_window + horizon, dtype=torch.float32) / past_window

        self.register_buffer(
            "backcast_basis",
            torch.stack([t_back ** p for p in range(polynomial_degree + 1)]),
        )
        self.register_buffer(
            "forecast_basis",
            torch.stack([t_for ** p for p in range(polynomial_degree + 1)]),
        )

    def forward(self, x):
        h = self.fc(x)
        theta_b = self.backcast_theta(h)
        theta_f = self.forecast_theta(h)
        backcast = torch.matmul(theta_b, self.backcast_basis)
        forecast = torch.matmul(theta_f, self.forecast_basis)
        return backcast, forecast


class SeasonalityBlock(nn.Module):
    """Блок сезонности: FC → θ → гармонический базис (период 24 ч)."""

    def __init__(self, past_window, horizon, layer_sizes, num_harmonics=None, dropout=0.0):
        super().__init__()
        num_harmonics = num_harmonics or max(horizon, 12)

        self.fc = nn.Sequential(
            nn.Linear(past_window, layer_sizes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_sizes, layer_sizes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_sizes, layer_sizes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_sizes, layer_sizes),
            nn.ReLU(),
        )

        basis_size = 2 * num_harmonics
        self.backcast_theta = nn.Linear(layer_sizes, basis_size)
        self.forecast_theta = nn.Linear(layer_sizes, basis_size)

        t_back = torch.arange(past_window, dtype=torch.float32)
        t_for = torch.arange(past_window, past_window + horizon, dtype=torch.float32)

        back_basis, for_basis = [], []
        for i in range(1, num_harmonics + 1):
            freq = 2.0 * np.pi * i / 24.0
            back_basis.extend([torch.sin(freq * t_back), torch.cos(freq * t_back)])
            for_basis.extend([torch.sin(freq * t_for), torch.cos(freq * t_for)])

        self.register_buffer("backcast_basis", torch.stack(back_basis))
        self.register_buffer("forecast_basis", torch.stack(for_basis))

    def forward(self, x):
        h = self.fc(x)
        theta_b = self.backcast_theta(h)
        theta_f = self.forecast_theta(h)
        backcast = torch.matmul(theta_b, self.backcast_basis)
        forecast = torch.matmul(theta_f, self.forecast_basis)
        return backcast, forecast


# ──────────────────────── модель ────────────────────────


class NBEATSForecaster(nn.Module):
    """N-BEATS с опциональным conditioning на будущие ковариаты.

    Архитектуры:
      • «generic» — все блоки GenericBlock
      • «interpretable» — первые стеки TrendBlock, остальные SeasonalityBlock
    """

    def __init__(
        self,
        past_window: int,
        horizon: int,
        future_cov_dim: int = 0,
        architecture: str = "generic",
        num_stacks: int = 3,
        num_blocks_per_stack: int = 3,
        layer_sizes: int = 256,
        polynomial_degree: int = 3,
        num_harmonics: Optional[int] = None,
        use_covariates: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.past_window = past_window
        self.horizon = horizon
        self.future_cov_dim = future_cov_dim
        self.architecture = architecture
        self.use_covariates = use_covariates and future_cov_dim > 0

        self.stacks = nn.ModuleList()

        if architecture == "generic":
            for _ in range(num_stacks):
                self.stacks.append(
                    nn.ModuleList([
                        GenericBlock(past_window, horizon, layer_sizes, dropout)
                        for _ in range(num_blocks_per_stack)
                    ])
                )
        elif architecture == "interpretable":
            n_trend = max(1, num_stacks // 2)
            n_season = num_stacks - n_trend
            for _ in range(n_trend):
                self.stacks.append(
                    nn.ModuleList([
                        TrendBlock(past_window, horizon, layer_sizes, polynomial_degree, dropout)
                        for _ in range(num_blocks_per_stack)
                    ])
                )
            for _ in range(n_season):
                self.stacks.append(
                    nn.ModuleList([
                        SeasonalityBlock(past_window, horizon, layer_sizes, num_harmonics, dropout)
                        for _ in range(num_blocks_per_stack)
                    ])
                )
        else:
            raise ValueError(f"Unknown architecture: {architecture}. Use 'generic' or 'interpretable'.")

        if self.use_covariates:
            self.cov_head = nn.Sequential(
                nn.Linear(future_cov_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

    def forward(self, x_past, future_cov=None):
        """
        x_past:       [B, past_window]
        future_cov:   [B, horizon, future_cov_dim]  (optional)
        returns:      [B, horizon]
        """
        residual = x_past
        forecast_total = torch.zeros(x_past.shape[0], self.horizon, device=x_past.device)

        for stack in self.stacks:
            stack_forecast = torch.zeros(x_past.shape[0], self.horizon, device=x_past.device)
            for block in stack:
                backcast, forecast = block(residual)
                residual = residual - backcast
                stack_forecast = stack_forecast + forecast
            forecast_total = forecast_total + stack_forecast

        if self.use_covariates and future_cov is not None:
            cov_adj = self.cov_head(future_cov).squeeze(-1)  # [B, horizon]
            forecast_total = forecast_total + cov_adj

        return forecast_total


# ──────────────────────── конфиг ────────────────────────


@dataclass
class NBEATSTrainConfig:
    past_window: int = 72
    epochs: int = 15
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    architecture: str = "generic"
    num_stacks: int = 3
    num_blocks_per_stack: int = 3
    layer_sizes: int = 256
    polynomial_degree: int = 3
    num_harmonics: Optional[int] = None
    use_covariates: bool = True
    dropout: float = 0.10
    window_stride: int = 1
    synthetic_window_stride: int = 24
    num_workers: int = 0
    device: str = "auto"
    seed: int = 42
    amp: bool = True


# ──────────────────────── обучение ────────────────────────


def train_nbeats(train_real, synthetic_train, horizon, cfg, pretrained_model=None, scale_source=None):
    """Обучает N-BEATS. Если передан pretrained_model — дообучается поверх него."""
    set_seed(cfg.seed)
    object_ids = list(train_real.columns)
    scales = compute_object_scales(scale_source if scale_source is not None else train_real)

    frames = [train_real]
    strides = [cfg.window_stride]
    if synthetic_train is not None:
        frames.append(synthetic_train[object_ids])
        strides.append(cfg.synthetic_window_stride)

    datasets = [
        NBEATSWindowDataset(
            [frame], object_ids, scales, cfg.past_window, horizon,
            stride=stride, is_synthetic=int(i > 0),
        )
        for i, (frame, stride) in enumerate(zip(frames, strides))
    ]
    usable = [d for d in datasets if len(d)]
    if not usable:
        raise ValueError("No N-BEATS training windows were created. Increase train size or reduce past_window.")
    train_dataset = usable[0] if len(usable) == 1 else torch.utils.data.ConcatDataset(usable)

    loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    sample_x, sample_future_cov, _ = usable[0][0]
    device = resolve_device(cfg.device)

    num_harmonics = cfg.num_harmonics if cfg.num_harmonics and cfg.num_harmonics > 0 else None

    model = NBEATSForecaster(
        past_window=cfg.past_window,
        horizon=horizon,
        future_cov_dim=sample_future_cov.shape[1],
        architecture=cfg.architecture,
        num_stacks=cfg.num_stacks,
        num_blocks_per_stack=cfg.num_blocks_per_stack,
        layer_sizes=cfg.layer_sizes,
        polynomial_degree=cfg.polynomial_degree,
        num_harmonics=num_harmonics,
        use_covariates=cfg.use_covariates,
        dropout=cfg.dropout,
    ).to(device)

    if pretrained_model is not None:
        model.load_state_dict(pretrained_model.state_dict())

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    model.train()
    history = []
    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0.0
        n_seen = 0
        for x_past, future_cov, y in loader:
            x_past = x_past.to(device=device, dtype=torch.float32)
            future_cov = future_cov.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                pred = model(x_past, future_cov)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch = x_past.shape[0]
            total_loss += float(loss.item()) * batch
            n_seen += batch

        epoch_loss = total_loss / max(n_seen, 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"N-BEATS h={horizon} epoch={epoch}/{cfg.epochs} loss={epoch_loss:.5f}")

    return model, scales, history


def train_nbeats_two_phase(train_real, synthetic_train, horizon, cfg, scale_source=None):
    """Фаза 1: предобучение на синтетике. Фаза 2: дообучение на реальных данных."""
    if synthetic_train is None:
        return train_nbeats(train_real, None, horizon, cfg, scale_source=scale_source)

    synth_cfg = replace(cfg, window_stride=24)
    model, scales, history_phase1 = train_nbeats(
        synthetic_train[train_real.columns], None, horizon, synth_cfg, scale_source=scale_source
    )

    finetune_epochs = max(3, cfg.epochs // 3)
    real_cfg = replace(cfg, epochs=finetune_epochs, window_stride=1)
    model, scales, history_phase2 = train_nbeats(
        train_real, None, horizon, real_cfg, pretrained_model=model, scale_source=scale_source
    )

    for item in history_phase1:
        item["phase"] = "synth_pretrain"
    for item in history_phase2:
        item["phase"] = "real_finetune"

    return model, scales, history_phase1 + history_phase2


# ──────────────────────── инференс ────────────────────────


def _make_inference_tensors(series, target_index, scale, past_window):
    if len(series) < past_window:
        pad_value = float(series.mean()) if len(series) else 0.0
        padded_values = np.pad(
            series.values.astype(np.float32),
            (past_window - len(series), 0),
            constant_values=pad_value,
        )
        hist_index = pd.date_range(end=series.index[-1], periods=past_window, freq="h")
        history = pd.Series(padded_values[-past_window:], index=hist_index)
    else:
        history = series.iloc[-past_window:]

    horizon = len(target_index)
    scale_log = np.log1p(scale)

    if len(series) >= 24:
        s_end_idx = -24 + horizon
        seasonal_values = (
            series.iloc[-24:s_end_idx].values.astype(np.float32)
            if s_end_idx < 0
            else series.iloc[-24:].values.astype(np.float32)
        )
    else:
        seasonal_values = np.full(horizon, float(series.mean()), dtype=np.float32)

    x_past = (np.log1p(history.values.astype(np.float32)) / scale_log).astype(np.float32)
    future_cov = make_time_covariates(target_index, is_synthetic=0).values.astype(np.float32)
    seasonal_hint = (np.log1p(seasonal_values) / scale_log).reshape(-1, 1)
    future_cov_with_hint = np.concatenate([future_cov, seasonal_hint], axis=1)

    return (
        torch.from_numpy(x_past).unsqueeze(0),
        torch.from_numpy(future_cov_with_hint).unsqueeze(0),
    )


def predict_nbeats(model, train_real, target_index, scales, cfg):
    device = resolve_device(cfg.device)
    model.eval()
    predictions = {}
    with torch.no_grad():
        for object_id in train_real.columns:
            x_past, future_cov = _make_inference_tensors(
                train_real[object_id], target_index, scales[object_id], cfg.past_window
            )
            x_past = x_past.to(device=device, dtype=torch.float32)
            future_cov = future_cov.to(device=device, dtype=torch.float32)
            pred_scaled = model(x_past, future_cov).cpu().numpy().reshape(-1)
            scale_log = np.log1p(scales[object_id])
            pred = np.expm1(np.clip(pred_scaled, 0.0, 4.0) * scale_log)
            pred = np.minimum(pred, scales[object_id] * 10)
            predictions[object_id] = np.maximum(pred, 0.0)
    return predictions


def predict_nbeats_batch(model, history_frame, target_index, scales, cfg, batch_size=None):
    """Пакетный прогноз всех объектов для одного origin."""
    history_frame = history_frame.sort_index()
    object_ids = list(history_frame.columns)
    batch_size = batch_size or cfg.batch_size
    device = resolve_device(cfg.device)
    model.eval()

    if len(history_frame) >= cfg.past_window:
        history = history_frame.iloc[-cfg.past_window:]
        history_index = history.index
        values = history.values.astype(np.float32).T  # [Objects x PastWindow]
    else:
        missing = cfg.past_window - len(history_frame)
        history_index = pd.date_range(
            end=target_index[0] - pd.Timedelta(hours=1),
            periods=cfg.past_window,
            freq="h",
        )
        values = []
        for object_id in object_ids:
            series_values = history_frame[object_id].values.astype(np.float32)
            pad_value = float(series_values.mean()) if len(series_values) else 0.0
            padded = np.pad(series_values, (missing, 0), constant_values=pad_value)[-cfg.past_window:]
            values.append(padded)
        values = np.asarray(values, dtype=np.float32)

    scale_logs = np.log1p(
        np.asarray([scales[oid] for oid in object_ids], dtype=np.float32)
    ).reshape(-1, 1)

    x_past = np.log1p(values) / scale_logs  # [Objects, PastWindow]

    horizon = len(target_index)

    # сезонная подсказка
    if cfg.past_window >= 24:
        s_start = cfg.past_window - 24
        s_end = min(s_start + horizon, cfg.past_window)
        seasonal_pax = values[:, s_start:s_end]
        if seasonal_pax.shape[1] < horizon:
            repeats = (horizon // seasonal_pax.shape[1]) + 1
            seasonal_pax = np.tile(seasonal_pax, (1, repeats))[:, :horizon]
    else:
        mean_vals = values.mean(axis=1, keepdims=True)
        seasonal_pax = np.broadcast_to(mean_vals, (len(object_ids), horizon)).copy()

    seasonal_hint = np.log1p(seasonal_pax) / scale_logs  # [Objects, horizon]

    future_cov = make_time_covariates(target_index, is_synthetic=0).values.astype(np.float32)
    future_cov = np.broadcast_to(future_cov[None, :, :], (len(object_ids), horizon, future_cov.shape[1]))
    future_cov_with_hint = np.concatenate([future_cov, seasonal_hint[:, :, None]], axis=2)

    preds = []
    with torch.no_grad():
        for start in range(0, len(object_ids), batch_size):
            stop = start + batch_size
            x_batch = torch.from_numpy(x_past[start:stop]).to(device=device, dtype=torch.float32)
            cov_batch = torch.from_numpy(future_cov_with_hint[start:stop]).to(device=device, dtype=torch.float32)
            pred_scaled = model(x_batch, cov_batch).cpu().numpy()

            pred = np.expm1(np.clip(pred_scaled, 0.0, 4.0) * scale_logs[start:stop])
            max_pred = np.expm1(scale_logs[start:stop]) * 10
            pred = np.minimum(pred, max_pred)
            preds.append(np.maximum(pred, 0.0))

    pred_matrix = np.vstack(preds)
    return pd.DataFrame(pred_matrix.T, index=target_index, columns=object_ids)


# ──────────────────────── эксперименты ────────────────────────


def _eval_starts(n_hours, train_hours, horizon, step_hours, max_eval_windows=None):
    starts = list(range(train_hours, n_hours - horizon + 1, step_hours))
    if max_eval_windows is not None and len(starts) > max_eval_windows:
        starts = starts[-max_eval_windows:]
    return starts


def run_nbeats_fast_experiment(
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
    cfg = cfg or NBEATSTrainConfig()
    horizons = sorted(set(int(h) for h in horizons))
    max_horizon = max(horizons)
    if train_hours < cfg.past_window + max_horizon:
        raise ValueError(
            f"train_hours={train_hours} is too small for past_window={cfg.past_window} "
            f"and max_horizon={max_horizon}."
        )

    pivot_df = pivot_df.sort_index()
    object_ids = list(pivot_df.columns[:max_objects]) if max_objects else list(pivot_df.columns)
    pivot_df = pivot_df[object_ids]
    train_real = pivot_df.iloc[:train_hours]

    rows = []
    histories = []
    synth_validation_rows = []

    synthetic_train = None
    if "real_plus_synth" in train_modes:
        dynamic_days = get_synth_days(len(train_real), base_synth_days=synth_days)
        print(f"N-BEATS fast: dynamic synth_days={dynamic_days} (train={len(train_real)}h)")
        synthetic_train = synthesize_from_train(
            train_real, gen_days=dynamic_days, seed=cfg.seed + max_horizon * 1000
        )
        validation = validate_synthetic(train_real, synthetic_train)
        validation.update({"horizon": max_horizon, "protocol": "fast_single_train"})
        synth_validation_rows.append(validation)

    for train_mode in train_modes:
        active_synth = synthetic_train if train_mode == "real_plus_synth" else None
        print(
            f"N-BEATS fast mode={train_mode} train={len(train_real)}h "
            f"max_horizon={max_horizon} objects={len(object_ids)}"
        )

        if train_mode == "real_plus_synth":
            model, scales, history = train_nbeats_two_phase(train_real, active_synth, max_horizon, cfg)
        else:
            model, scales, history = train_nbeats(train_real, None, max_horizon, cfg)

        model_path = config.OUTPUT_DIR / f"nbeats_model_fast_{train_mode}_h{max_horizon}.pt"
        save_nbeats_checkpoint(model, scales, cfg, model_path)

        for item in history:
            histories.append({
                **item,
                "horizon": max_horizon,
                "train_mode": train_mode,
                "model": "N-BEATS",
                "protocol": "fast_single_train",
                "train_hours": train_hours,
            })

        active_objects = get_active_objects(train_real, object_ids)
        print(f"  Активных объектов для метрик: {len(active_objects)} / {len(object_ids)}")

        for horizon in horizons:
            step_hours = eval_step_1h if horizon == 1 else (eval_step_24h if horizon == 24 else horizon)
            max_eval_windows = max_eval_windows_1h if horizon == 1 else (max_eval_windows_24h if horizon == 24 else None)

            starts = _eval_starts(len(pivot_df), train_hours, horizon, step_hours, max_eval_windows)
            for eval_idx, start in enumerate(starts):
                forecast_index = pd.date_range(start=pivot_df.index[start], periods=max_horizon, freq="h")
                y_true = pivot_df.iloc[start: start + horizon]
                history_frame = pivot_df.iloc[:start]
                pred = predict_nbeats_batch(model, history_frame, forecast_index, scales, cfg).iloc[:horizon]

                day_mask = get_day_mask(y_true.index)

                for object_id in active_objects:
                    y_true_day = y_true[object_id].values[day_mask]
                    y_pred_day = pred[object_id].values[day_mask]
                    if len(y_true_day) == 0:
                        continue

                    metric_row = forecast_metrics(
                        y_true_day, y_pred_day,
                        model_name="N-BEATS", horizon=horizon, train_mode=train_mode,
                        object_id=object_id, fold=eval_idx,
                    )
                    metric_row.update({
                        "test_start": y_true.index[0],
                        "test_end": y_true.index[-1],
                        "train_hours": train_hours,
                        "test_data": "real",
                        "protocol": "fast_single_train",
                    })
                    rows.append(metric_row)

            print(f"N-BEATS fast mode={train_mode} h={horizon}: evaluated {len(starts)} real windows")

    results = pd.DataFrame(rows)
    summary = summarize_results(results)
    histories = pd.DataFrame(histories)
    synth_validation = pd.DataFrame(synth_validation_rows)
    return results, summary, histories, synth_validation


def run_nbeats_backtest(
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
    """Rolling-origin backtest для N-BEATS."""
    cfg = cfg or NBEATSTrainConfig()
    pivot_df = pivot_df.sort_index()
    object_ids = list(pivot_df.columns[:max_objects]) if max_objects else list(pivot_df.columns)
    pivot_df = pivot_df[object_ids]

    prepended_synth = None
    if "real_plus_synth" in train_modes:
        from ...synthesis import prepend_synthetic_week
        prepended_synth = prepend_synthetic_week(pivot_df, seed=cfg.seed)
        print(f"Synthetic week: {prepended_synth.index[0]} → {prepended_synth.index[-1]}")

    folds = make_rolling_folds(len(pivot_df), horizon, min_train_hours, step_hours)
    if max_folds is not None:
        folds = folds[-max_folds:]
    if not folds:
        raise ValueError(f"Not enough folds for N-BEATS horizon={horizon}.")

    rows = []
    histories = []
    synth_validation_rows = []

    for fold in folds:
        real_train_slice = pivot_df.iloc[fold["train_start"]:fold["train_end"]]
        test_real = pivot_df.iloc[fold["test_start"]:fold["test_end"]]
        target_index = test_real.index

        if fold["fold"] == 0 and prepended_synth is not None:
            validation = validate_synthetic(real_train_slice, prepended_synth)
            validation.update({"fold": fold["fold"], "horizon": horizon})
            synth_validation_rows.append(validation)

        active_objects = get_active_objects(real_train_slice, object_ids)

        for train_mode in train_modes:
            if train_mode == "real_plus_synth" and prepended_synth is not None:
                train_real = pd.concat([prepended_synth, real_train_slice]).sort_index()
                fold_synth = None
                if len(real_train_slice) < 72:
                    dynamic_days = get_synth_days(len(real_train_slice), base_synth_days=synth_days)
                    fold_synth = synthesize_from_train(
                        real_train_slice,
                        gen_days=dynamic_days,
                        seed=cfg.seed + fold["fold"] + horizon * 1000,
                    )
            else:
                train_real = real_train_slice
                fold_synth = None

            print(
                f"N-BEATS h={horizon} fold={fold['fold']} mode={train_mode} "
                f"train={len(train_real)}h (real={len(real_train_slice)}h) "
                f"test={len(test_real)}h active={len(active_objects)}/{len(object_ids)}"
            )

            if train_mode == "real_plus_synth":
                model, scales, history = train_nbeats_two_phase(
                    train_real, fold_synth, horizon, cfg, scale_source=real_train_slice
                )
            else:
                model, scales, history = train_nbeats(real_train_slice, None, horizon, cfg)

            if fold["fold"] == folds[-1]["fold"]:
                model_path = config.OUTPUT_DIR / f"nbeats_model_rolling_{train_mode}_h{horizon}.pt"
                save_nbeats_checkpoint(model, scales, cfg, model_path)

            for item in history:
                histories.append({
                    **item,
                    "horizon": horizon,
                    "fold": fold["fold"],
                    "train_mode": train_mode,
                    "model": "N-BEATS",
                })

            pred_df = predict_nbeats_batch(model, real_train_slice, target_index, scales, cfg)

            day_mask = get_day_mask(target_index)

            for object_id in active_objects:
                y_true_obj = test_real[object_id].values[day_mask]
                y_pred_obj = pred_df[object_id].values[day_mask]
                if len(y_true_obj) == 0:
                    continue

                metric_row = forecast_metrics(
                    y_true_obj, y_pred_obj,
                    model_name="N-BEATS",
                    horizon=horizon,
                    train_mode=train_mode,
                    object_id=object_id,
                    fold=fold["fold"],
                )
                metric_row.update({
                    "test_start": target_index[0],
                    "test_end": target_index[-1],
                    "train_hours": len(real_train_slice),
                    "test_data": "real",
                })
                rows.append(metric_row)

    results = pd.DataFrame(rows)
    summary = summarize_results(results)
    histories = pd.DataFrame(histories)
    synth_validation = pd.DataFrame(synth_validation_rows)
    return results, summary, histories, synth_validation