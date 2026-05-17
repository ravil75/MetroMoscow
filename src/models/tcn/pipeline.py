import random
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ...backtest import make_rolling_folds, summarize_results
from ...baseline_models import forecast_metrics
from ...synthesis import synthesize_from_train, validate_synthetic, get_synth_days
from ...windowing import make_time_covariates


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


class TCNWindowDataset(Dataset):
    """Оптимизированный Dataset: NumPy для скорости + сохранение временных индексов."""

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

        # 1. Предвычисления: готовим данные в NumPy, пока есть доступ к индексам Pandas
        for frame in frames:
            if len(frame) == 0:
                continue

            # Сортируем один раз
            frame = frame.sort_index()

            # Сохраняем значения как массив [Time x Stations]
            self.frames_np.append(frame[self.object_ids].values.astype(np.float32))

            # Генерируем ковариаты, используя ИНДЕКС датафрейма, и сразу в NumPy
            cov = make_time_covariates(frame.index, is_synthetic=is_synthetic)
            self.covariates_np.append(cov.values.astype(np.float32))

        # Предвычисляем логарифмы скейлеров
        self.scales_log_np = np.log1p(
            np.array([scales[oid] for oid in self.object_ids], dtype=np.float32)
        )

        # Собираем индексы сэмплов (теперь работаем только с длинами массивов)
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

        # --- УМНЫЙ SEASONAL HINT С ЗАПОЛНЕНИЕМ СРЕДНИМ ---
        s_start = t_start - 24
        s_end = t_end - 24

        if s_start >= 0:
            seasonal_values = data_np[s_start:s_end, col_idx]
        else:
            fallback_mean = np.mean(history)
            available_len = s_end
            if available_len > 0:
                missing_len = self.horizon - available_len
                parts = [
                    np.full(missing_len, fallback_mean),
                    data_np[0:available_len, col_idx],
                ]
                seasonal_values = np.concatenate(parts)
            else:
                seasonal_values = np.full(self.horizon, fallback_mean)

        seasonal_hint = (np.log1p(seasonal_values) / scale_log).reshape(-1, 1)

        x_value = (np.log1p(history) / scale_log).reshape(-1, 1)
        x_cov = cov_np[h_start:h_end]
        future_cov = cov_np[t_start:t_end]
        future_cov_with_hint = np.concatenate([future_cov, seasonal_hint], axis=1)
        x_seq = np.concatenate([x_value, x_cov], axis=1).T

        return (
            torch.from_numpy(x_seq.astype(np.float32)),
            torch.from_numpy(future_cov_with_hint.astype(np.float32)),
            torch.from_numpy((np.log1p(target) / scale_log).astype(np.float32)),
        )


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(out + residual)


class TCNForecaster(nn.Module):
    def __init__(
        self,
        input_channels,
        future_cov_dim,
        horizon,
        hidden_channels=64,
        levels=4,
        kernel_size=3,
        dropout=0.10,
    ):
        super().__init__()
        layers = []
        for level in range(levels):
            dilation = 2**level
            in_ch = input_channels if level == 0 else hidden_channels
            layers.append(
                TemporalBlock(in_ch, hidden_channels, kernel_size, dilation, dropout)
            )
        self.tcn = nn.Sequential(*layers)
        self.horizon = horizon
        self.head = nn.Sequential(
            nn.Linear(hidden_channels + future_cov_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x_seq, future_cov):
        encoded = self.tcn(x_seq)[:, :, -1]
        context = encoded.unsqueeze(1).expand(-1, self.horizon, -1)
        head_input = torch.cat([context, future_cov], dim=-1)
        return self.head(head_input).squeeze(-1)


@dataclass
class TCNTrainConfig:
    past_window: int = 72          # было 48 — теперь захватываем 3 дня
    epochs: int = 15
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_channels: int = 64
    levels: int = 4
    kernel_size: int = 3
    dropout: float = 0.10
    window_stride: int = 1
    synthetic_window_stride: int = 24
    num_workers: int = 0
    device: str = "auto"
    seed: int = 42
    amp: bool = True


def train_tcn(train_real, synthetic_train, horizon, cfg, pretrained_model=None, scale_source=None):
    """Обучает TCN. Если передан pretrained_model — дообучается поверх него."""
    set_seed(cfg.seed)
    object_ids = list(train_real.columns)
    scales = compute_object_scales(scale_source if scale_source is not None else train_real)

    frames = [train_real]
    strides = [cfg.window_stride]
    if synthetic_train is not None:
        frames.append(synthetic_train[object_ids])
        strides.append(cfg.synthetic_window_stride)

    datasets = [
        TCNWindowDataset(
            [frame],
            object_ids,
            scales,
            cfg.past_window,
            horizon,
            stride=stride,
            is_synthetic=int(i > 0),
        )
        for i, (frame, stride) in enumerate(zip(frames, strides))
    ]
    usable = [dataset for dataset in datasets if len(dataset)]
    if not usable:
        raise ValueError(
            "No TCN training windows were created. Increase train size or reduce past_window."
        )
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
    model = TCNForecaster(
        input_channels=sample_x.shape[0],
        future_cov_dim=sample_future_cov.shape[1],
        horizon=horizon,
        hidden_channels=cfg.hidden_channels,
        levels=cfg.levels,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
    ).to(device)

    # Загружаем веса из pretrained_model если передан (fine-tune фаза)
    if pretrained_model is not None:
        model.load_state_dict(pretrained_model.state_dict())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.SmoothL1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    model.train()
    history = []
    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0.0
        n_seen = 0
        for x_seq, future_cov, y in loader:
            x_seq = x_seq.to(device=device, dtype=torch.float32)
            future_cov = future_cov.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
                pred = model(x_seq, future_cov)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch = x_seq.shape[0]
            total_loss += float(loss.item()) * batch
            n_seen += batch

        epoch_loss = total_loss / max(n_seen, 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"TCN h={horizon} epoch={epoch}/{cfg.epochs} loss={epoch_loss:.5f}")

    return model, scales, history


def train_tcn_two_phase(train_real, synthetic_train, horizon, cfg, scale_source=None):
    if synthetic_train is None:
        return train_tcn(train_real, None, horizon, cfg, scale_source=scale_source)

    synth_cfg = replace(cfg, window_stride=24)
    synth_cols = train_real.columns
    model, scales, history_phase1 = train_tcn(
        synthetic_train[synth_cols], None, horizon, synth_cfg,
        scale_source=scale_source  # передаём реальные данные для scales
    )

    finetune_epochs = max(3, cfg.epochs // 3)
    real_cfg = replace(cfg, epochs=finetune_epochs, window_stride=1)
    model, scales, history_phase2 = train_tcn(
        train_real, None, horizon, real_cfg,
        pretrained_model=model,
        scale_source=scale_source
    )

    for item in history_phase1:
        item["phase"] = "synth_pretrain"
    for item in history_phase2:
        item["phase"] = "real_finetune"

    return model, scales, history_phase1 + history_phase2


def _make_inference_tensors(series, target_index, scale, past_window):
    if len(series) < past_window:
        pad_value = float(series.mean()) if len(series) else 0.0
        padded_values = np.pad(
            series.values.astype(np.float32),
            (past_window - len(series), 0),
            constant_values=pad_value,
        )
        hist_index = pd.date_range(
            end=series.index[-1], periods=past_window, freq="h"
        )
        history = pd.Series(padded_values[-past_window:], index=hist_index)
    else:
        history = series.iloc[-past_window:]

    horizon = len(target_index)
    scale_log = np.log1p(scale)

    if len(series) >= 24:
        s_end_idx = -24 + horizon
        if s_end_idx < 0:
            seasonal_values = series.iloc[-24:s_end_idx].values.astype(np.float32)
        else:
            seasonal_values = series.iloc[-24:].values.astype(np.float32)
    else:
        seasonal_values = np.full(horizon, float(series.mean()), dtype=np.float32)

    x_value = (np.log1p(history.values.astype(np.float32)) / scale_log).reshape(-1, 1)
    x_cov = make_time_covariates(history.index, is_synthetic=0).values.astype(np.float32)
    future_cov = make_time_covariates(target_index, is_synthetic=0).values.astype(
        np.float32
    )
    seasonal_hint = (np.log1p(seasonal_values) / scale_log).reshape(-1, 1)
    future_cov_with_hint = np.concatenate([future_cov, seasonal_hint], axis=1)
    x_seq = np.concatenate([x_value, x_cov], axis=1).T

    return (
        torch.from_numpy(x_seq).unsqueeze(0),
        torch.from_numpy(future_cov_with_hint).unsqueeze(0),
    )


def predict_tcn(model, train_real, target_index, scales, cfg):
    device = resolve_device(cfg.device)
    model.eval()
    predictions = {}
    with torch.no_grad():
        for object_id in train_real.columns:
            x_seq, future_cov = _make_inference_tensors(
                train_real[object_id],
                target_index,
                scales[object_id],
                cfg.past_window,
            )
            x_seq = x_seq.to(device=device, dtype=torch.float32)
            future_cov = future_cov.to(device=device, dtype=torch.float32)
            pred_scaled = model(x_seq, future_cov).cpu().numpy().reshape(-1)
            # Клиппинг в лог-пространстве чтобы избежать взрывных предсказаний
            scale_log = np.log1p(scales[object_id])
            pred = np.expm1(np.clip(pred_scaled, 0.0, 4.0) * scale_log)
            pred = np.minimum(pred, scales[object_id] * 10)  # не более 10× среднего
            predictions[object_id] = np.maximum(pred, 0.0)

    return predictions


def predict_tcn_batch(model, history_frame, target_index, scales, cfg, batch_size=None):
    """Predict all objects in batches for one forecast origin."""
    history_frame = history_frame.sort_index()
    object_ids = list(history_frame.columns)
    batch_size = batch_size or cfg.batch_size
    device = resolve_device(cfg.device)
    model.eval()

    if len(history_frame) >= cfg.past_window:
        history = history_frame.iloc[-cfg.past_window :]
        history_index = history.index
        values = history.values.astype(np.float32).T  # [Stations x PastWindow]
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
            padded = np.pad(
                series_values, (missing, 0), constant_values=pad_value
            )[-cfg.past_window :]
            values.append(padded)
        values = np.asarray(values, dtype=np.float32)

    scale_logs = np.log1p(
        np.asarray([scales[object_id] for object_id in object_ids], dtype=np.float32)
    ).reshape(-1, 1)
    x_value = np.log1p(values) / scale_logs
    x_cov = make_time_covariates(history_index, is_synthetic=0).values.astype(
        np.float32
    )
    x_cov = np.broadcast_to(
        x_cov[None, :, :], (len(object_ids), cfg.past_window, x_cov.shape[1])
    )
    x_seq = np.concatenate([x_value[:, :, None], x_cov], axis=2).transpose(0, 2, 1)

    horizon = len(target_index)
    s_start = cfg.past_window - 24
    s_end = s_start + horizon
    seasonal_pax = values[:, s_start:s_end]
    seasonal_hint = np.log1p(seasonal_pax) / scale_logs  # [Stations x Horizon]

    future_cov = make_time_covariates(target_index, is_synthetic=0).values.astype(
        np.float32
    )
    future_cov = np.broadcast_to(
        future_cov[None, :, :], (len(object_ids), horizon, future_cov.shape[1])
    )
    future_cov_with_hint = np.concatenate(
        [future_cov, seasonal_hint[:, :, None]], axis=2
    )

    preds = []
    with torch.no_grad():
        for start in range(0, len(object_ids), batch_size):
            stop = start + batch_size
            x_batch = torch.from_numpy(x_seq[start:stop]).to(
                device=device, dtype=torch.float32
            )
            cov_batch = torch.from_numpy(future_cov_with_hint[start:stop]).to(
                device=device, dtype=torch.float32
            )
            pred_scaled = model(x_batch, cov_batch).cpu().numpy()
            # Клиппинг в лог-пространстве — убирает взрывной WAPE
            pred = np.expm1(np.clip(pred_scaled, 0.0, 4.0) * scale_logs[start:stop])
            max_pred = np.expm1(scale_logs[start:stop]) * 10  # expm1(log1p(scale)) = scale
            pred = np.minimum(pred, max_pred)
            preds.append(np.maximum(pred, 0.0))

    pred_matrix = np.vstack(preds)
    return pd.DataFrame(pred_matrix.T, index=target_index, columns=object_ids)


def _eval_starts(n_hours, train_hours, horizon, step_hours, max_eval_windows=None):
    starts = list(range(train_hours, n_hours - horizon + 1, step_hours))
    if max_eval_windows is not None and len(starts) > max_eval_windows:
        starts = starts[-max_eval_windows:]
    return starts


def run_tcn_fast_experiment(
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
    """Train once per mode, then evaluate many real forecast origins."""
    cfg = cfg or TCNTrainConfig()
    horizons = sorted(set(int(horizon) for horizon in horizons))
    max_horizon = max(horizons)
    if train_hours < cfg.past_window + max_horizon:
        raise ValueError(
            f"train_hours={train_hours} is too small for past_window={cfg.past_window} "
            f"and max_horizon={max_horizon}."
        )

    pivot_df = pivot_df.sort_index()
    object_ids = (
        list(pivot_df.columns[:max_objects]) if max_objects else list(pivot_df.columns)
    )
    pivot_df = pivot_df[object_ids]
    train_real = pivot_df.iloc[:train_hours]

    rows = []
    histories = []
    synth_validation_rows = []

    synthetic_train = None
    if "real_plus_synth" in train_modes:
        # Динамически подбираем количество синтетических дней
        dynamic_days = get_synth_days(len(train_real), base_synth_days=synth_days)
        print(f"TCN fast: dynamic synth_days={dynamic_days} (train={len(train_real)}h)")
        synthetic_train = synthesize_from_train(
            train_real,
            gen_days=dynamic_days,
            seed=cfg.seed + max_horizon * 1000,
        )
        validation = validate_synthetic(train_real, synthetic_train)
        validation.update({"horizon": max_horizon, "protocol": "fast_single_train"})
        synth_validation_rows.append(validation)

    for train_mode in train_modes:
        active_synth = synthetic_train if train_mode == "real_plus_synth" else None
        print(
            f"TCN fast mode={train_mode} train={len(train_real)}h "
            f"max_horizon={max_horizon} objects={len(object_ids)}"
        )

        # Двухфазное обучение для real_plus_synth
        if train_mode == "real_plus_synth":
            model, scales, history = train_tcn_two_phase(
                train_real, active_synth, max_horizon, cfg
            )
        else:
            model, scales, history = train_tcn(train_real, None, max_horizon, cfg)

        for item in history:
            histories.append(
                {
                    **item,
                    "horizon": max_horizon,
                    "train_mode": train_mode,
                    "model": "TCN",
                    "protocol": "fast_single_train",
                    "train_hours": train_hours,
                }
            )

        for horizon in horizons:
            if horizon == 1:
                step_hours = eval_step_1h
                max_eval_windows = max_eval_windows_1h
            elif horizon == 24:
                step_hours = eval_step_24h
                max_eval_windows = max_eval_windows_24h
            else:
                step_hours = horizon
                max_eval_windows = None

            starts = _eval_starts(
                len(pivot_df), train_hours, horizon, step_hours, max_eval_windows
            )
            for eval_idx, start in enumerate(starts):
                forecast_index = pd.date_range(
                    start=pivot_df.index[start],
                    periods=max_horizon,
                    freq="h",
                )
                y_true = pivot_df.iloc[start : start + horizon]
                history_frame = pivot_df.iloc[:start]
                pred = predict_tcn_batch(
                    model, history_frame, forecast_index, scales, cfg
                ).iloc[:horizon]

                for object_id in object_ids:
                    metric_row = forecast_metrics(
                        y_true[object_id].values,
                        pred[object_id].values,
                        model_name="TCN",
                        horizon=horizon,
                        train_mode=train_mode,
                        object_id=object_id,
                        fold=eval_idx,
                    )
                    metric_row.update(
                        {
                            "test_start": y_true.index[0],
                            "test_end": y_true.index[-1],
                            "train_hours": train_hours,
                            "test_data": "real",
                            "protocol": "fast_single_train",
                        }
                    )
                    rows.append(metric_row)

            print(
                f"TCN fast mode={train_mode} h={horizon}: evaluated {len(starts)} real windows"
            )

    results = pd.DataFrame(rows)
    summary = summarize_results(results)
    histories = pd.DataFrame(histories)
    synth_validation = pd.DataFrame(synth_validation_rows)
    return results, summary, histories, synth_validation


def run_tcn_backtest(
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
    cfg = cfg or TCNTrainConfig()
    pivot_df = pivot_df.sort_index()
    object_ids = (
        list(pivot_df.columns[:max_objects]) if max_objects else list(pivot_df.columns)
    )
    pivot_df = pivot_df[object_ids]

    # Генерируем синтетическую неделю один раз до всех фолдов
    prepended_synth = None
    if "real_plus_synth" in train_modes:
        from ...synthesis import prepend_synthetic_week
        prepended_synth = prepend_synthetic_week(pivot_df, seed=cfg.seed)
        print(
            f"Synthetic week: {prepended_synth.index[0]} → {prepended_synth.index[-1]}"
        )

    folds = make_rolling_folds(len(pivot_df), horizon, min_train_hours, step_hours)
    if max_folds is not None:
        folds = folds[-max_folds:]
    if not folds:
        raise ValueError(f"Not enough folds for TCN horizon={horizon}.")

    rows = []
    histories = []
    synth_validation_rows = []

    for fold in folds:
        real_train_slice = pivot_df.iloc[fold["train_start"]:fold["train_end"]]
        test_real = pivot_df.iloc[fold["test_start"]:fold["test_end"]]
        target_index = test_real.index

        # Валидируем синтетику один раз (только на первом фолде)
        if fold["fold"] == 0 and prepended_synth is not None:
            validation = validate_synthetic(real_train_slice, prepended_synth)
            validation.update({"fold": fold["fold"], "horizon": horizon})
            synth_validation_rows.append(validation)

        for train_mode in train_modes:
            if train_mode == "real_plus_synth" and prepended_synth is not None:
                # Синтетическая неделя + реальные данные этого фолда
                train_real = pd.concat(
                    [prepended_synth, real_train_slice]
                ).sort_index()

                # Если реала совсем мало — добавляем ещё внутрифолдовой синтетики
                fold_synth = None
                if len(real_train_slice) < 72:
                    dynamic_days = get_synth_days(
                        len(real_train_slice), base_synth_days=synth_days
                    )
                    fold_synth = synthesize_from_train(
                        real_train_slice,
                        gen_days=dynamic_days,
                        seed=cfg.seed + fold["fold"] + horizon * 1000,
                    )
            else:
                train_real = real_train_slice
                fold_synth = None

            print(
                f"TCN h={horizon} fold={fold['fold']} mode={train_mode} "
                f"train={len(train_real)}h (real={len(real_train_slice)}h) "
                f"test={len(test_real)}h objects={len(object_ids)}"
            )

            if train_mode == "real_plus_synth":
                model, scales, history = train_tcn_two_phase(
                    train_real, fold_synth, horizon, cfg,
                    scale_source=real_train_slice  # scales только по реалу
                )
            else:
                model, scales, history = train_tcn(
                    real_train_slice, None, horizon, cfg
                )

            for item in history:
                histories.append(
                    {
                        **item,
                        "horizon": horizon,
                        "fold": fold["fold"],
                        "train_mode": train_mode,
                        "model": "TCN",
                    }
                )

            pred_df = predict_tcn_batch(model, real_train_slice, target_index, scales, cfg)

            for object_id in object_ids:
                metric_row = forecast_metrics(
                    test_real[object_id].values,
                    pred_df[object_id].values,
                    model_name="TCN",
                    horizon=horizon,
                    train_mode=train_mode,
                    object_id=object_id,
                    fold=fold["fold"],
                )
                metric_row.update(
                    {
                        "test_start": target_index[0],
                        "test_end": target_index[-1],
                        "train_hours": len(real_train_slice),
                        "test_data": "real",
                    }
                )
                rows.append(metric_row)

    results = pd.DataFrame(rows)
    summary = summarize_results(results)
    histories = pd.DataFrame(histories)
    synth_validation = pd.DataFrame(synth_validation_rows)
    return results, summary, histories, synth_validation