import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ... import config
from ...backtest import make_rolling_folds, summarize_results
from ...baseline_models import forecast_metrics
from ...synthesis import get_synth_days, synthesize_from_train, validate_synthetic
from ...windowing import make_time_covariates
from ...static_features import build_static_covariates, encode_static_covariates

NIGHT_HOURS = frozenset({0, 1, 2, 3, 4, 5})
MIN_DAILY_PAX = 100


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


# ──────────────────────── Чекпойнты ────────────────────────

def save_tft_checkpoint(model, scales, static_enc, cfg, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "scales": scales,
        "static_enc": static_enc,  # сохраняем словари кодирования
        "cfg": cfg.__dict__,
        "model_params": {
            "cat_cardinalities": model.cat_cardinalities,
            "num_cont_static": model.num_cont_static,
            "num_past_features": model.num_past_features,
            "num_future_features": model.num_future_features,
        }
    }
    torch.save(checkpoint, filepath)
    print(f"Модель, скейлеры и статический энкодер сохранены в: {filepath}")


def load_tft_checkpoint(filepath, device="auto"):
    device = resolve_device(device)
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    cfg = TFTTrainConfig(**checkpoint["cfg"])
    mp = checkpoint["model_params"]
    
    model = TFTForecaster(
        cat_cardinalities=mp["cat_cardinalities"],
        num_cont_static=mp["num_cont_static"],
        num_past_features=mp["num_past_features"],
        num_future_features=mp["num_future_features"],
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["scales"], checkpoint["static_enc"], cfg


# ──────────────────────── Архитектура TFT ────────────────────────

class GLU(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model * 2)

    def forward(self, x):
        x, gate = self.fc(x).chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class GRN(nn.Module):
    """Gated Residual Network"""
    def __init__(self, d_model, hidden_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = GLU(d_model)
        self.ln = nn.LayerNorm(d_model)
        
        # Проекция контекста (опционально)
        self.context_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, context=None):
        a = self.fc1(x)
        if context is not None:
            a = a + self.context_proj(context)
        a = self.elu(a)
        a = self.fc2(a)
        a = self.dropout(a)
        return self.ln(x + self.gate(a))


class TFTForecaster(nn.Module):
    """Temporal Fusion Transformer (Оптимизированная версия)"""
    def __init__(
        self, 
        cat_cardinalities, 
        num_cont_static, 
        num_past_features, 
        num_future_features, 
        hidden_size=64, 
        num_heads=4, 
        dropout=0.1
    ):
        super().__init__()
        self.cat_cardinalities = cat_cardinalities
        self.num_cont_static = num_cont_static
        self.num_past_features = num_past_features
        self.num_future_features = num_future_features
        self.hidden_size = hidden_size

        # 1. Статические энкодеры
        self.cat_embs = nn.ModuleList([
            nn.Embedding(c, hidden_size) for c in cat_cardinalities
        ])
        self.cont_proj = nn.Linear(num_cont_static, hidden_size) if num_cont_static > 0 else None
        self.static_grn = GRN(hidden_size, hidden_size, dropout)
        
        # 2. Проекции временных рядов
        self.past_proj = nn.Linear(num_past_features, hidden_size)
        self.future_proj = nn.Linear(num_future_features, hidden_size)
        
        self.past_grn = GRN(hidden_size, hidden_size, dropout)
        self.future_grn = GRN(hidden_size, hidden_size, dropout)

        # 3. Seq2Seq LSTM (с инициализацией из статического контекста)
        self.h0_proj = nn.Linear(hidden_size, hidden_size)
        self.c0_proj = nn.Linear(hidden_size, hidden_size)
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # 4. Multi-Head Attention (Будущее -> Прошлое)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_size)

        # 5. Выход
        self.post_attn_grn = GRN(hidden_size, hidden_size, dropout)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x_past, x_future, cat_static, cont_static):
        # 1. Формирование статического контекста
        embs = [emb(cat_static[:, i]) for i, emb in enumerate(self.cat_embs)]
        if self.cont_proj is not None:
            embs.append(self.cont_proj(cont_static))
        
        static_stacked = torch.stack(embs, dim=1) # [B, N_features, hidden_size]
        static_context = self.static_grn(static_stacked.mean(dim=1)) # [B, hidden_size]

        # 2. Временные признаки + статический контекст
        past_emb = self.past_proj(x_past)
        future_emb = self.future_proj(x_future)
        
        past_emb = self.past_grn(past_emb, context=static_context.unsqueeze(1))
        future_emb = self.future_grn(future_emb, context=static_context.unsqueeze(1))

        # 3. Инициализация и проход LSTM
        h0 = self.h0_proj(static_context).unsqueeze(0) # [1, B, H]
        c0 = self.c0_proj(static_context).unsqueeze(0) # [1, B, H]
        
        past_out, (h_past, c_past) = self.encoder_lstm(past_emb, (h0, c0))
        future_out, _ = self.decoder_lstm(future_emb, (h_past, c_past))

        # 4. Attention (Decoder обращается к памяти Encoder-а)
        attn_out, _ = self.attn(query=future_out, key=past_out, value=past_out)
        attn_out = self.attn_norm(future_out + attn_out)

        # 5. Предсказание
        out = self.post_attn_grn(attn_out)
        return self.output_layer(out).squeeze(-1) # [B, Horizon]


# ──────────────────────── Датасет ────────────────────────

class TFTWindowDataset(Dataset):
    def __init__(
        self, frames, object_ids, scales, static_enc, past_window, horizon, stride=1, is_synthetic=0
    ):
        self.object_ids = list(object_ids)
        self.past_window = past_window
        self.horizon = horizon
        self.stride = max(int(stride), 1)
        self.samples = []
        self.frames_np = []
        self.covariates_np = []

        # Извлекаем статику и упорядочиваем по object_ids
        obj_to_idx = {obj: i for i, obj in enumerate(static_enc["object_ids"])}
        indices = [obj_to_idx[obj] for obj in self.object_ids]
        self.static_cat = static_enc["categorical"][indices].astype(np.int64)
        self.static_cont = static_enc["continuous"][indices].astype(np.float32)

        for frame in frames:
            if len(frame) == 0: continue
            frame = frame.sort_index()
            self.frames_np.append(frame[self.object_ids].values.astype(np.float32))
            cov = make_time_covariates(frame.index, is_synthetic=is_synthetic)
            self.covariates_np.append(cov.values.astype(np.float32))

        self.scales_log_np = np.log1p(
            np.array([scales[oid] for oid in self.object_ids], dtype=np.float32)
        )

        for frame_idx, frame_np in enumerate(self.frames_np):
            max_start = len(frame_np) - past_window - horizon
            if max_start < 0: continue
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

        s_start, s_end = t_start - 24, t_end - 24
        if s_start >= 0:
            seasonal_values = data_np[s_start:s_end, col_idx]
        else:
            fallback_mean = np.mean(history)
            available_len = s_end
            if available_len > 0:
                seasonal_values = np.concatenate([
                    np.full(self.horizon - available_len, fallback_mean),
                    data_np[0:available_len, col_idx],
                ])
            else:
                seasonal_values = np.full(self.horizon, fallback_mean)

        # Подготовка тензоров
        x_past_y = (np.log1p(history) / scale_log).reshape(-1, 1)
        x_past_cov = cov_np[h_start:h_end]
        x_past = np.concatenate([x_past_y, x_past_cov], axis=1)

        seasonal_hint = (np.log1p(seasonal_values) / scale_log).reshape(-1, 1)
        x_future_cov = cov_np[t_start:t_end]
        x_future = np.concatenate([x_future_cov, seasonal_hint], axis=1)

        y_target = np.log1p(target) / scale_log

        return (
            torch.from_numpy(x_past.astype(np.float32)),
            torch.from_numpy(x_future.astype(np.float32)),
            torch.from_numpy(self.static_cat[col_idx]),
            torch.from_numpy(self.static_cont[col_idx]),
            torch.from_numpy(y_target.astype(np.float32)),
        )


# ──────────────────────── Обучение ────────────────────────

@dataclass
class TFTTrainConfig:
    past_window: int = 120
    epochs: int = 15
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_size: int = 64
    num_heads: int = 4
    dropout: float = 0.15
    window_stride: int = 1
    synthetic_window_stride: int = 24
    num_workers: int = 0
    device: str = "auto"
    seed: int = 42
    amp: bool = True


def train_tft(train_real, synthetic_train, static_enc, horizon, cfg, pretrained_model=None, scale_source=None):
    set_seed(cfg.seed)
    object_ids = list(train_real.columns)
    scales = compute_object_scales(scale_source if scale_source is not None else train_real)

    frames = [train_real]
    strides = [cfg.window_stride]
    if synthetic_train is not None:
        frames.append(synthetic_train[object_ids])
        strides.append(cfg.synthetic_window_stride)

    datasets = [
        TFTWindowDataset(
            [frame], object_ids, scales, static_enc, cfg.past_window, horizon,
            stride=stride, is_synthetic=int(i > 0)
        )
        for i, (frame, stride) in enumerate(zip(frames, strides))
    ]
    usable = [d for d in datasets if len(d)]
    if not usable:
        raise ValueError("No TFT training windows created.")
    train_dataset = usable[0] if len(usable) == 1 else torch.utils.data.ConcatDataset(usable)

    loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available()
    )

    x_past, x_future, cat_stat, cont_stat, _ = usable[0][0]
    device = resolve_device(cfg.device)

    cat_cardinalities = [static_enc["cardinalities"][c] for c in static_enc["categorical_cols"]]
    
    model = TFTForecaster(
        cat_cardinalities=cat_cardinalities,
        num_cont_static=cont_stat.shape[0],
        num_past_features=x_past.shape[1],
        num_future_features=x_future.shape[1],
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_heads,
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
        total_loss, n_seen = 0.0, 0
        for xp, xf, c_cat, c_cont, y in loader:
            xp = xp.to(device=device, dtype=torch.float32)
            xf = xf.to(device=device, dtype=torch.float32)
            c_cat = c_cat.to(device=device, dtype=torch.long)
            c_cont = c_cont.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                pred = model(xp, xf, c_cat, c_cont)
                loss = loss_fn(pred, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch = xp.shape[0]
            total_loss += float(loss.item()) * batch
            n_seen += batch

        epoch_loss = total_loss / max(n_seen, 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"TFT h={horizon} epoch={epoch}/{cfg.epochs} loss={epoch_loss:.5f}")

    return model, scales, history


def train_tft_two_phase(train_real, synthetic_train, static_enc, horizon, cfg, scale_source=None):
    if synthetic_train is None:
        return train_tft(train_real, None, static_enc, horizon, cfg, scale_source=scale_source)

    synth_cfg = replace(cfg, window_stride=24)
    model, scales, history_phase1 = train_tft(
        synthetic_train[train_real.columns], None, static_enc, horizon, synth_cfg, scale_source=scale_source
    )

    finetune_epochs = max(3, cfg.epochs // 3)
    real_cfg = replace(cfg, epochs=finetune_epochs, window_stride=1)
    model, scales, history_phase2 = train_tft(
        train_real, None, static_enc, horizon, real_cfg, pretrained_model=model, scale_source=scale_source
    )

    for item in history_phase1: item["phase"] = "synth_pretrain"
    for item in history_phase2: item["phase"] = "real_finetune"

    return model, scales, history_phase1 + history_phase2


# ──────────────────────── Инференс ────────────────────────

def predict_tft_batch(model, history_frame, target_index, scales, static_enc, cfg, batch_size=None):
    history_frame = history_frame.sort_index()
    object_ids = list(history_frame.columns)
    batch_size = batch_size or cfg.batch_size
    device = resolve_device(cfg.device)
    model.eval()

    # Подготовка статики
    obj_to_idx = {obj: i for i, obj in enumerate(static_enc["object_ids"])}
    indices = [obj_to_idx[obj] for obj in object_ids]
    static_cat = static_enc["categorical"][indices]
    static_cont = static_enc["continuous"][indices]

    if len(history_frame) >= cfg.past_window:
        history = history_frame.iloc[-cfg.past_window:]
        history_index = history.index
        values = history.values.astype(np.float32).T
    else:
        missing = cfg.past_window - len(history_frame)
        history_index = pd.date_range(end=target_index[0] - pd.Timedelta(hours=1), periods=cfg.past_window, freq="h")
        values = []
        for obj_id in object_ids:
            series_values = history_frame[obj_id].values.astype(np.float32)
            pad_val = float(series_values.mean()) if len(series_values) else 0.0
            values.append(np.pad(series_values, (missing, 0), constant_values=pad_val)[-cfg.past_window:])
        values = np.asarray(values, dtype=np.float32)

    scale_logs = np.log1p(np.asarray([scales[oid] for oid in object_ids], dtype=np.float32)).reshape(-1, 1)

    # Past Tensors
    y_past = np.log1p(values) / scale_logs
    cov_past = make_time_covariates(history_index, is_synthetic=0).values.astype(np.float32)
    cov_past = np.broadcast_to(cov_past[None, :, :], (len(object_ids), cfg.past_window, cov_past.shape[1]))
    x_past = np.concatenate([y_past[:, :, None], cov_past], axis=2)

    # Future Tensors
    horizon = len(target_index)
    if cfg.past_window >= 24:
        s_start, s_end = cfg.past_window - 24, min(cfg.past_window - 24 + horizon, cfg.past_window)
        seasonal_pax = values[:, s_start:s_end]
        if seasonal_pax.shape[1] < horizon:
            seasonal_pax = np.tile(seasonal_pax, (1, (horizon // seasonal_pax.shape[1]) + 1))[:, :horizon]
    else:
        seasonal_pax = np.broadcast_to(values.mean(axis=1, keepdims=True), (len(object_ids), horizon)).copy()
    
    seasonal_hint = np.log1p(seasonal_pax) / scale_logs
    cov_future = make_time_covariates(target_index, is_synthetic=0).values.astype(np.float32)
    cov_future = np.broadcast_to(cov_future[None, :, :], (len(object_ids), horizon, cov_future.shape[1]))
    x_future = np.concatenate([cov_future, seasonal_hint[:, :, None]], axis=2)

    preds = []
    with torch.no_grad():
        for start in range(0, len(object_ids), batch_size):
            stop = start + batch_size
            xp_batch = torch.from_numpy(x_past[start:stop]).to(device, dtype=torch.float32)
            xf_batch = torch.from_numpy(x_future[start:stop]).to(device, dtype=torch.float32)
            c_cat_batch = torch.from_numpy(static_cat[start:stop]).to(device, dtype=torch.long)
            c_cont_batch = torch.from_numpy(static_cont[start:stop]).to(device, dtype=torch.float32)

            pred_scaled = model(xp_batch, xf_batch, c_cat_batch, c_cont_batch).cpu().numpy()

            pred = np.expm1(np.clip(pred_scaled, 0.0, 4.0) * scale_logs[start:stop])
            max_pred = np.expm1(scale_logs[start:stop]) * 10
            preds.append(np.maximum(np.minimum(pred, max_pred), 0.0))

    return pd.DataFrame(np.vstack(preds).T, index=target_index, columns=object_ids)


# ──────────────────────── Эксперименты ────────────────────────

def _eval_starts(n_hours, train_hours, horizon, step_hours, max_eval_windows=None):
    starts = list(range(train_hours, n_hours - horizon + 1, step_hours))
    if max_eval_windows is not None and len(starts) > max_eval_windows:
        starts = starts[-max_eval_windows:]
    return starts


def run_tft_fast_experiment(
    pivot_df, horizons=(1, 24), train_modes=("real_only", "real_plus_synth"),
    synth_days=30, train_hours=96, eval_step_1h=1, eval_step_24h=24,
    max_eval_windows_1h=None, max_eval_windows_24h=None, max_objects=None, cfg=None
):
    cfg = cfg or TFTTrainConfig()
    horizons = sorted(set(int(h) for h in horizons))
    max_horizon = max(horizons)
    
    pivot_df = pivot_df.sort_index()
    object_ids = list(pivot_df.columns[:max_objects]) if max_objects else list(pivot_df.columns)
    pivot_df = pivot_df[object_ids]
    train_real = pivot_df.iloc[:train_hours]

    # Инициализация статики
    static_df = build_static_covariates(pivot_df, object_ids=object_ids)
    static_enc = encode_static_covariates(static_df)

    rows, histories, synth_validation_rows = [], [], []

    synthetic_train = None
    if "real_plus_synth" in train_modes:
        dynamic_days = get_synth_days(len(train_real), base_synth_days=synth_days)
        synthetic_train = synthesize_from_train(train_real, gen_days=dynamic_days, seed=cfg.seed + max_horizon * 1000)
        validation = validate_synthetic(train_real, synthetic_train)
        validation.update({"horizon": max_horizon, "protocol": "fast_single_train"})
        synth_validation_rows.append(validation)

    for train_mode in train_modes:
        active_synth = synthetic_train if train_mode == "real_plus_synth" else None
        print(f"TFT fast mode={train_mode} train={len(train_real)}h max_horizon={max_horizon} objects={len(object_ids)}")

        if train_mode == "real_plus_synth":
            model, scales, history = train_tft_two_phase(train_real, active_synth, static_enc, max_horizon, cfg)
        else:
            model, scales, history = train_tft(train_real, None, static_enc, max_horizon, cfg)

        model_path = config.OUTPUT_DIR / f"tft_model_fast_{train_mode}_h{max_horizon}.pt"
        save_tft_checkpoint(model, scales, static_enc, cfg, model_path)

        for item in history:
            histories.append({**item, "horizon": max_horizon, "train_mode": train_mode, "model": "TFT"})

        active_objects = get_active_objects(train_real, object_ids)

        for horizon in horizons:
            step_hours = eval_step_1h if horizon == 1 else (eval_step_24h if horizon == 24 else horizon)
            max_eval_windows = max_eval_windows_1h if horizon == 1 else (max_eval_windows_24h if horizon == 24 else None)

            starts = _eval_starts(len(pivot_df), train_hours, horizon, step_hours, max_eval_windows)
            for eval_idx, start in enumerate(starts):
                forecast_index = pd.date_range(start=pivot_df.index[start], periods=max_horizon, freq="h")
                y_true = pivot_df.iloc[start: start + horizon]
                history_frame = pivot_df.iloc[:start]
                pred = predict_tft_batch(model, history_frame, forecast_index, scales, static_enc, cfg).iloc[:horizon]

                day_mask = get_day_mask(y_true.index)
                for object_id in active_objects:
                    y_true_day = y_true[object_id].values[day_mask]
                    y_pred_day = pred[object_id].values[day_mask]
                    if len(y_true_day) == 0: continue

                    metric_row = forecast_metrics(
                        y_true_day, y_pred_day, model_name="TFT", horizon=horizon,
                        train_mode=train_mode, object_id=object_id, fold=eval_idx,
                    )
                    metric_row.update({"test_data": "real", "protocol": "fast"})
                    rows.append(metric_row)

    results = pd.DataFrame(rows)
    summary = summarize_results(results)
    return results, summary, pd.DataFrame(histories), pd.DataFrame(synth_validation_rows)


def run_tft_backtest(
    pivot_df, horizon, train_modes=("real_only", "real_plus_synth"),
    synth_days=30, min_train_hours=None, step_hours=None, max_folds=None, max_objects=None, cfg=None
):
    cfg = cfg or TFTTrainConfig()
    pivot_df = pivot_df.sort_index()
    object_ids = list(pivot_df.columns[:max_objects]) if max_objects else list(pivot_df.columns)
    pivot_df = pivot_df[object_ids]

    static_df = build_static_covariates(pivot_df, object_ids=object_ids)
    static_enc = encode_static_covariates(static_df)

    prepended_synth = None
    if "real_plus_synth" in train_modes:
        from ...synthesis import prepend_synthetic_week
        prepended_synth = prepend_synthetic_week(pivot_df, seed=cfg.seed)

    folds = make_rolling_folds(len(pivot_df), horizon, min_train_hours, step_hours)
    if max_folds is not None: folds = folds[-max_folds:]

    rows, histories, synth_validation_rows = [], [], []

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
                    fold_synth = synthesize_from_train(real_train_slice, gen_days=dynamic_days, seed=cfg.seed + fold["fold"])
            else:
                train_real = real_train_slice
                fold_synth = None

            print(f"TFT h={horizon} fold={fold['fold']} mode={train_mode} train={len(train_real)}h")

            if train_mode == "real_plus_synth":
                model, scales, history = train_tft_two_phase(train_real, fold_synth, static_enc, horizon, cfg, scale_source=real_train_slice)
            else:
                model, scales, history = train_tft(real_train_slice, None, static_enc, horizon, cfg)

            if fold["fold"] == folds[-1]["fold"]:
                model_path = config.OUTPUT_DIR / f"tft_model_rolling_{train_mode}_h{horizon}.pt"
                save_tft_checkpoint(model, scales, static_enc, cfg, model_path)

            for item in history:
                histories.append({**item, "horizon": horizon, "fold": fold["fold"], "train_mode": train_mode, "model": "TFT"})

            pred_df = predict_tft_batch(model, real_train_slice, target_index, scales, static_enc, cfg)
            day_mask = get_day_mask(target_index)

            for object_id in active_objects:
                y_true_obj = test_real[object_id].values[day_mask]
                y_pred_obj = pred_df[object_id].values[day_mask]
                if len(y_true_obj) == 0: continue

                metric_row = forecast_metrics(
                    y_true_obj, y_pred_obj, model_name="TFT", horizon=horizon,
                    train_mode=train_mode, object_id=object_id, fold=fold["fold"]
                )
                metric_row.update({"test_data": "real"})
                rows.append(metric_row)

    results = pd.DataFrame(rows)
    summary = summarize_results(results)
    return results, summary, pd.DataFrame(histories), pd.DataFrame(synth_validation_rows)