import math
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
from ...windowing import make_time_covariates
from ... import config


# ── Константы ─────────────────────────────────────────────────────────────────
NIGHT_HOURS     = frozenset({0, 1, 2, 3, 4, 5})
NIGHT_HOURS_ARR = np.array(sorted(NIGHT_HOURS))
MIN_DAILY_PAX   = 100


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


# ── Ego-граф: residual-корреляции + статики узлов ─────────────────────────────
#
# Сырые корреляции часовых рядов бесполезны: общий суточный цикл даёт ~0.7-0.95
# между любыми живыми объектами. Поэтому из каждого ряда вычитается его средний
# профиль (час × будний/выходной), и граф строится по корреляции ОСТАТКОВ —
# совместных отклонений от нормы (пересадки, узлы, общегородские события).

def _hourly_profile(log_values, index):
    hours = index.hour.values
    wknd = (index.dayofweek.values >= 5).astype(np.int64)
    key = hours * 2 + wknd if np.unique(wknd).size > 1 else hours
    profile = np.zeros_like(log_values)
    for g in np.unique(key):
        m = key == g
        profile[m] = log_values[m].mean(axis=0, keepdims=True)
    return profile


def compute_node_stats(train_slice):
    """Статические признаки объектов из train-среза:
    24-часовой профиль нормированного потока + сводные статистики."""
    values = train_slice.values.astype(np.float64)
    index = pd.DatetimeIndex(train_slice.index)
    hours = index.hour.values
    n_hours, n_objects = values.shape
    n_days = max(n_hours / 24.0, 1.0)

    scale = np.maximum(values.mean(axis=0), 1.0)
    norm = np.log1p(np.maximum(values, 0.0)) / np.log1p(scale)[None, :]

    profile = np.zeros((24, n_objects))
    overall = norm.mean(axis=0)
    for h in range(24):
        m = hours == h
        profile[h] = norm[m].mean(axis=0) if m.any() else overall

    prof_sum = np.maximum(profile.sum(axis=0), 1e-6)
    log_volume = np.log1p(values.sum(axis=0) / n_days)
    peakiness = profile.max(axis=0) / np.maximum(profile.mean(axis=0), 1e-6)
    night_share = profile[NIGHT_HOURS_ARR].sum(axis=0) / prof_sum
    morn_eve = profile[[7, 8, 9]].sum(axis=0) / np.maximum(profile[[17, 18, 19]].sum(axis=0), 1e-6)

    wknd_mask = index.dayofweek.values >= 5
    if wknd_mask.any() and (~wknd_mask).any():
        we_wd = norm[wknd_mask].mean(axis=0) / np.maximum(norm[~wknd_mask].mean(axis=0), 1e-6)
    else:
        we_wd = np.ones(n_objects)

    feats = np.concatenate(
        [profile.T, np.stack([log_volume, peakiness, night_share, morn_eve, we_wd], axis=1)],
        axis=1,
    )
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return np.clip((feats - mean) / std, -5.0, 5.0).astype(np.float32)


def build_graph(train_slice, top_k=8, min_corr=0.05):
    """Возвращает ego-граф: top-k соседей по residual-корреляции на объект.

    neigh_idx  : [N, K] int64  — индексы соседей; слоты без соседа = свой индекс
    neigh_w    : [N, K] float32 — residual-корреляция ребра (0 для пустых слотов)
    node_stats : [N, S] float32 — статические признаки узлов
    """
    values = train_slice.values.astype(np.float64)
    index = pd.DatetimeIndex(train_slice.index)
    n_objects = values.shape[1]

    log_v = np.log1p(np.maximum(values, 0.0))
    resid = log_v - _hourly_profile(log_v, index)
    resid -= resid.mean(axis=0, keepdims=True)
    std = resid.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    rn = resid / std
    corr = (rn.T @ rn) / rn.shape[0]
    np.fill_diagonal(corr, -2.0)

    top_k = min(top_k, max(n_objects - 1, 1))
    order = np.argsort(-corr, axis=1)[:, :top_k]
    rows = np.arange(n_objects)[:, None]
    weights = corr[rows, order].astype(np.float32)
    invalid = weights < min_corr
    neigh_idx = np.where(invalid, rows, order).astype(np.int64)
    neigh_w = np.where(invalid, 0.0, weights).astype(np.float32)

    return {
        "neigh_idx": neigh_idx,
        "neigh_w": neigh_w,
        "node_stats": compute_node_stats(train_slice),
    }


# ── Чекпойнты ────────────────────────────────────────────────────────────────

def save_gat_checkpoint(model, scales, cfg, graph, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "scales": scales,
        "cfg": cfg.__dict__,
        "in_features": model.in_features,
        "dec_features": model.dec_features,
        "stat_dim": model.stat_dim,
        "d_model": model.d_model,
        "horizon": model.horizon,
        "past_window": model.past_window,
        "n_neighbors": model.n_neighbors,
        "n_nodes": model.n_nodes,
        "use_adaptive": model.use_adaptive,
        "adj_emb_dim": model.adj_emb_dim,
        "graph": graph,
    }
    torch.save(checkpoint, filepath)
    print(f"GAT модель и скейлеры сохранены в: {filepath}")


def load_gat_checkpoint(filepath, device="auto"):
    device = resolve_device(device)
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    cfg = GATTrainConfig(**checkpoint["cfg"])
    model = GATForecaster(
        in_features=checkpoint["in_features"],
        dec_features=checkpoint["dec_features"],
        stat_dim=checkpoint["stat_dim"],
        d_model=checkpoint["d_model"],
        n_heads=cfg.n_heads,
        n_neighbors=checkpoint["n_neighbors"],
        past_window=checkpoint["past_window"],
        horizon=checkpoint["horizon"],
        dropout=cfg.dropout,
        neighbor_dropout=cfg.neighbor_dropout,
        n_nodes=checkpoint.get("n_nodes", 1),
        use_adaptive=checkpoint.get("use_adaptive", False),
        adj_emb_dim=checkpoint.get("adj_emb_dim", 16),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["scales"], cfg, checkpoint["graph"]


# ── Dataset: 1 сэмпл = (объект, окно) + ego-граф соседей ──────────────────────
#
# Батчинг по (объект, окно), как у TFT: ~75 окон × 1500 объектов ≈ 110k сэмплов
# на фолд вместо ~75 полнографовых. Это даёт тысячи i.i.d.-шагов оптимизатора
# вместо десятков шагов с внутрибатчевой корреляцией.

class GATWindowDataset(Dataset):
    def __init__(
        self, frames, object_ids, scales, graph, past_window, horizon,
        stride=1, is_synthetic=0, night_weight=0.3,
    ):
        self.object_ids = list(object_ids)
        self.past_window = past_window
        self.horizon = horizon
        self.stride = max(int(stride), 1)
        self.night_weight = float(night_weight)
        self.neigh_idx = graph["neigh_idx"]
        self.neigh_w = graph["neigh_w"].astype(np.float32)
        self.node_stats = graph["node_stats"].astype(np.float32)
        self.samples = []
        self.frames_np = []
        self.covariates_np = []
        self.tod_np = []
        self.dow_np = []

        self.scales_log = np.log1p(
            np.array([scales[oid] for oid in self.object_ids], dtype=np.float32)
        )

        for frame in frames:
            if len(frame) == 0:
                continue
            frame = frame.sort_index()
            self.frames_np.append(frame[self.object_ids].values.astype(np.float32))
            cov = make_time_covariates(frame.index, is_synthetic=is_synthetic)
            self.covariates_np.append(cov.values.astype(np.float32))
            idx = pd.DatetimeIndex(frame.index)
            self.tod_np.append(idx.hour.values.astype(np.int64))
            self.dow_np.append(idx.dayofweek.values.astype(np.int64))

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
        cov = self.covariates_np[fi]
        tod = self.tod_np[fi]
        dow = self.dow_np[fi]

        h_s, h_e = start, start + self.past_window
        t_s, t_e = h_e, h_e + self.horizon

        nidx = self.neigh_idx[ci]
        cols = np.concatenate(([ci], nidx))
        hist = data[h_s:h_e][:, cols]                                # [pw, 1+K]
        sl = self.scales_log[cols]
        hist_norm = np.log1p(hist) / sl[None, :]

        x_self = hist_norm[:, 0]                                     # [pw]
        x_neigh = np.ascontiguousarray(hist_norm[:, 1:].T)           # [K, pw]

        target = data[t_s:t_e, ci]
        y = (np.log1p(target) / sl[0]).astype(np.float32)

        # Сезонная подсказка: последние 24 часа, растиражированные на горизонт
        last_k = min(24, self.past_window)
        last24 = x_self[-last_k:]
        if last_k < 24:
            last24 = np.concatenate(
                [np.full(24 - last_k, last24[0], dtype=np.float32), last24]
            )
        reps = (self.horizon // 24) + 1
        seasonal = np.tile(last24, reps)[: self.horizon]

        enc_cov = cov[h_s:h_e]                                       # [pw, 9]
        dec_cov = cov[t_s:t_e]                                       # [H, 9]

        encoder_x = np.concatenate([x_self[:, None], enc_cov], axis=1)
        n_neigh = x_neigh.shape[0]
        neigh_cov = np.broadcast_to(enc_cov[None], (n_neigh,) + enc_cov.shape)
        neigh_x = np.concatenate([x_neigh[:, :, None], neigh_cov], axis=2)
        decoder_x = np.concatenate([seasonal[:, None], dec_cov], axis=1)

        # Метрики считаются по дневным часам — ночь в лоссе даунвейтится
        target_hours = np.rint(dec_cov[:, 0] * 23.0).astype(int)
        weight = np.where(
            np.isin(target_hours, NIGHT_HOURS_ARR), self.night_weight, 1.0
        ).astype(np.float32)

        return (
            torch.from_numpy(encoder_x.astype(np.float32)),
            torch.from_numpy(neigh_x.astype(np.float32)),
            torch.from_numpy(self.neigh_w[ci]),
            torch.from_numpy(self.node_stats[ci]),
            torch.from_numpy(decoder_x.astype(np.float32)),
            torch.from_numpy(y),
            torch.from_numpy(weight),
            torch.tensor(ci, dtype=torch.long),
            torch.from_numpy(np.ascontiguousarray(nidx).astype(np.int64)),
            torch.from_numpy(tod[h_s:h_e].astype(np.int64)),
            torch.from_numpy(dow[h_s:h_e].astype(np.int64)),
            torch.from_numpy(tod[t_s:t_e].astype(np.int64)),
            torch.from_numpy(dow[t_s:t_e].astype(np.int64)),
        )


# ── Строительные блоки ────────────────────────────────────────────────────────

class GRN(nn.Module):
    """Gated Residual Network (как в TFT)."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x if self.skip is None else self.skip(x)
        h = F.elu(self.fc1(x))
        h = self.drop(h)
        out = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))
        return self.norm(gate * out + (1 - gate) * residual)


class TemporalEncoder(nn.Module):
    """Каузальный dilated Conv1d стек (RF=63). Возвращает ВСЕ T состояний —
    временная ось сохраняется до декодера, никакого сжатия в один вектор."""

    def __init__(self, in_features, d_model, dropout=0.1):
        super().__init__()
        self.dilations = [1, 2, 4, 8, 16]
        self.input_proj = nn.Conv1d(in_features, d_model, 1)
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=3, dilation=d, padding=0)
            for d in self.dilations
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in self.dilations])
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                           # x: [B, T, F]
        h = self.input_proj(x.transpose(1, 2))      # [B, d, T]
        for conv, norm, dil in zip(self.convs, self.norms, self.dilations):
            residual = h
            h = conv(F.pad(h, (2 * dil, 0)))        # левое (каузальное) дополнение
            h = F.gelu(norm(h.transpose(1, 2)).transpose(1, 2))
            h = self.drop(h) + residual
        return h.transpose(1, 2)                    # [B, T, d]


class CrossAttention(nn.Module):
    """Scaled dot-product cross-attention: запросно-зависимое внимание
    (в отличие от статического GATv1-скоринга a_src·h_i + a_dst·h_j)."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, kv, bias=None):
        B, Lq, D = query.shape
        Lk = kv.shape[1]
        q = self.q_proj(query).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(kv).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(kv).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        if bias is not None:
            logits = logits + bias                  # [B,1,1,Lk] broadcast
        attn = self.drop(torch.softmax(logits, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, Lq, D)
        return self.out_proj(out)


# ── EgoGAT Forecaster ─────────────────────────────────────────────────────────
#
# Общий темпоральный энкодер для цели и соседей + per-step декодер:
# каждый шаг горизонта делает cross-attention (a) на свои 72 состояния истории
# и (b) на K×72 развёрнутых во времени состояний соседей — lead-lag между
# объектами выражается напрямую. Пространственный путь гейтится: worst case
# модель деградирует до чисто темпоральной (TFT-паритет), а не до среднего.

class GATForecaster(nn.Module):
    def __init__(
        self, in_features=10, dec_features=10, stat_dim=29, d_model=128,
        n_heads=4, n_neighbors=8, past_window=72, horizon=24,
        dropout=0.1, neighbor_dropout=0.15, n_nodes=1, use_adaptive=True,
        adj_emb_dim=16,
    ):
        super().__init__()
        self.in_features = in_features
        self.dec_features = dec_features
        self.stat_dim = stat_dim
        self.d_model = d_model
        self.horizon = horizon
        self.past_window = past_window
        self.n_neighbors = n_neighbors
        self.neighbor_dropout = neighbor_dropout
        self.n_nodes = n_nodes
        self.use_adaptive = use_adaptive
        self.adj_emb_dim = adj_emb_dim

        self.encoder = TemporalEncoder(in_features, d_model, dropout)
        self.pos_emb = nn.Parameter(torch.zeros(1, past_window, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # ── Adaptive embeddings (STAEformer, CIKM'23) + adaptive adjacency
        #    (Graph WaveNet / AGCRN). Малая инициализация → стартуют около нуля
        #    и включаются по мере обучения, не ломая базовую динамику.
        if use_adaptive:
            self.node_emb = nn.Embedding(n_nodes, d_model)   # идентичность узла
            self.tod_emb = nn.Embedding(24, d_model)         # час суток
            self.dow_emb = nn.Embedding(7, d_model)          # день недели
            self.adj_src = nn.Embedding(n_nodes, adj_emb_dim)  # выученный граф:
            self.adj_dst = nn.Embedding(n_nodes, adj_emb_dim)  # ребро = ⟨e_i, e_j⟩
            for emb in (self.node_emb, self.tod_emb, self.dow_emb,
                        self.adj_src, self.adj_dst):
                nn.init.normal_(emb.weight, std=0.02)

        # FiLM-кондиционирование на статики узла; нулевая инициализация —
        # модуляция включается постепенно по мере обучения
        self.stat_film = nn.Linear(stat_dim, 2 * d_model)
        nn.init.zeros_(self.stat_film.weight)
        nn.init.zeros_(self.stat_film.bias)

        self.h0_proj = nn.Linear(d_model, d_model)
        self.dec_embed = nn.Linear(dec_features, d_model)
        self.dec_gru = nn.GRU(d_model, d_model, batch_first=True)

        self.temporal_attn = CrossAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.spatial_attn = CrossAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.edge_scale = nn.Parameter(torch.tensor(1.0))
        # Гейт стартует прикрытым (sigmoid(-2)≈0.12): сначала темпоральный путь
        self.spatial_gate = nn.Linear(2 * d_model, 1)
        nn.init.constant_(self.spatial_gate.bias, -2.0)

        self.drop = nn.Dropout(dropout)
        self.out_grn = GRN(d_model, d_model, d_model, dropout)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, enc_x, neigh_x, edge_w, node_stat, dec_x,
                node_idx=None, neigh_idx=None, enc_tod=None, enc_dow=None,
                dec_tod=None, dec_dow=None):
        B, K, T, Fin = neigh_x.shape
        adaptive = self.use_adaptive and node_idx is not None

        h_self = self.encoder(enc_x) + self.pos_emb[:, :T]
        gamma, beta = self.stat_film(node_stat).chunk(2, dim=-1)
        h_self = h_self * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        h_neigh = self.encoder(neigh_x.reshape(B * K, T, Fin)) + self.pos_emb[:, :T]
        h_neigh = h_neigh.reshape(B, K, T, self.d_model)

        if adaptive:
            tod_e = self.tod_emb(enc_tod) + self.dow_emb(enc_dow)    # [B, T, d]
            node_e = self.node_emb(node_idx)                         # [B, d]
            h_self = h_self + node_e.unsqueeze(1) + tod_e
            neigh_node_e = self.node_emb(neigh_idx)                  # [B, K, d]
            h_neigh = h_neigh + neigh_node_e.unsqueeze(2) + tod_e.unsqueeze(1)

        h_neigh = h_neigh.reshape(B, K * T, self.d_model)

        h0 = torch.tanh(self.h0_proj(h_self[:, -1])).unsqueeze(0).contiguous()
        dec_in = self.dec_embed(dec_x)
        if adaptive:
            dec_te = self.tod_emb(dec_tod) + self.dow_emb(dec_dow)   # [B, H, d]
            dec_in = dec_in + node_e.unsqueeze(1) + dec_te
        dec_h, _ = self.dec_gru(dec_in, h0)                          # [B, H, d]

        x = self.norm1(dec_h + self.drop(self.temporal_attn(dec_h, h_self)))

        bias = self.edge_scale * edge_w                              # [B, K]
        if adaptive:
            a_src = self.adj_src(node_idx)                           # [B, da]
            a_dst = self.adj_dst(neigh_idx)                          # [B, K, da]
            bias = bias + (a_src.unsqueeze(1) * a_dst).sum(-1) / math.sqrt(self.adj_emb_dim)
        if self.training and self.neighbor_dropout > 0 and K > 1:
            # Neighbor dropout: слот 0 (сильнейший сосед) не глушится — softmax
            # всегда имеет хотя бы один живой ключ. -1e4 безопасно под fp16.
            drop_mask = torch.rand(B, K, device=bias.device) < self.neighbor_dropout
            drop_mask[:, 0] = False
            bias = bias.masked_fill(drop_mask, -1e4)
        bias = bias.repeat_interleave(T, dim=1)[:, None, None, :]    # [B,1,1,K*T]

        spatial = self.spatial_attn(x, h_neigh, bias=bias)
        gate = torch.sigmoid(self.spatial_gate(torch.cat([x, spatial], dim=-1)))
        x = self.norm2(x + gate * self.drop(spatial))

        # Выход = сезонная наивка + предсказанная поправка
        delta = self.out_proj(self.out_grn(x)).squeeze(-1)           # [B, H]
        return dec_x[..., 0] + delta


# ── Конфигурация ──────────────────────────────────────────────────────────────

@dataclass
class GATTrainConfig:
    past_window: int = 72
    d_model: int = 128
    n_heads: int = 4
    top_k_neighbors: int = 8
    min_corr: float = 0.05
    dropout: float = 0.1
    neighbor_dropout: float = 0.15
    night_weight: float = 0.3
    use_adaptive: bool = True
    adj_emb_dim: int = 16
    epochs: int = 12
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    window_stride: int = 1
    synthetic_window_stride: int = 12
    num_workers: int = 0
    device: str = "auto"
    seed: int = 42
    amp: bool = True


# ── Обучение ──────────────────────────────────────────────────────────────────

def train_gat(train_real, synthetic_train, horizon, cfg, graph, pretrained_model=None, scale_source=None, seed_offset=0):
    set_seed(cfg.seed + seed_offset)

    object_ids = list(train_real.columns)
    scales = compute_object_scales(scale_source if scale_source is not None else train_real)

    frames = [train_real]
    strides = [cfg.window_stride]
    if synthetic_train is not None:
        frames.append(synthetic_train[object_ids])
        strides.append(cfg.synthetic_window_stride)

    datasets = [
        GATWindowDataset(
            [frame], object_ids, scales, graph, cfg.past_window, horizon,
            stride=stride, is_synthetic=int(i > 0), night_weight=cfg.night_weight,
        )
        for i, (frame, stride) in enumerate(zip(frames, strides))
    ]
    usable = [d for d in datasets if len(d)]
    if not usable:
        raise ValueError("No GAT training windows.")

    train_dataset = usable[0] if len(usable) == 1 else torch.utils.data.ConcatDataset(usable)
    effective_bs = max(1, min(cfg.batch_size, len(train_dataset)))

    loader = DataLoader(
        train_dataset, batch_size=effective_bs, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False,
    )

    sample = usable[0][0]
    device = resolve_device(cfg.device)

    model = GATForecaster(
        in_features=sample[0].shape[-1],
        dec_features=sample[4].shape[-1],
        stat_dim=graph["node_stats"].shape[1],
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_neighbors=graph["neigh_idx"].shape[1],
        past_window=cfg.past_window,
        horizon=horizon,
        dropout=cfg.dropout,
        neighbor_dropout=cfg.neighbor_dropout,
        n_nodes=len(object_ids),
        use_adaptive=cfg.use_adaptive,
        adj_emb_dim=cfg.adj_emb_dim,
    ).to(device)

    if pretrained_model is not None:
        model.load_state_dict(pretrained_model.state_dict())

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    warmup_epochs = max(1, min(3, cfg.epochs // 4))
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(cfg.epochs - warmup_epochs, 1)
        return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.SmoothL1Loss(beta=0.5, reduction="none")
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    print(f"GAT h={horizon}: {len(train_dataset)} сэмплов, {len(loader)} шагов/эпоху")

    model.train()
    history = []
    for epoch in range(1, cfg.epochs + 1):
        total_loss, n_seen = 0.0, 0
        for batch in loader:
            (enc_x, neigh_x, edge_w, node_stat, dec_x, y, w,
             node_idx, neigh_idx, enc_tod, enc_dow, dec_tod, dec_dow) = batch
            enc_x = enc_x.to(device=device, dtype=torch.float32, non_blocking=True)
            neigh_x = neigh_x.to(device=device, dtype=torch.float32, non_blocking=True)
            edge_w = edge_w.to(device=device, dtype=torch.float32, non_blocking=True)
            node_stat = node_stat.to(device=device, dtype=torch.float32, non_blocking=True)
            dec_x = dec_x.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.float32, non_blocking=True)
            w = w.to(device=device, dtype=torch.float32, non_blocking=True)
            node_idx = node_idx.to(device=device, non_blocking=True)
            neigh_idx = neigh_idx.to(device=device, non_blocking=True)
            enc_tod = enc_tod.to(device=device, non_blocking=True)
            enc_dow = enc_dow.to(device=device, non_blocking=True)
            dec_tod = dec_tod.to(device=device, non_blocking=True)
            dec_dow = dec_dow.to(device=device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                pred = model(enc_x, neigh_x, edge_w, node_stat, dec_x,
                             node_idx, neigh_idx, enc_tod, enc_dow, dec_tod, dec_dow)
                loss = (loss_fn(pred, y) * w).sum() / w.sum().clamp_min(1.0)

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


def train_gat_two_phase(train_real, synthetic_train, horizon, cfg, graph, scale_source=None):
    if synthetic_train is None:
        return train_gat(train_real, None, horizon, cfg, graph, scale_source=scale_source, seed_offset=0)

    synth_cfg = replace(cfg, window_stride=cfg.synthetic_window_stride)
    model, scales, history_phase1 = train_gat(
        synthetic_train[train_real.columns], None, horizon, synth_cfg, graph,
        scale_source=scale_source, seed_offset=0,
    )

    finetune_epochs = max(5, cfg.epochs // 2)
    real_cfg = replace(
        cfg, epochs=finetune_epochs, window_stride=1,
        learning_rate=cfg.learning_rate * 0.3,
    )
    model, scales, history_phase2 = train_gat(
        train_real, None, horizon, real_cfg, graph,
        pretrained_model=model, scale_source=scale_source, seed_offset=1,
    )

    for item in history_phase1:
        item["phase"] = "synth_pretrain"
    for item in history_phase2:
        item["phase"] = "real_finetune"
    return model, scales, history_phase1 + history_phase2


# ── Инференс ──────────────────────────────────────────────────────────────────

def predict_gat_batch(model, history_frame, target_index, scales, cfg, graph, batch_size=256):
    history_frame = history_frame.sort_index()
    object_ids = list(history_frame.columns)
    device = resolve_device(cfg.device)
    model.eval()

    if len(history_frame) >= cfg.past_window:
        hist_slice = history_frame.iloc[-cfg.past_window:]
        values = hist_slice.values.astype(np.float32).T
        hist_index = hist_slice.index
    else:
        missing = cfg.past_window - len(history_frame)
        hist_index = pd.date_range(end=target_index[0] - pd.Timedelta(hours=1), periods=cfg.past_window, freq="h")
        values = []
        for oid in object_ids:
            sv = history_frame[oid].values.astype(np.float32)
            pv = float(sv.mean()) if len(sv) else 0.0
            values.append(np.pad(sv, (missing, 0), constant_values=pv)[-cfg.past_window:])
        values = np.asarray(values, dtype=np.float32)

    horizon = len(target_index)
    n_objects = len(object_ids)
    scale_logs = np.log1p(np.asarray([scales[oid] for oid in object_ids], dtype=np.float32))
    x_val = (np.log1p(values) / scale_logs[:, None]).astype(np.float32)      # [N, pw]

    enc_cov = make_time_covariates(hist_index, is_synthetic=0).values.astype(np.float32)
    enc_cov_b = np.broadcast_to(enc_cov[None], (n_objects,) + enc_cov.shape)
    encoder_x = np.concatenate([x_val[:, :, None], enc_cov_b], axis=2).astype(np.float32)

    neigh_idx = graph["neigh_idx"]
    n_neigh = neigh_idx.shape[1]
    neigh_vals = x_val[neigh_idx]                                            # [N, K, pw]
    neigh_cov = np.broadcast_to(enc_cov[None, None], (n_objects, n_neigh) + enc_cov.shape)
    neigh_x = np.concatenate([neigh_vals[..., None], neigh_cov], axis=3).astype(np.float32)

    edge_w = graph["neigh_w"].astype(np.float32)
    node_stats = graph["node_stats"].astype(np.float32)

    last_k = min(24, x_val.shape[1])
    last24 = x_val[:, -last_k:]
    if last_k < 24:
        pad = np.tile(last24.mean(axis=1, keepdims=True), (1, 24 - last_k))
        last24 = np.concatenate([pad, last24], axis=1)
    reps = (horizon // 24) + 1
    seasonal = np.tile(last24, (1, reps))[:, :horizon]

    dec_cov = make_time_covariates(target_index, is_synthetic=0).values.astype(np.float32)
    dec_cov_b = np.broadcast_to(dec_cov[None], (n_objects,) + dec_cov.shape)
    decoder_x = np.concatenate([seasonal[:, :, None], dec_cov_b], axis=2).astype(np.float32)

    node_idx_all = np.arange(n_objects, dtype=np.int64)
    enc_tod = pd.DatetimeIndex(hist_index).hour.values.astype(np.int64)
    enc_dow = pd.DatetimeIndex(hist_index).dayofweek.values.astype(np.int64)
    dec_tod = pd.DatetimeIndex(target_index).hour.values.astype(np.int64)
    dec_dow = pd.DatetimeIndex(target_index).dayofweek.values.astype(np.int64)

    preds = []
    with torch.no_grad():
        for start in range(0, n_objects, batch_size):
            stop = start + batch_size
            b = min(stop, n_objects) - start
            to_dev = lambda a: torch.from_numpy(a[start:stop]).to(device=device, dtype=torch.float32)
            to_long = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device=device)
            enc_tod_b = to_long(np.broadcast_to(enc_tod[None], (b, enc_tod.shape[0])))
            enc_dow_b = to_long(np.broadcast_to(enc_dow[None], (b, enc_dow.shape[0])))
            dec_tod_b = to_long(np.broadcast_to(dec_tod[None], (b, dec_tod.shape[0])))
            dec_dow_b = to_long(np.broadcast_to(dec_dow[None], (b, dec_dow.shape[0])))
            pred_scaled = model(
                to_dev(encoder_x), to_dev(neigh_x), to_dev(edge_w),
                to_dev(node_stats), to_dev(decoder_x),
                to_long(node_idx_all[start:stop]), to_long(neigh_idx[start:stop]),
                enc_tod_b, enc_dow_b, dec_tod_b, dec_dow_b,
            ).cpu().numpy()

            pred = np.expm1(np.clip(pred_scaled, 0.0, 4.0) * scale_logs[start:stop, None])
            max_pred = np.expm1(scale_logs[start:stop, None]) * 10.0
            preds.append(np.clip(pred, 0.0, max_pred))

    return pd.DataFrame(np.vstack(preds).T, index=target_index, columns=object_ids)


# ── Fast experiment & Rolling backtest ───────────────────────────────────────

def _eval_starts(n_hours, train_hours, horizon, step_hours, max_eval_windows=None):
    starts = list(range(train_hours, n_hours - horizon + 1, step_hours))
    if max_eval_windows is not None and len(starts) > max_eval_windows:
        starts = starts[-max_eval_windows:]
    return starts


def run_gat_fast_experiment(
    pivot_df, horizons=(1, 24), train_modes=("real_only", "real_plus_synth"),
    synth_days=30, train_hours=96, eval_step_1h=1, eval_step_24h=24,
    max_eval_windows_1h=None, max_eval_windows_24h=None, max_objects=None, cfg=None,
):
    cfg = cfg or GATTrainConfig()
    horizons = sorted(set(int(h) for h in horizons))
    max_horizon = max(horizons)
    if train_hours < cfg.past_window + max_horizon:
        raise ValueError("train_hours too small")

    pivot_df = pivot_df.sort_index()
    object_ids = list(pivot_df.columns[:max_objects]) if max_objects else list(pivot_df.columns)
    pivot_df = pivot_df[object_ids]

    train_real = pivot_df.iloc[:train_hours]
    graph = build_graph(train_real, cfg.top_k_neighbors, cfg.min_corr)

    rows, histories, synth_validation_rows = [], [], []
    synthetic_train = None
    if "real_plus_synth" in train_modes:
        dynamic_days = get_synth_days(len(train_real), base_synth_days=synth_days)
        print(f"GAT fast: dynamic synth_days={dynamic_days}")
        synthetic_train = synthesize_from_train(
            train_real, gen_days=dynamic_days, seed=cfg.seed + max_horizon * 1000,
        )
        validation = validate_synthetic(train_real, synthetic_train)
        validation.update({"horizon": max_horizon, "protocol": "fast_single_train"})
        synth_validation_rows.append(validation)

    for train_mode in train_modes:
        active_synth = synthetic_train if train_mode == "real_plus_synth" else None

        if train_mode == "real_plus_synth":
            model, scales, history = train_gat_two_phase(train_real, active_synth, max_horizon, cfg, graph, scale_source=train_real)
        else:
            model, scales, history = train_gat(train_real, None, max_horizon, cfg, graph, scale_source=train_real)

        model_path = config.OUTPUT_DIR / f"gat_model_fast_{train_mode}_h{max_horizon}.pt"
        save_gat_checkpoint(model, scales, cfg, graph, model_path)

        for item in history:
            histories.append({**item, "horizon": max_horizon, "train_mode": train_mode, "model": "GAT", "protocol": "fast_single_train", "train_hours": train_hours})

        active_objects = get_active_objects(train_real, object_ids)

        for eval_horizon in horizons:
            step_hours = eval_step_1h if eval_horizon == 1 else (eval_step_24h if eval_horizon == 24 else eval_horizon)
            max_eval = max_eval_windows_1h if eval_horizon == 1 else (max_eval_windows_24h if eval_horizon == 24 else None)

            starts = _eval_starts(len(pivot_df), train_hours, eval_horizon, step_hours, max_eval)
            for eval_idx, start in enumerate(starts):
                forecast_index = pd.date_range(start=pivot_df.index[start], periods=max_horizon, freq="h")
                y_true = pivot_df.iloc[start: start + eval_horizon]
                history_frame = pivot_df.iloc[:start]
                pred = predict_gat_batch(model, history_frame, forecast_index, scales, cfg, graph).iloc[:eval_horizon]

                day_mask = get_day_mask(y_true.index)
                for oid in active_objects:
                    yt = y_true[oid].values[day_mask]
                    yp = pred[oid].values[day_mask]
                    if len(yt) == 0:
                        continue
                    row = forecast_metrics(yt, yp, model_name="GAT", horizon=eval_horizon, train_mode=train_mode, object_id=oid, fold=eval_idx)
                    row.update({"test_start": y_true.index[0], "test_end": y_true.index[-1], "train_hours": train_hours, "test_data": "real", "protocol": "fast_single_train"})
                    rows.append(row)

    results = pd.DataFrame(rows)
    if results.empty:
        print("⚠ GAT: пустой результат — все окна оценки попали на ночные часы "
              "(день-маска 6:00–23:00). Для h=1 используй шаг, не кратный 24.")
        cols = ["horizon", "train_mode", "model", "MAE", "RMSE", "MAPE", "SMAPE", "WAPE", "n_objects", "n_rows"]
        return results, pd.DataFrame(columns=cols), pd.DataFrame(histories), pd.DataFrame(synth_validation_rows)
    return results, summarize_results(results), pd.DataFrame(histories), pd.DataFrame(synth_validation_rows)


def run_gat_backtest(
    pivot_df, horizon, train_modes=("real_only", "real_plus_synth"), synth_days=30,
    min_train_hours=None, step_hours=None, max_folds=None, max_objects=None, cfg=None,
):
    cfg = cfg or GATTrainConfig()
    pivot_df = pivot_df.sort_index()
    object_ids = list(pivot_df.columns[:max_objects]) if max_objects else list(pivot_df.columns)
    pivot_df = pivot_df[object_ids]

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

        graph = build_graph(real_train_slice, cfg.top_k_neighbors, cfg.min_corr)
        active_objects = get_active_objects(real_train_slice, object_ids)

        for train_mode in train_modes:
            if train_mode == "real_plus_synth" and prepended_synth is not None:
                train_real = pd.concat([prepended_synth, real_train_slice]).sort_index()
                fold_synth = None
                if len(real_train_slice) < 72:
                    dyn_days = get_synth_days(len(real_train_slice), base_synth_days=synth_days)
                    fold_synth = synthesize_from_train(
                        real_train_slice, gen_days=dyn_days, seed=cfg.seed + fold["fold"] + horizon * 1000,
                    )
            else:
                train_real = real_train_slice
                fold_synth = None

            if train_mode == "real_plus_synth":
                model, scales, history = train_gat_two_phase(
                    train_real, fold_synth, horizon, cfg, graph, scale_source=real_train_slice
                )
            else:
                model, scales, history = train_gat(
                    real_train_slice, None, horizon, cfg, graph, scale_source=real_train_slice
                )

            if fold["fold"] == folds[-1]["fold"]:
                model_path = config.OUTPUT_DIR / f"gat_model_rolling_{train_mode}_h{horizon}.pt"
                save_gat_checkpoint(model, scales, cfg, graph, model_path)

            for item in history:
                histories.append({**item, "horizon": horizon, "fold": fold["fold"], "train_mode": train_mode, "model": "GAT"})

            pred_df = predict_gat_batch(model, real_train_slice, target_index, scales, cfg, graph)
            day_mask = get_day_mask(target_index)

            for oid in active_objects:
                yt = test_real[oid].values[day_mask]
                yp = pred_df[oid].values[day_mask]
                if len(yt) == 0:
                    continue
                row = forecast_metrics(yt, yp, model_name="GAT", horizon=horizon, train_mode=train_mode, object_id=oid, fold=fold["fold"])
                row.update({"test_start": target_index[0], "test_end": target_index[-1], "train_hours": len(real_train_slice), "test_data": "real"})
                rows.append(row)

    results = pd.DataFrame(rows)
    if results.empty:
        print("⚠ GAT: пустой результат — все окна оценки попали на ночные часы "
              "(день-маска 6:00–23:00). Для h=1 используй шаг, не кратный 24.")
        cols = ["horizon", "train_mode", "model", "MAE", "RMSE", "MAPE", "SMAPE", "WAPE", "n_objects", "n_rows"]
        return results, pd.DataFrame(columns=cols), pd.DataFrame(histories), pd.DataFrame(synth_validation_rows)
    return results, summarize_results(results), pd.DataFrame(histories), pd.DataFrame(synth_validation_rows)
