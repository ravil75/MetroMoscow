"""Smoke-тест нового EgoGAT пайплайна на синтетических данных (без реального датасета)."""
import numpy as np
import pandas as pd
import torch

from src.models.gat.pipeline import (
    GATTrainConfig, GATWindowDataset, GATForecaster,
    build_graph, compute_object_scales, train_gat, predict_gat_batch,
    save_gat_checkpoint, load_gat_checkpoint,
)

rng = np.random.default_rng(0)
N, T = 40, 168
index = pd.date_range("2026-01-05", periods=T, freq="h")

# Суточный профиль + лидер-фолловер пары (объект 2i+1 повторяет 2i со сдвигом 1 час)
hours = index.hour.values
base = 200 * np.exp(-((hours - 8) % 24 - 0) ** 2 / 18) + 150 * np.exp(-((hours - 18) % 24) ** 2 / 18) + 30
cols = {}
for i in range(N // 2):
    shock = rng.normal(0, 0.25, T).cumsum() * 0.1
    lead = base * (1 + 0.3 * np.sin(i)) * np.exp(shock) + rng.gamma(2, 5, T)
    follow = np.roll(lead, 1) * 0.8 + rng.gamma(2, 5, T)
    cols[f"obj_{2*i:03d}"] = np.maximum(lead, 0)
    cols[f"obj_{2*i+1:03d}"] = np.maximum(follow, 0)
pivot = pd.DataFrame(cols, index=index)

# 1. Граф
graph = build_graph(pivot.iloc[:120], top_k=4, min_corr=0.05)
assert graph["neigh_idx"].shape == (N, 4)
assert graph["node_stats"].shape[0] == N
print("build_graph OK:", graph["neigh_idx"].shape, "node_stats:", graph["node_stats"].shape)
# Лидер должен быть среди соседей фолловера
hit = sum(2 * i in graph["neigh_idx"][2 * i + 1] for i in range(N // 2))
print(f"лидер найден среди соседей фолловера: {hit}/{N//2}")

# 2. Dataset (новый тапл: + node_idx, neigh_idx, tod/dow для enc и dec)
scales = compute_object_scales(pivot.iloc[:120])
ds = GATWindowDataset([pivot.iloc[:120]], list(pivot.columns), scales, graph, 72, 24)
enc_x, neigh_x, edge_w, node_stat, dec_x, y, w, node_idx, neigh_idx, e_tod, e_dow, d_tod, d_dow = ds[0]
assert enc_x.shape == (72, 10) and neigh_x.shape == (4, 72, 10)
assert dec_x.shape == (24, 10) and y.shape == (24,) and w.shape == (24,)
assert neigh_idx.shape == (4,) and e_tod.shape == (72,) and d_tod.shape == (24,)
assert int(d_tod.max()) <= 23 and int(d_dow.max()) <= 6
print(f"dataset OK: {len(ds)} сэмплов; node_idx={int(node_idx)}, соседи={neigh_idx.tolist()}")

# 3. Модель: forward + train (CPU, 2 эпохи) — adaptive embeddings + adaptive adjacency
cfg = GATTrainConfig(epochs=2, batch_size=64, top_k_neighbors=4, device="cpu", amp=False, num_workers=0)
model, scales, history = train_gat(pivot.iloc[:120], None, 24, cfg, graph)
assert len(history) == 2 and np.isfinite(history[-1]["loss"])
print("train OK (adaptive), loss:", [round(h["loss"], 4) for h in history])

# 3b. Baseline-режим (--no-adaptive) тоже должен работать
cfg_base = GATTrainConfig(epochs=1, batch_size=64, top_k_neighbors=4, device="cpu", amp=False, num_workers=0, use_adaptive=False)
model_base, _, hist_base = train_gat(pivot.iloc[:120], None, 24, cfg_base, graph)
assert np.isfinite(hist_base[-1]["loss"])
print("train OK (baseline, no-adaptive), loss:", round(hist_base[-1]["loss"], 4))

# 4. Инференс: оба горизонта
target_24 = pd.date_range(pivot.index[120], periods=24, freq="h")
pred24 = predict_gat_batch(model, pivot.iloc[:120], target_24, scales, cfg, graph)
assert pred24.shape == (24, N) and np.isfinite(pred24.values).all() and (pred24.values >= 0).all()
target_1 = target_24[:1]
pred1 = predict_gat_batch(model, pivot.iloc[:120], target_1, scales, cfg, graph)
assert pred1.shape == (1, N) and np.isfinite(pred1.values).all()
print("predict OK: h24", pred24.shape, "h1", pred1.shape)

# 5. Чекпойнт: save/load + идентичность прогноза
path = "smoke_gat_ckpt.pt"
save_gat_checkpoint(model, scales, cfg, graph, path)
model2, scales2, cfg2, graph2 = load_gat_checkpoint(path, device="cpu")
pred24b = predict_gat_batch(model2, pivot.iloc[:120], target_24, scales2, cfg2, graph2)
assert np.allclose(pred24.values, pred24b.values, atol=1e-4)
print("checkpoint OK: прогнозы идентичны")

# 6. Sanity: MAE против seasonal naive на дневных часах
actual = pivot.iloc[120:144]
naive = pivot.iloc[96:120].values
day = np.array([ts.hour >= 6 for ts in target_24])
mae_model = np.abs(pred24.values[day] - actual.values[day]).mean()
mae_naive = np.abs(naive[day] - actual.values[day]).mean()
print(f"MAE (день): EgoGAT={mae_model:.2f} vs seasonal-naive={mae_naive:.2f} (2 эпохи CPU)")
print("\nSMOKE TEST PASSED")
