"""Проверка раздельных флагов adaptive: 4 комбинации (embed × adj)."""
import numpy as np
import pandas as pd
import torch

from src.models.gat.pipeline import (
    GATTrainConfig, build_graph, compute_object_scales, train_gat,
    predict_gat_batch, GATForecaster,
)

rng = np.random.default_rng(0)
N, T = 30, 168
index = pd.date_range("2026-01-05", periods=T, freq="h")
hours = index.hour.values
base = 200 * np.exp(-(((hours - 8) % 24)) ** 2 / 18) + 150 * np.exp(-(((hours - 18) % 24)) ** 2 / 18) + 30
cols = {f"obj_{i:03d}": np.maximum(base * (1 + 0.3 * np.sin(i)) + rng.gamma(2, 5, T), 0) for i in range(N)}
pivot = pd.DataFrame(cols, index=index)
graph = build_graph(pivot.iloc[:120], top_k=4, min_corr=0.05)
scales = compute_object_scales(pivot.iloc[:120])
target = pd.date_range(pivot.index[120], periods=24, freq="h")

combos = [
    ("обе идеи",       True,  True),
    ("только embed",   True,  False),
    ("только adj",     False, True),
    ("без обеих",      False, False),
]

for name, emb, adj in combos:
    cfg = GATTrainConfig(
        epochs=2, batch_size=64, top_k_neighbors=4, device="cpu", amp=False,
        num_workers=0, use_adaptive_embed=emb, use_adaptive_adj=adj,
    )
    model, sc, hist = train_gat(pivot.iloc[:120], None, 24, cfg, graph)
    # наличие параметров строго по флагам
    has_node = hasattr(model, "node_emb")
    has_adj = hasattr(model, "adj_src")
    assert has_node == emb, f"{name}: node_emb={has_node}, ожидалось {emb}"
    assert has_adj == adj, f"{name}: adj_src={has_adj}, ожидалось {adj}"
    pred = predict_gat_batch(model, pivot.iloc[:120], target, sc, cfg, graph)
    ok = np.isfinite(pred.values).all() and (pred.values >= 0).all()
    assert ok and pred.shape == (24, N), f"{name}: некорректный прогноз"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"OK  {name:14s} embed={emb} adj={adj}  params={n_params:>7d}  loss={hist[-1]['loss']:.4f}")

print("\nABLATION TEST PASSED")
