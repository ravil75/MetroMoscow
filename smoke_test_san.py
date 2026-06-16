"""Проверка SAN (Slice Adaptive Normalization) как опционального плагина."""
import numpy as np
import pandas as pd
import torch

from src.models.gat.pipeline import (
    GATTrainConfig, build_graph, compute_object_scales, train_gat,
    predict_gat_batch, GATForecaster, SAN,
)

rng = np.random.default_rng(0)
N, T = 30, 168
index = pd.date_range("2026-01-05", periods=T, freq="h")
hours = index.hour.values
base = 200 * np.exp(-(((hours - 8) % 24)) ** 2 / 18) + 150 * np.exp(-(((hours - 18) % 24)) ** 2 / 18) + 30
# добавим нестационарность: линейный дрейф уровня по дням
drift = 1 + 0.04 * (np.arange(T) // 24)
cols = {f"obj_{i:03d}": np.maximum(base * (1 + 0.3 * np.sin(i)) * drift + rng.gamma(2, 5, T), 0) for i in range(N)}
pivot = pd.DataFrame(cols, index=index)
graph = build_graph(pivot.iloc[:120], top_k=4, min_corr=0.05)
scales = compute_object_scales(pivot.iloc[:120])

# 1. SAN-модуль напрямую: формы и персистентный старт
san = SAN(period=24, past_window=72, horizon=24)
v = torch.randn(8, 72)
v_norm, mu, sd = san.slice_norm(v)
assert v_norm.shape == (8, 72) and mu.shape == (8, 3) and sd.shape == (8, 3)
fmu, fsd = san.future_stats(mu, sd)
assert fmu.shape == (8, 24) and fsd.shape == (8, 24)
# при нулевой инициализации выходов → персистентность последнего слайса
assert torch.allclose(fmu, mu[:, -1:].expand(-1, 24), atol=1e-5), "SAN: старт не персистентный (mean)"
assert torch.allclose(fsd, sd[:, -1:].expand(-1, 24), atol=1e-5), "SAN: старт не персистентный (std)"
print("OK  SAN-модуль: формы и персистентный старт")

# 2. h=1 (горизонт < периода) — denorm не должен падать
san1 = SAN(period=24, past_window=72, horizon=1)
f1m, f1s = san1.future_stats(*san1.slice_norm(torch.randn(4, 72))[1:])
assert f1m.shape == (4, 1) and f1s.shape == (4, 1)
print("OK  SAN h=1: горизонт меньше периода обрабатывается")

# 3. Наличие san-модуля строго по флагу + полный train/predict на обоих горизонтах
for h in (24, 1):
    target = pd.date_range(pivot.index[120], periods=h, freq="h")
    for use_san in (False, True):
        cfg = GATTrainConfig(epochs=2, batch_size=64, top_k_neighbors=4, device="cpu",
                             amp=False, num_workers=0, use_san=use_san, san_period=24)
        model, sc, hist = train_gat(pivot.iloc[:120], None, h, cfg, graph)
        assert (model.san is not None) == use_san, f"san-модуль не по флагу (h={h}, san={use_san})"
        pred = predict_gat_batch(model, pivot.iloc[:120], target, sc, cfg, graph)
        ok = np.isfinite(pred.values).all() and (pred.values >= 0).all() and pred.shape == (h, N)
        assert ok, f"некорректный прогноз (h={h}, san={use_san})"
        print(f"OK  train/predict h={h:>2} san={str(use_san):5}  loss={hist[-1]['loss']:.4f}")

print("\nSAN TEST PASSED")
