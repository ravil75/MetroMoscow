"""Проверка TimeMixer-энкодера (ICLR'24) как опции."""
import numpy as np
import pandas as pd
import torch

from src.models.gat.pipeline import (
    GATTrainConfig, build_graph, compute_object_scales, train_gat,
    predict_gat_batch, TimeMixerEncoder, _SeriesDecomp,
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

# 1. Декомпозиция: сезон+тренд = исходный ряд, тренд глаже
dec = _SeriesDecomp(kernel=13)
x = torch.randn(4, 72, 8).cumsum(dim=1)  # ряд с трендом
season, trend = dec(x)
assert torch.allclose(season + trend, x, atol=1e-4), "сезон+тренд != исходный ряд"
assert trend.diff(dim=1).abs().mean() < x.diff(dim=1).abs().mean(), "тренд не глаже исходного"
print("OK  SeriesDecomp: сезон+тренд=ряд, тренд глаже")

# 2. TimeMixerEncoder: форма выхода [B,T,d] (тонкий масштаб)
enc = TimeMixerEncoder(in_features=10, d_model=64, n_scales=3, n_blocks=2)
h = enc(torch.randn(6, 72, 10))
assert h.shape == (6, 72, 64), f"неверная форма: {h.shape}"
print("OK  TimeMixerEncoder: форма выхода [B,T,d]")

# 3. Выбор энкодера по флагу + полный train/predict на обоих горизонтах
for h_ in (24, 1):
    target = pd.date_range(pivot.index[120], periods=h_, freq="h")
    for use_tm in (False, True):
        cfg = GATTrainConfig(epochs=2, batch_size=64, top_k_neighbors=4, device="cpu",
                             amp=False, num_workers=0, use_timemixer=use_tm,
                             timemixer_scales=3, timemixer_blocks=2)
        model, sc, hist = train_gat(pivot.iloc[:120], None, h_, cfg, graph)
        is_tm = isinstance(model.encoder, TimeMixerEncoder)
        assert is_tm == use_tm, f"энкодер не по флагу (h={h_}, timemixer={use_tm})"
        pred = predict_gat_batch(model, pivot.iloc[:120], target, sc, cfg, graph)
        ok = np.isfinite(pred.values).all() and (pred.values >= 0).all() and pred.shape == (h_, N)
        assert ok, f"некорректный прогноз (h={h_}, timemixer={use_tm})"
        print(f"OK  train/predict h={h_:>2} enc={'TimeMixer' if is_tm else 'dilated  '}  loss={hist[-1]['loss']:.4f}")

print("\nTIMEMIXER TEST PASSED")
