"""Проверка TimesNet-энкодера (ICLR'23) как опции."""
import numpy as np
import pandas as pd
import torch

from src.models.gat.pipeline import (
    GATTrainConfig, build_graph, compute_object_scales, train_gat,
    predict_gat_batch, TimesNetEncoder, TemporalEncoder, _fft_top_periods,
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

# 1. FFT находит суточный период (24) на синтетике с суточным циклом
t = torch.arange(72).float()
daily = torch.sin(2 * np.pi * t / 24).view(1, 72, 1).repeat(4, 1, 8)
periods, weights = _fft_top_periods(daily, k=2)
assert 24 in periods, f"FFT не нашёл суточный период: {periods}"
print(f"OK  FFT-периоды на суточном сигнале: {periods}")

# 2. TimesNetEncoder: форма выхода [B,T,d], та же что у TemporalEncoder
enc = TimesNetEncoder(in_features=10, d_model=64, n_blocks=2, k_periods=2)
x = torch.randn(6, 72, 10)
h = enc(x)
assert h.shape == (6, 72, 64), f"неверная форма выхода TimesNet: {h.shape}"
print("OK  TimesNetEncoder: форма выхода [B,T,d]")

# 3. Выбор энкодера по флагу + полный train/predict на обоих горизонтах
for h_ in (24, 1):
    target = pd.date_range(pivot.index[120], periods=h_, freq="h")
    for use_tn in (False, True):
        cfg = GATTrainConfig(epochs=2, batch_size=64, top_k_neighbors=4, device="cpu",
                             amp=False, num_workers=0, use_timesnet=use_tn,
                             timesnet_blocks=2, timesnet_k=2)
        model, sc, hist = train_gat(pivot.iloc[:120], None, h_, cfg, graph)
        is_tn = isinstance(model.encoder, TimesNetEncoder)
        assert is_tn == use_tn, f"энкодер не по флагу (h={h_}, timesnet={use_tn})"
        pred = predict_gat_batch(model, pivot.iloc[:120], target, sc, cfg, graph)
        ok = np.isfinite(pred.values).all() and (pred.values >= 0).all() and pred.shape == (h_, N)
        assert ok, f"некорректный прогноз (h={h_}, timesnet={use_tn})"
        enc_name = "TimesNet" if is_tn else "dilated "
        print(f"OK  train/predict h={h_:>2} enc={enc_name}  loss={hist[-1]['loss']:.4f}")

print("\nTIMESNET TEST PASSED")
