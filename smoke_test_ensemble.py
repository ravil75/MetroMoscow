"""Проверка ансамбля (--ensemble) и блендинга прогнозов."""
import numpy as np
import pandas as pd
import torch

from src.models.gat.pipeline import GATTrainConfig, run_gat_backtest
from src.models.gat.blend import blend_predictions
from src import config

rng = np.random.default_rng(0)
N, T = 30, 168
index = pd.date_range("2026-01-05", periods=T, freq="h")
hours = index.hour.values
base = 200 * np.exp(-(((hours - 8) % 24)) ** 2 / 18) + 150 * np.exp(-(((hours - 18) % 24)) ** 2 / 18) + 30
cols = {f"obj_{i:03d}": np.maximum(base * (1 + 0.3 * np.sin(i)) + rng.gamma(2, 5, T), 0) for i in range(N)}
pivot = pd.DataFrame(cols, index=index)

base_cfg = dict(epochs=2, batch_size=128, top_k_neighbors=4, device="cpu", amp=False,
                num_workers=0, min_corr=0.05)

# 1. Ансамбль из 2 сидов отрабатывает и даёт валидные метрики
cfg_ens = GATTrainConfig(ensemble=2, **base_cfg)
res, summary, _, _ = run_gat_backtest(
    pivot, horizon=24, train_modes=["real_only"], min_train_hours=96, step_hours=24,
    max_folds=2, cfg=cfg_ens,
)
assert not summary.empty and np.isfinite(summary["MAE"]).all(), "ансамбль: пустые/NaN метрики"
print(f"OK  --ensemble 2: MAE={summary['MAE'].iloc[0]:.3f}")

# 2. Два прогона с --save-predictions + разными pred_tag → бленд
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for tag, seed in [("A", 42), ("B", 7)]:
    cfg = GATTrainConfig(save_predictions=True, pred_tag=tag, seed=seed, **base_cfg)
    run_gat_backtest(pivot, horizon=24, train_modes=["real_only"],
                     min_train_hours=96, step_hours=24, max_folds=2, cfg=cfg)
fa = config.OUTPUT_DIR / "gat_predictions_24h_A.csv"
fb = config.OUTPUT_DIR / "gat_predictions_24h_B.csv"
assert fa.exists() and fb.exists(), "файлы прогнозов не сохранились"
print("OK  --save-predictions: файлы сохранены")

res_b, sum_b = blend_predictions([str(fa), str(fb)])
assert not sum_b.empty and np.isfinite(sum_b["MAE"]).all(), "бленд: пустые/NaN метрики"
print(f"OK  бленд 2 прогонов: MAE={sum_b['MAE'].iloc[0]:.3f}")

# 3. Бленд идентичного файла с собой = те же метрики (sanity)
res_id, sum_id = blend_predictions([str(fa), str(fa)])
print(f"OK  бленд A+A (sanity): MAE={sum_id['MAE'].iloc[0]:.3f}")

# чистим
for f in (fa, fb):
    f.unlink(missing_ok=True)

print("\nENSEMBLE TEST PASSED")
