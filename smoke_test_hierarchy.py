"""Проверка иерархии (станция→кластер→итог) и MinT-согласования."""
import numpy as np
import pandas as pd

from src.models.gat.pipeline import GATTrainConfig, run_gat_backtest
from src.models.gat.hierarchy import (
    build_summing_matrix, mint_projection, augment_with_aggregates, AGG_PREFIX,
)
from src.models.gat.reconcile import reconcile
from src import config

# 1. Математика MinT: инвариант G·S = I (когерентные прогнозы не меняются)
station_ids = [f"obj_{i:03d}" for i in range(12)]
cluster_map = {o: i % 3 for i, o in enumerate(station_ids)}
S, node_ids = build_summing_matrix(station_ids, cluster_map)
assert S.shape == (12 + 3 + 1, 12), f"неверная форма S: {S.shape}"
for method in ("structural", "ols"):
    G = mint_projection(S, method)
    assert np.allclose(G @ S, np.eye(12), atol=1e-8), f"G·S != I ({method})"
print("OK  MinT: формы и инвариант G·S=I (structural+ols)")

# 2. augment: агрегатные колонки = суммы членов
idx = pd.date_range("2026-01-05", periods=48, freq="h")
pivot = pd.DataFrame({o: np.random.rand(48) * 100 for o in station_ids}, index=idx)
aug = augment_with_aggregates(pivot, cluster_map)
agg_cols = [c for c in aug.columns if c.startswith(AGG_PREFIX)]
assert len(agg_cols) == 4, f"ожидалось 4 агрегата, получили {len(agg_cols)}"
assert np.allclose(aug[f"{AGG_PREFIX}TOTAL"].values, pivot.sum(axis=1).values), "итог != сумма станций"
cl0 = [o for o in station_ids if cluster_map[o] == 0]
assert np.allclose(aug[f"{AGG_PREFIX}CL0"].values, pivot[cl0].sum(axis=1).values), "кластер != сумма членов"
print(f"OK  augment: {len(agg_cols)} агрегатов, суммы корректны")

# 3. Полный прогон с --hierarchy → reconcile
N, T = 30, 168
hours = pd.date_range("2026-01-05", periods=T, freq="h").hour.values
base = 200 * np.exp(-(((hours - 8) % 24)) ** 2 / 18) + 150 * np.exp(-(((hours - 18) % 24)) ** 2 / 18) + 30
rng = np.random.default_rng(0)
cols = {f"obj_{i:03d}": np.maximum(base * (1 + 0.3 * np.sin(i)) + rng.gamma(2, 5, T), 0) for i in range(N)}
pivot = pd.DataFrame(cols, index=pd.date_range("2026-01-05", periods=T, freq="h"))

# временный clusters CSV (3 кластера)
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
clusters_csv = config.CLUSTERS_CSV
backup = clusters_csv.with_suffix(".bak") if clusters_csv.exists() else None
if backup:
    clusters_csv.rename(backup)
pd.DataFrame({"object_id_str": list(cols), "cluster": [i % 3 for i in range(N)]}).to_csv(clusters_csv, index=False)

try:
    cfg = GATTrainConfig(epochs=2, batch_size=128, top_k_neighbors=4, device="cpu",
                         amp=False, num_workers=0, min_corr=0.05, hierarchy=True)
    res, summary, _, _ = run_gat_backtest(
        pivot, horizon=24, train_modes=["real_only"], min_train_hours=96, step_hours=24,
        max_folds=2, cfg=cfg,
    )
    assert not summary.empty, "пустые метрики иерархии"
    pred_file = config.OUTPUT_DIR / "gat_predictions_24h.csv"
    assert pred_file.exists(), "файл прогнозов не сохранён"
    pdf = pd.read_csv(pred_file)
    n_agg = pdf["object_id"].astype(str).str.startswith(AGG_PREFIX).sum()
    assert n_agg > 0, "агрегаты не сохранены"
    print(f"OK  --hierarchy: метрики по {N} станциям, агрегаты в прогнозах ({n_agg} строк)")

    base_s, recon_s = reconcile(str(pred_file))
    assert not recon_s.empty and np.isfinite(recon_s["MAE"]).all(), "MinT: пустые/NaN метрики"
    print(f"OK  reconcile (MinT): base MAE={base_s['MAE'].mean():.2f}, recon MAE={recon_s['MAE'].mean():.2f}")
    pred_file.unlink(missing_ok=True)
finally:
    clusters_csv.unlink(missing_ok=True)
    if backup:
        backup.rename(clusters_csv)

print("\nHIERARCHY TEST PASSED")
