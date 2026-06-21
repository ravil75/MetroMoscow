"""MinT-согласование сохранённых иерархических прогнозов GAT.

Берёт файл прогнозов от прогона с --hierarchy (станции + агрегаты), строит
суммирующую матрицу S и MinT-проекцию, согласует прогнозы станций (оптимально
блендит с прогнозами кластеров/итога) и сравнивает метрики base vs reconciled.

Пример:
  python -m src.models.gat.reconcile eda_output/gat_predictions_24h.csv
"""
import argparse

import numpy as np
import pandas as pd

from ...backtest import summarize_results
from ...baseline_models import forecast_metrics
from .hierarchy import AGG_PREFIX, build_summing_matrix, mint_projection, load_cluster_map


def _metrics(df_long, model_name):
    rows = []
    for (oid, fold, horizon, tmode), g in df_long.groupby(
        ["object_id", "fold", "horizon", "train_mode"], sort=False
    ):
        rows.append(forecast_metrics(
            g["y_true"].values, g["y_pred"].values, model_name=model_name,
            horizon=int(horizon), train_mode=tmode, object_id=oid, fold=int(fold),
        ))
    return summarize_results(pd.DataFrame(rows))


def reconcile(pred_path, method="structural", out_path=None):
    df = pd.read_csv(pred_path)
    df["object_id"] = df["object_id"].astype(str)
    is_agg = df["object_id"].str.startswith(AGG_PREFIX)
    station_ids = sorted(df.loc[~is_agg, "object_id"].unique())
    cluster_map = load_cluster_map(station_ids)
    S, node_ids = build_summing_matrix(station_ids, cluster_map)
    G = mint_projection(S, method=method)
    print(f"Иерархия: {len(station_ids)} станций, {S.shape[0] - len(station_ids)} агрегатов; MinT({method})")

    recon_long = []
    for (horizon, fold, tmode), sub in df.groupby(["horizon", "fold", "train_mode"], sort=False):
        wide_p = sub.pivot_table(index="object_id", columns="timestamp", values="y_pred")
        wide_t = sub.pivot_table(index="object_id", columns="timestamp", values="y_true")
        cols = wide_p.columns
        Y = np.nan_to_num(wide_p.reindex(node_ids).values)      # [n × T] базовые прогнозы
        recon = np.maximum(G @ Y, 0.0)                          # [m × T] согласованные станции
        for i, sid in enumerate(station_ids):
            yt = wide_t.reindex([sid]).values[0]
            for j, ts in enumerate(cols):
                recon_long.append({"horizon": horizon, "fold": fold, "train_mode": tmode,
                                   "object_id": sid, "y_true": yt[j], "y_pred": recon[i, j]})

    base_long = df.loc[~is_agg, ["horizon", "fold", "train_mode", "object_id", "y_true", "y_pred"]]
    base_summary = _metrics(base_long, "GAT-base")
    recon_summary = _metrics(pd.DataFrame(recon_long), "GAT-MinT")

    print("\n── BASE (без согласования) ──")
    print(base_summary.to_string(index=False))
    print("\n── RECONCILED (MinT) ──")
    print(recon_summary.to_string(index=False))
    delta = base_summary["MAE"].mean() - recon_summary["MAE"].mean()
    print(f"\nΔMAE (base − MinT) = {delta:+.4f}  →  {'MinT помог' if delta > 0 else 'MinT не помог'}")

    if out_path:
        recon_summary.to_csv(out_path, index=False)
        print(f"saved: {out_path}")
    return base_summary, recon_summary


def main():
    ap = argparse.ArgumentParser(description="MinT-согласование прогнозов GAT.")
    ap.add_argument("predictions", help="CSV прогнозов от прогона с --hierarchy.")
    ap.add_argument("--method", choices=["structural", "ols"], default="structural")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    reconcile(args.predictions, method=args.method, out_path=args.out)


if __name__ == "__main__":
    main()
