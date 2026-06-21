"""Бленд (ансамбль) прогнозов нескольких прогонов GAT.

Усредняет y_pred из ≥2 файлов прогнозов (сохранённых с --save-predictions) по
ключам (horizon, fold, train_mode, object_id, timestamp), затем пересчитывает
метрики тем же forecast_metrics, что и пайплайн. Каждый прогон гоняется в своей
сессии (<12ч), а бленд — мгновенный.

Пример:
  python -m src.models.gat.blend \
    eda_output/gat_predictions_24h_dilated.csv \
    eda_output/gat_predictions_24h_timesnet.csv
"""
import argparse

import pandas as pd

from ...backtest import summarize_results
from ...baseline_models import forecast_metrics

KEYS = ["horizon", "fold", "train_mode", "object_id", "timestamp"]


def blend_predictions(pred_paths, weights=None, out_path=None):
    dfs = [pd.read_csv(p) for p in pred_paths]
    n = len(dfs)
    weights = weights or [1.0 / n] * n
    if abs(sum(weights) - 1.0) > 1e-6:                       # нормируем веса
        s = sum(weights)
        weights = [w / s for w in weights]

    blended = dfs[0][KEYS + ["y_true"]].copy()
    blended["y_pred"] = 0.0
    for w, df in zip(weights, dfs):
        d = df[KEYS + ["y_pred"]].rename(columns={"y_pred": "_yp"})
        blended = blended.merge(d, on=KEYS, how="inner")
        blended["y_pred"] += w * blended["_yp"]
        blended = blended.drop(columns="_yp")

    if blended.empty:
        raise ValueError("Нет общих (fold, object, timestamp) между файлами — "
                         "проверь, что прогоны на одних горизонте/фолдах/данных.")

    rows = []
    for (oid, fold, horizon, tmode), g in blended.groupby(
        ["object_id", "fold", "horizon", "train_mode"], sort=False
    ):
        row = forecast_metrics(
            g["y_true"].values, g["y_pred"].values, model_name="GAT-blend",
            horizon=int(horizon), train_mode=tmode, object_id=oid, fold=int(fold),
        )
        rows.append(row)

    results = pd.DataFrame(rows)
    summary = summarize_results(results)
    print(f"\nБленд {n} прогонов (веса {[round(w, 3) for w in weights]}):")
    print(summary.to_string(index=False))
    if out_path:
        summary.to_csv(out_path, index=False)
        print(f"saved: {out_path}")
    return results, summary


def main():
    ap = argparse.ArgumentParser(description="Бленд прогнозов GAT.")
    ap.add_argument("preds", nargs="+", help="CSV-файлы прогнозов (≥2).")
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="Веса прогонов (по умолчанию равные).")
    ap.add_argument("--out", default=None, help="Куда сохранить summary бленда.")
    args = ap.parse_args()
    if len(args.preds) < 2:
        ap.error("нужно ≥2 файлов прогнозов")
    if args.weights and len(args.weights) != len(args.preds):
        ap.error("число весов должно совпадать с числом файлов")
    blend_predictions(args.preds, args.weights, args.out)


if __name__ == "__main__":
    main()
