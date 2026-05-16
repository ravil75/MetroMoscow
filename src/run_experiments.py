import argparse

import pandas as pd

from . import config
from .backtest import run_backtest, save_backtest_outputs
from .clustering import run_clustering
from .data_prep import create_hourly_parquet, create_object_hourly_parquet, load_hourly, make_pivot
from .synthesis import save_generation_config


def parse_args():
    parser = argparse.ArgumentParser(description="Passenger flow forecasting backtest.")
    parser.add_argument("--prepare", action="store_true", help="Create hourly parquet before running experiments.")
    parser.add_argument("--force-prepare", action="store_true", help="Recreate hourly parquet even if it exists.")
    parser.add_argument("--cluster", action="store_true", help="Create final_clusters.csv before synthesis.")
    parser.add_argument("--force-cluster", action="store_true", help="Recreate clusters even if they exist.")
    parser.add_argument("--top-n", type=int, default=config.DEFAULT_TOP_N, help="Number of busiest objects to keep.")
    parser.add_argument("--max-objects", type=int, default=None, help="Optional object cap for quick smoke runs.")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 24], help="Forecast horizons in hours.")
    parser.add_argument("--synth-days", type=int, default=60, help="Synthetic days generated inside every train fold.")
    parser.add_argument("--train-modes", nargs="+", default=["real_only", "real_plus_synth"])
    parser.add_argument("--models", nargs="+", default=None, help="Subset of model names from the registry.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-train-1h", type=int, default=96)
    parser.add_argument("--step-1h", type=int, default=1)
    parser.add_argument("--min-train-24h", type=int, default=72)
    parser.add_argument("--step-24h", type=int, default=24)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.prepare or args.force_prepare or not config.HOURLY_PARQUET.exists():
        create_hourly_parquet(force=args.force_prepare)
    if args.prepare or args.force_prepare or not config.OBJECT_HOURLY_PARQUET.exists():
        create_object_hourly_parquet(force=args.force_prepare)

    if args.cluster or args.force_cluster or not config.CLUSTERS_CSV.exists():
        run_clustering(force=args.force_cluster)

    hourly = load_hourly()
    pivot = make_pivot(hourly, top_n=args.top_n)
    print(f"pivot: {pivot.shape[0]} hours x {pivot.shape[1]} objects")

    all_validation = []
    for horizon in args.horizons:
        if horizon == 1:
            min_train_hours = args.min_train_1h
            step_hours = args.step_1h
        elif horizon == 24:
            min_train_hours = args.min_train_24h
            step_hours = args.step_24h
        else:
            min_train_hours = None
            step_hours = None

        results, summary, synth_validation = run_backtest(
            pivot,
            horizon=horizon,
            train_modes=args.train_modes,
            synth_days=args.synth_days,
            min_train_hours=min_train_hours,
            step_hours=step_hours,
            models=args.models,
            max_objects=args.max_objects,
            seed=args.seed,
        )
        paths = save_backtest_outputs(results, summary, synth_validation, horizon)
        print(f"saved: {paths[0]}")
        print(f"saved: {paths[1]}")
        if len(synth_validation):
            print(f"saved: {paths[2]}")
            all_validation.extend(synth_validation.to_dict("records"))

        print(summary.to_string(index=False))

    save_generation_config(config.GENERATION_CONFIG, args.seed, args.synth_days, all_validation)


if __name__ == "__main__":
    main()
