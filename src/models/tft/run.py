import argparse
import pandas as pd

from ... import config
from ...clustering import run_clustering
from ...data_prep import create_hourly_parquet, create_object_hourly_parquet, load_hourly, make_pivot
from ...synthesis import save_generation_config
from ...static_features import build_static_covariates, encode_static_covariates
from .pipeline import TFTTrainConfig, run_tft_backtest, run_tft_fast_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="TFT passenger-flow experiment.")
    parser.add_argument("--prepare", action="store_true", help="Create hourly parquet before running.")
    parser.add_argument("--force-prepare", action="store_true", help="Recreate hourly parquet.")
    parser.add_argument("--cluster", action="store_true", help="Create final_clusters.csv for cluster-aware synthesis.")
    parser.add_argument("--force-cluster", action="store_true", help="Recreate final_clusters.csv.")
    parser.add_argument("--top-n", type=int, default=config.DEFAULT_TOP_N)
    parser.add_argument("--max-objects", type=int, default=None, help="Use only the first N busiest objects.")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 24])
    parser.add_argument("--protocol", choices=["fast", "rolling"], default="fast")
    parser.add_argument("--train-modes", nargs="+", default=["real_only", "real_plus_synth"])
    parser.add_argument("--synth-days", type=int, default=30)
    parser.add_argument("--train-hours", type=int, default=96, help="Fast protocol real train length.")
    
    # TFT специфичные параметры
    parser.add_argument("--past-window", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    
    # Общие параметры
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--synthetic-window-stride", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA.")
    
    # Параметры бэктеста
    parser.add_argument("--eval-step-1h", type=int, default=1)
    parser.add_argument("--eval-step-24h", type=int, default=24)
    parser.add_argument("--max-eval-windows-1h", type=int, default=None)
    parser.add_argument("--max-eval-windows-24h", type=int, default=None)
    parser.add_argument("--min-train-1h", type=int, default=144)
    parser.add_argument("--step-1h", type=int, default=6)
    parser.add_argument("--min-train-24h", type=int, default=144)
    parser.add_argument("--step-24h", type=int, default=24)
    parser.add_argument("--max-folds", type=int, default=None)
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

    # ИЗВЛЕКАЕМ СТАТИКУ НАПРЯМУЮ ИЗ HOURLY
    object_ids = list(pivot.columns[:args.max_objects]) if args.max_objects else list(pivot.columns)
    print("Генерация статических признаков...")
    static_df = build_static_covariates(hourly, object_ids=object_ids)
    static_enc = encode_static_covariates(static_df)

    train_cfg = TFTTrainConfig(
        past_window=args.past_window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        dropout=args.dropout,
        window_stride=args.window_stride,
        synthetic_window_stride=args.synthetic_window_stride,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        amp=not args.no_amp,
    )

    all_validation = []
    if args.protocol == "fast":
        results, summary, history, synth_validation = run_tft_fast_experiment(
            pivot, static_enc, horizons=args.horizons, train_modes=args.train_modes,
            synth_days=args.synth_days, train_hours=args.train_hours,
            eval_step_1h=args.eval_step_1h, eval_step_24h=args.eval_step_24h,
            max_eval_windows_1h=args.max_eval_windows_1h, max_eval_windows_24h=args.max_eval_windows_24h,
            max_objects=args.max_objects, cfg=train_cfg,
        )

        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for horizon in sorted(results["horizon"].unique()):
            results_path = config.OUTPUT_DIR / f"tft_results_{horizon}h.csv"
            summary_path = config.OUTPUT_DIR / f"tft_summary_{horizon}h.csv"
            results[results["horizon"] == horizon].to_csv(results_path, index=False)
            summary[summary["horizon"] == horizon].to_csv(summary_path, index=False)
            print(f"saved: {results_path}")
            print(summary[summary["horizon"] == horizon].to_string(index=False))

    else:
        for horizon in args.horizons:
            if horizon == 1:
                min_train_hours, step_hours = args.min_train_1h, args.step_1h
            elif horizon == 24:
                min_train_hours, step_hours = args.min_train_24h, args.step_24h
            else:
                min_train_hours, step_hours = max(args.past_window + horizon + 24, 96), horizon

            results, summary, history, synth_validation = run_tft_backtest(
                pivot, static_enc, horizon=horizon, train_modes=args.train_modes,
                synth_days=args.synth_days, min_train_hours=min_train_hours,
                step_hours=step_hours, max_folds=args.max_folds,
                max_objects=args.max_objects, cfg=train_cfg,
            )

            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            results_path = config.OUTPUT_DIR / f"tft_results_{horizon}h.csv"
            summary_path = config.OUTPUT_DIR / f"tft_summary_{horizon}h.csv"
            results.to_csv(results_path, index=False)
            summary.to_csv(summary_path, index=False)

            print(f"saved: {results_path}")
            print(summary.to_string(index=False))

    save_generation_config(config.OUTPUT_DIR / "tft_generation_config.json", args.seed, args.synth_days, all_validation)

if __name__ == "__main__":
    main()