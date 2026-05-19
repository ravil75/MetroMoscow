import argparse

from ... import config
from ...clustering import run_clustering
from ...data_prep import create_hourly_parquet, create_object_hourly_parquet, load_hourly, make_pivot
from ...synthesis import save_generation_config
from .pipeline import NBEATSTrainConfig, run_nbeats_backtest, run_nbeats_fast_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="N-BEATS passenger-flow experiment.")
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
    # архитектура N-BEATS
    parser.add_argument("--architecture", choices=["generic", "interpretable"], default="generic")
    parser.add_argument("--num-stacks", type=int, default=3)
    parser.add_argument("--num-blocks-per-stack", type=int, default=3)
    parser.add_argument("--layer-sizes", type=int, default=256)
    parser.add_argument("--polynomial-degree", type=int, default=3)
    parser.add_argument("--num-harmonics", type=int, default=None, help="0 or None = auto (horizon//2).")
    parser.add_argument("--no-covariates", action="store_true", help="Disable future covariate conditioning.")
    # общие гиперпараметры
    parser.add_argument("--past-window", type=int, default=72)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--synthetic-window-stride", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA.")
    # fast-протокол
    parser.add_argument("--eval-step-1h", type=int, default=1)
    parser.add_argument("--eval-step-24h", type=int, default=24)
    parser.add_argument("--max-eval-windows-1h", type=int, default=None)
    parser.add_argument("--max-eval-windows-24h", type=int, default=None)
    # rolling-протокол
    parser.add_argument("--min-train-1h", type=int, default=96)
    parser.add_argument("--step-1h", type=int, default=6)
    parser.add_argument("--min-train-24h", type=int, default=96)
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

    num_harmonics = args.num_harmonics if args.num_harmonics and args.num_harmonics > 0 else None

    train_cfg = NBEATSTrainConfig(
        past_window=args.past_window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        architecture=args.architecture,
        num_stacks=args.num_stacks,
        num_blocks_per_stack=args.num_blocks_per_stack,
        layer_sizes=args.layer_sizes,
        polynomial_degree=args.polynomial_degree,
        num_harmonics=num_harmonics,
        use_covariates=not args.no_covariates,
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
        results, summary, history, synth_validation = run_nbeats_fast_experiment(
            pivot,
            horizons=args.horizons,
            train_modes=args.train_modes,
            synth_days=args.synth_days,
            train_hours=args.train_hours,
            eval_step_1h=args.eval_step_1h,
            eval_step_24h=args.eval_step_24h,
            max_eval_windows_1h=args.max_eval_windows_1h,
            max_eval_windows_24h=args.max_eval_windows_24h,
            max_objects=args.max_objects,
            cfg=train_cfg,
        )

        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for horizon in sorted(results["horizon"].unique()):
            results_path = config.OUTPUT_DIR / f"nbeats_results_{horizon}h.csv"
            summary_path = config.OUTPUT_DIR / f"nbeats_summary_{horizon}h.csv"
            results[results["horizon"] == horizon].to_csv(results_path, index=False)
            summary[summary["horizon"] == horizon].to_csv(summary_path, index=False)
            print(f"saved: {results_path}")
            print(f"saved: {summary_path}")
            print(summary[summary["horizon"] == horizon].to_string(index=False))

        history_path = config.OUTPUT_DIR / "nbeats_history_fast.csv"
        validation_path = config.OUTPUT_DIR / "nbeats_synth_validation_fast.csv"
        history.to_csv(history_path, index=False)
        if len(synth_validation):
            synth_validation.to_csv(validation_path, index=False)
            all_validation.extend(synth_validation.to_dict("records"))
        print(f"saved: {history_path}")
        if len(synth_validation):
            print(f"saved: {validation_path}")

    else:
        for horizon in args.horizons:
            if horizon == 1:
                min_train_hours = args.min_train_1h
                step_hours = args.step_1h
            elif horizon == 24:
                min_train_hours = args.min_train_24h
                step_hours = args.step_24h
            else:
                min_train_hours = max(args.past_window + horizon + 24, 96)
                step_hours = horizon

            results, summary, history, synth_validation = run_nbeats_backtest(
                pivot,
                horizon=horizon,
                train_modes=args.train_modes,
                synth_days=args.synth_days,
                min_train_hours=min_train_hours,
                step_hours=step_hours,
                max_folds=args.max_folds,
                max_objects=args.max_objects,
                cfg=train_cfg,
            )

            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            results_path = config.OUTPUT_DIR / f"nbeats_results_{horizon}h.csv"
            summary_path = config.OUTPUT_DIR / f"nbeats_summary_{horizon}h.csv"
            history_path = config.OUTPUT_DIR / f"nbeats_history_{horizon}h.csv"
            validation_path = config.OUTPUT_DIR / f"nbeats_synth_validation_{horizon}h.csv"

            results.to_csv(results_path, index=False)
            summary.to_csv(summary_path, index=False)
            history.to_csv(history_path, index=False)
            if len(synth_validation):
                synth_validation.to_csv(validation_path, index=False)
                all_validation.extend(synth_validation.to_dict("records"))

            print(f"saved: {results_path}")
            print(f"saved: {summary_path}")
            print(f"saved: {history_path}")
            if len(synth_validation):
                print(f"saved: {validation_path}")
            print(summary.to_string(index=False))

    save_generation_config(
        config.OUTPUT_DIR / "nbeats_generation_config.json",
        args.seed,
        args.synth_days,
        all_validation,
    )


if __name__ == "__main__":
    main()