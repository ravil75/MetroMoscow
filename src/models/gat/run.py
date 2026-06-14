import argparse

from ... import config
from ...clustering import run_clustering
from ...data_prep import (
    create_hourly_parquet,
    create_object_hourly_parquet,
    load_hourly,
    make_pivot,
)
from ...synthesis import save_generation_config
from .pipeline import GATTrainConfig, run_gat_backtest, run_gat_fast_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="GAT passenger-flow experiment.")

    # Подготовка данных
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--force-cluster", action="store_true")

    # Данные
    parser.add_argument("--top-n", type=int, default=config.DEFAULT_TOP_N)
    parser.add_argument("--max-objects", type=int, default=None)

    # Протокол оценки
    parser.add_argument("--protocol", choices=["fast", "rolling"], default="rolling")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 24])
    parser.add_argument("--train-modes", nargs="+", default=["real_plus_synth"])

    # Rolling backtest
    parser.add_argument("--min-train-1h", type=int, default=96)
    parser.add_argument("--step-1h", type=int, default=6)
    parser.add_argument("--min-train-24h", type=int, default=96)
    parser.add_argument("--step-24h", type=int, default=24)
    parser.add_argument("--max-folds", type=int, default=None)

    # Fast protocol
    parser.add_argument("--train-hours", type=int, default=96)
    parser.add_argument("--eval-step-1h", type=int, default=1)
    parser.add_argument("--eval-step-24h", type=int, default=24)
    parser.add_argument("--max-eval-windows-1h", type=int, default=None)
    parser.add_argument("--max-eval-windows-24h", type=int, default=None)

    # Синтетика
    parser.add_argument("--synth-days", type=int, default=45)
    parser.add_argument("--synthetic-window-stride", type=int, default=12)

    # Архитектура GAT (EgoGAT)
    parser.add_argument("--past-window", type=int, default=72)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--top-k-neighbors", type=int, default=8)
    parser.add_argument(
        "--min-corr", type=float, default=0.05,
        help="Порог residual-корреляции для рёбер ego-графа.",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--neighbor-dropout", type=float, default=0.15,
        help="Вероятность глушения соседа в пространственном внимании при обучении.",
    )
    parser.add_argument(
        "--night-weight", type=float, default=0.3,
        help="Вес ночных часов (0-5) в лоссе; метрики считаются по дневным.",
    )
    parser.add_argument(
        "--no-adaptive", action="store_true",
        help="Отключить ОБЕ идеи из статей — вернуться к baseline-EgoGAT для A/B.",
    )
    parser.add_argument(
        "--no-adaptive-embed", action="store_true",
        help="Отключить только идею 1 — adaptive embeddings (STAEformer, CIKM'23).",
    )
    parser.add_argument(
        "--no-adaptive-adj", action="store_true",
        help="Отключить только идею 2 — adaptive adjacency (Graph WaveNet/AGCRN).",
    )
    parser.add_argument(
        "--bidirectional-encoder", action="store_true",
        help="Двунаправленный энкодер истории (как в STD-MAE). Сильнее каузального "
             "при masked-предобучении; история полностью наблюдаема — leak'а нет.",
    )
    parser.add_argument(
        "--pretrain-epochs", type=int, default=0,
        help="Эпохи masked-предобучения энкодера (STD-MAE'24 / GPT-ST'23). 0 = выкл.",
    )
    parser.add_argument(
        "--pretrain-mask-ratio", type=float, default=0.4,
        help="Доля маскируемых временных шагов при предобучении.",
    )

    # Обучение
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Число сэмплов (объект × окно) в батче, как у TFT.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Подготовка данных ─────────────────────────────────────────────────────
    if args.prepare or args.force_prepare or not config.HOURLY_PARQUET.exists():
        create_hourly_parquet(force=args.force_prepare)
    if args.prepare or args.force_prepare or not config.OBJECT_HOURLY_PARQUET.exists():
        create_object_hourly_parquet(force=args.force_prepare)
    if args.cluster or args.force_cluster or not config.CLUSTERS_CSV.exists():
        run_clustering(force=args.force_cluster)

    hourly = load_hourly()
    pivot = make_pivot(hourly, top_n=args.top_n)
    print(f"pivot: {pivot.shape[0]} hours x {pivot.shape[1]} objects")

    # ── Конфигурация ──────────────────────────────────────────────────────────
    train_cfg = GATTrainConfig(
        past_window=args.past_window,
        d_model=args.d_model,
        n_heads=args.n_heads,
        top_k_neighbors=args.top_k_neighbors,
        min_corr=args.min_corr,
        dropout=args.dropout,
        neighbor_dropout=args.neighbor_dropout,
        night_weight=args.night_weight,
        use_adaptive_embed=not (args.no_adaptive or args.no_adaptive_embed),
        use_adaptive_adj=not (args.no_adaptive or args.no_adaptive_adj),
        bidirectional_encoder=args.bidirectional_encoder,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_mask_ratio=args.pretrain_mask_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        window_stride=args.window_stride,
        synthetic_window_stride=args.synthetic_window_stride,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        amp=not args.no_amp,
    )

    all_validation = []

    if args.protocol == "fast":
        results, summary, history, synth_validation = run_gat_fast_experiment(
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
            results_path = config.OUTPUT_DIR / f"gat_results_{horizon}h.csv"
            summary_path = config.OUTPUT_DIR / f"gat_summary_{horizon}h.csv"
            results[results["horizon"] == horizon].to_csv(results_path, index=False)
            summary[summary["horizon"] == horizon].to_csv(summary_path, index=False)
            print(f"saved: {results_path}")
            print(f"saved: {summary_path}")
            print(summary[summary["horizon"] == horizon].to_string(index=False))

        history_path = config.OUTPUT_DIR / "gat_history_fast.csv"
        validation_path = config.OUTPUT_DIR / "gat_synth_validation_fast.csv"
        history.to_csv(history_path, index=False)
        if len(synth_validation):
            synth_validation.to_csv(validation_path, index=False)
            all_validation.extend(synth_validation.to_dict("records"))
        print(f"saved: {history_path}")
        if len(synth_validation):
            print(f"saved: {validation_path}")

    else:
        # ── Rolling backtest ──────────────────────────────────────────────────
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

            results, summary, history, synth_validation = run_gat_backtest(
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
            results_path = config.OUTPUT_DIR / f"gat_results_{horizon}h.csv"
            summary_path = config.OUTPUT_DIR / f"gat_summary_{horizon}h.csv"
            history_path = config.OUTPUT_DIR / f"gat_history_{horizon}h.csv"
            validation_path = config.OUTPUT_DIR / f"gat_synth_validation_{horizon}h.csv"

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
        config.OUTPUT_DIR / "gat_generation_config.json",
        args.seed,
        args.synth_days,
        all_validation,
    )


if __name__ == "__main__":
    main()