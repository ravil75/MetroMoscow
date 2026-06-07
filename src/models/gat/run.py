# src/models/gat/run.py

import argparse
import pandas as pd

from ... import config
from ...clustering import run_clustering
from ...data_prep import create_hourly_parquet, create_object_hourly_parquet, load_hourly, make_pivot
from ...synthesis import save_generation_config
from .pipeline import GATTrainConfig, run_gat_backtest

def parse_args():
    parser = argparse.ArgumentParser(description="GAT passenger-flow experiment.")

    parser.add_argument("--prepare",       action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--cluster",       action="store_true")
    parser.add_argument("--force-cluster", action="store_true")

    parser.add_argument("--top-n",       type=int,   default=config.DEFAULT_TOP_N)
    parser.add_argument("--max-objects", type=int,   default=None)

    parser.add_argument("--protocol",    choices=["rolling"], default="rolling")
    parser.add_argument("--horizons",    type=int, nargs="+", default=[1, 24])
    parser.add_argument("--train-modes", nargs="+", default=["real_plus_synth"])

    parser.add_argument("--min-train-1h",  type=int, default=96)
    parser.add_argument("--step-1h",       type=int, default=6)
    parser.add_argument("--min-train-24h", type=int, default=96)
    parser.add_argument("--step-24h",      type=int, default=24)
    parser.add_argument("--max-folds",     type=int, default=None)

    parser.add_argument("--synth-days",              type=int,   default=45)
    parser.add_argument("--synthetic-window-stride", type=int,   default=24)

    # Архитектура GAT
    parser.add_argument("--past-window",  type=int,   default=72)
    parser.add_argument("--d-model",      type=int,   default=64, help="Hidden dim for GAT")
    parser.add_argument("--n-heads",      type=int,   default=4)
    parser.add_argument("--lstm-layers",  type=int,   default=2)
    parser.add_argument("--dropout",      type=float, default=0.1)

    # Обучение
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--batch-size",    type=int,   default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay",  type=float, default=1e-4)
    parser.add_argument("--window-stride", type=int,   default=1)
    parser.add_argument("--num-workers",   type=int,   default=0)
    parser.add_argument("--device",        default="auto")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--no-amp",        action="store_true")

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
    pivot  = make_pivot(hourly, top_n=args.top_n)
    print(f"pivot: {pivot.shape[0]} hours x {pivot.shape[1]} objects")

    # ВАЖНО: В GAT мы обрабатываем весь граф (1500 объектов) разом за один сэмпл.
    # Batch Size = 1024 вызовет OutOfMemory. Корректируем его до адекватного для графов.
    actual_batch_size = args.batch_size
    if actual_batch_size >= 128:
        actual_batch_size = 16
        print(f"⚠️ Внимание: Для GAT batch_size автоматически уменьшен с {args.batch_size} до {actual_batch_size}, "
              f"так как один сэмпл включает в себя матрицу всего графа ({pivot.shape[1]} узлов).")

    cfg = GATTrainConfig(
        past_window              = args.past_window,
        hidden_dim               = args.d_model,
        n_heads                  = args.n_heads,
        lstm_layers              = args.lstm_layers,
        dropout                  = args.dropout,
        epochs                   = args.epochs,
        batch_size               = actual_batch_size,
        learning_rate            = args.learning_rate,
        weight_decay             = args.weight_decay,
        window_stride            = args.window_stride,
        synthetic_window_stride  = args.synthetic_window_stride,
        num_workers              = args.num_workers,
        device                   = args.device,
        seed                     = args.seed,
        amp                      = not args.no_amp,
    )

    all_validation = []

    for horizon in args.horizons:
        if horizon == 1:
            min_train_hours = args.min_train_1h
            step_hours      = args.step_1h
        elif horizon == 24:
            min_train_hours = args.min_train_24h
            step_hours      = args.step_24h
        else:
            min_train_hours = max(args.past_window + horizon + 24, 96)
            step_hours      = horizon

        results, summary, history, synth_val = run_gat_backtest(
            pivot,
            horizon         = horizon,
            train_modes     = args.train_modes,
            synth_days      = args.synth_days,
            min_train_hours = min_train_hours,
            step_hours      = step_hours,
            max_folds       = args.max_folds,
            max_objects     = args.max_objects,
            cfg             = cfg,
        )

        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        results_path  = config.OUTPUT_DIR / f"gat_results_{horizon}h.csv"
        summary_path  = config.OUTPUT_DIR / f"gat_summary_{horizon}h.csv"
        history_path  = config.OUTPUT_DIR / f"gat_history_{horizon}h.csv"
        val_path      = config.OUTPUT_DIR / f"gat_synth_validation_{horizon}h.csv"

        results.to_csv(results_path,  index=False)
        summary.to_csv(summary_path,  index=False)
        history.to_csv(history_path,  index=False)

        print(f"saved: {results_path}")
        print(f"saved: {summary_path}")
        if len(synth_val):
            synth_val.to_csv(val_path, index=False)
            all_validation.extend(synth_val.to_dict("records"))

        print(summary.to_string(index=False))

    save_generation_config(config.OUTPUT_DIR / "gat_generation_config.json", args.seed, args.synth_days, all_validation)

if __name__ == "__main__":
    main()