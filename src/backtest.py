import numpy as np
import pandas as pd

from . import config
from .baseline_models import forecast_metrics, make_model_registry
from .synthesis import synthesize_from_train, validate_synthetic


def make_rolling_folds(n_hours, horizon, min_train_hours=None, step_hours=None):
    if horizon == 1:
        min_train_hours = 96 if min_train_hours is None else min_train_hours
        step_hours = 1 if step_hours is None else step_hours
    elif horizon == 24:
        min_train_hours = 72 if min_train_hours is None else min_train_hours
        step_hours = 24 if step_hours is None else step_hours
    else:
        min_train_hours = max(72, 3 * horizon) if min_train_hours is None else min_train_hours
        step_hours = horizon if step_hours is None else step_hours

    folds = []
    for train_end in range(min_train_hours, n_hours - horizon + 1, step_hours):
        folds.append(
            {
                "fold": len(folds),
                "train_start": 0,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": train_end + horizon,
            }
        )
    return folds


def summarize_results(results_df):
    return (
        results_df.groupby(["horizon", "train_mode", "model"])
        .agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            MAPE=("MAPE", "mean"),
            SMAPE=("SMAPE", "mean"),
            WAPE=("WAPE", "mean"),
            n_objects=("object_id", "nunique"),
            n_rows=("MAE", "count"),
        )
        .round(4)
        .reset_index()
        .sort_values(["horizon", "train_mode", "MAE"])
    )


def run_backtest(
    pivot_df,
    horizon,
    train_modes=("real_only", "real_plus_synth"),
    synth_days=60,
    min_train_hours=None,
    step_hours=None,
    models=None,
    max_objects=None,
    seed=42,
):
    pivot_df = pivot_df.sort_index()
    folds = make_rolling_folds(len(pivot_df), horizon, min_train_hours, step_hours)
    if not folds:
        raise ValueError(f"Not enough data for horizon={horizon}. n_hours={len(pivot_df)}")

    model_registry = make_model_registry()
    if models:
        missing = sorted(set(models) - set(model_registry))
        if missing:
            raise ValueError(f"Unknown models: {missing}. Available: {sorted(model_registry)}")
        model_registry = {name: model_registry[name] for name in models}

    object_ids = list(pivot_df.columns[:max_objects]) if max_objects else list(pivot_df.columns)
    rows = []
    synth_validation_rows = []

    for fold in folds:
        train_real = pivot_df.iloc[fold["train_start"] : fold["train_end"], :][object_ids]
        test_real = pivot_df.iloc[fold["test_start"] : fold["test_end"], :][object_ids]
        target_index = test_real.index

        synthetic_train = None
        if "real_plus_synth" in train_modes:
            synthetic_train = synthesize_from_train(
                train_real,
                gen_days=synth_days,
                seed=seed + fold["fold"] + horizon * 1000,
            )
            validation = validate_synthetic(train_real, synthetic_train)
            validation.update({"fold": fold["fold"], "horizon": horizon})
            synth_validation_rows.append(validation)

        for object_id in object_ids:
            series_real = train_real[object_id]
            y_true = test_real[object_id].values
            synth_series = synthetic_train[object_id] if synthetic_train is not None else None

            for train_mode in train_modes:
                active_synth = synth_series if train_mode == "real_plus_synth" else None
                for model_name, predict_fn in model_registry.items():
                    error = None
                    try:
                        y_pred = predict_fn(series_real, target_index, horizon, active_synth)
                    except Exception as exc:
                        error = repr(exc)
                        y_pred = np.repeat(series_real.iloc[-24:].mean() if len(series_real) >= 24 else series_real.mean(), horizon)
                    metric_row = forecast_metrics(
                        y_true,
                        y_pred,
                        model_name=model_name,
                        horizon=horizon,
                        train_mode=train_mode,
                        object_id=object_id,
                        fold=fold["fold"],
                    )
                    metric_row.update(
                        {
                            "test_start": target_index[0],
                            "test_end": target_index[-1],
                            "train_hours": len(series_real),
                            "test_data": "real",
                            "error": error,
                        }
                    )
                    rows.append(metric_row)

        print(
            f"h={horizon} fold={fold['fold']} train={len(train_real)}h "
            f"test={len(test_real)}h objects={len(object_ids)}"
        )

    results = pd.DataFrame(rows)
    synth_validation = pd.DataFrame(synth_validation_rows)
    return results, summarize_results(results), synth_validation


def save_backtest_outputs(results, summary, synth_validation, horizon):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = config.OUTPUT_DIR / config.RESULTS_TEMPLATE.format(horizon=horizon)
    summary_path = config.OUTPUT_DIR / config.SUMMARY_TEMPLATE.format(horizon=horizon)
    validation_path = config.OUTPUT_DIR / config.SYNTH_VALIDATION_TEMPLATE.format(horizon=horizon)
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    if len(synth_validation):
        synth_validation.to_csv(validation_path, index=False)
    return results_path, summary_path, validation_path
