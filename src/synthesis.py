import json

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from statsmodels.tsa.stattools import acf

from . import config


def generate_global_warp(n_hours=24, n_knots=4, sigma=0.06, low=0.75, high=1.25):
    x = np.linspace(0, n_hours - 1, n_knots)
    y = np.random.normal(1.0, sigma, size=n_knots)
    return np.clip(CubicSpline(x, y)(np.arange(n_hours)), low, high)


def generate_cluster_warp(n_hours=24, n_knots=4, sigma=0.04, low=-0.20, high=0.20):
    x = np.linspace(0, n_hours - 1, n_knots)
    y = np.random.normal(0.0, sigma, size=n_knots)
    return np.clip(CubicSpline(x, y)(np.arange(n_hours)), low, high)


def build_day_blocks(pivot_df):
    matrix = pivot_df.values
    n_days = len(pivot_df) // 24
    blocks = {day: [] for day in range(7)}
    for day_idx in range(n_days):
        start = day_idx * 24
        end = start + 24
        if end > len(pivot_df):
            break
        day_of_week = pivot_df.index[start].dayofweek
        blocks[day_of_week].append(matrix[start:end, :])
    return blocks


def load_cluster_mapping():
    if not config.CLUSTERS_CSV.exists():
        return {}
    clusters = pd.read_csv(config.CLUSTERS_CSV)
    if "object_id_str" not in clusters.columns:
        return {}
    return dict(zip(clusters["object_id_str"], clusters["cluster"]))


def _choose_foundation_block(day_blocks, target_dow, is_holiday):
    if is_holiday:
        candidates = day_blocks.get(6, [])
    else:
        candidates = day_blocks.get(target_dow, [])

    if not candidates:
        candidates = [block for blocks in day_blocks.values() for block in blocks]
    if not candidates:
        raise ValueError("Cannot synthesize data: no full 24-hour real blocks in train fold.")
    return candidates[np.random.randint(len(candidates))].copy()


def synthesize_from_train(
    train_pivot,
    gen_days=30,
    seed=42,
    use_cluster_warp=True,
    use_anomalies=True,
    use_trends=True,
):
    if len(train_pivot) < 24:
        raise ValueError("At least 24 training hours are required for synthetic generation.")

    np.random.seed(seed)
    train_pivot = train_pivot.sort_index()
    day_blocks = build_day_blocks(train_pivot)
    object_ids = list(train_pivot.columns)
    obj_to_cluster = load_cluster_mapping()
    clusters = sorted({obj_to_cluster.get(object_id, -1) for object_id in object_ids})

    last_real_time = train_pivot.index[-1]
    synth_start = (last_real_time + pd.Timedelta(hours=1)).normalize()
    if synth_start <= last_real_time:
        synth_start += pd.Timedelta(days=1)
    global_trend = np.random.uniform(-0.04, 0.04) if use_trends else 0.0
    cluster_trends = {cluster: np.random.uniform(-0.03, 0.03) for cluster in clusters} if use_trends else {}
    object_trend_noise = np.random.normal(0.0, 0.01, size=len(object_ids)) if use_trends else np.zeros(len(object_ids))

    synth_days = []
    for day_idx in range(gen_days):
        target_date = synth_start + pd.Timedelta(days=day_idx)
        target_dow = target_date.dayofweek
        is_holiday = (target_date.month, target_date.day) in config.HOLIDAYS_MD

        chosen_day = _choose_foundation_block(day_blocks, target_dow, is_holiday)
        if is_holiday:
            chosen_day *= np.random.uniform(0.75, 0.90)

        global_warp = generate_global_warp(sigma=0.06)
        warped_day = chosen_day * global_warp[:, None]

        if use_cluster_warp:
            cluster_warps = {cluster: generate_cluster_warp(sigma=0.04) for cluster in clusters}
            for col_idx, object_id in enumerate(object_ids):
                cluster = obj_to_cluster.get(object_id, -1)
                warped_day[:, col_idx] *= 1.0 + cluster_warps[cluster]

        warped_day = np.maximum(warped_day, 0)

        if use_anomalies and np.random.random() < 0.10 and clusters:
            spike_hour = np.random.randint(7, 22)
            spike_cluster = np.random.choice(clusters)
            spike_magnitude = np.random.uniform(1.4, 1.8)
            for col_idx, object_id in enumerate(object_ids):
                if obj_to_cluster.get(object_id, -1) == spike_cluster:
                    base = warped_day[spike_hour, col_idx]
                    warped_day[spike_hour, col_idx] = base * spike_magnitude + max(50.0, 2.0 * np.sqrt(base))

        if use_trends:
            for col_idx, object_id in enumerate(object_ids):
                cluster = obj_to_cluster.get(object_id, -1)
                local_trend = global_trend + cluster_trends.get(cluster, 0.0) + object_trend_noise[col_idx]
                warped_day[:, col_idx] *= 1.0 + local_trend * (day_idx / max(gen_days, 1))

        synth_days.append(np.random.poisson(np.maximum(warped_day, 0)).astype(np.float32))

    synth_matrix = np.concatenate(synth_days, axis=0)
    synth_index = pd.date_range(
        start=synth_start,
        periods=gen_days * 24,
        freq="h",
    )
    return pd.DataFrame(synth_matrix, columns=object_ids, index=synth_index)


def validate_synthetic(train_pivot, synth_pivot, sample_hours=168):
    real = train_pivot.iloc[-min(sample_hours, len(train_pivot)) :]
    synth = synth_pivot.iloc[: min(sample_hours, len(synth_pivot))]
    common_cols = real.columns.intersection(synth.columns)
    real = real[common_cols]
    synth = synth[common_cols]

    metrics = {}
    if len(real) > 3 and len(synth) > 3:
        nlags = min(48, len(real) // 2 - 1, len(synth) // 2 - 1)
        if nlags > 1:
            real_acf = acf(real.mean(axis=1).values, nlags=nlags, fft=True)[1:]
            synth_acf = acf(synth.mean(axis=1).values, nlags=nlags, fft=True)[1:]
            metrics["acf_similarity"] = float(1.0 - np.mean(np.abs(real_acf - synth_acf)))
        else:
            metrics["acf_similarity"] = np.nan
    else:
        metrics["acf_similarity"] = np.nan

    real_profile = real.groupby(real.index.hour).mean().mean(axis=1)
    synth_profile = synth.groupby(synth.index.hour).mean().mean(axis=1)
    metrics["profile_corr"] = float(real_profile.corr(synth_profile))

    real_values = real.values.ravel()
    synth_values = synth.values.ravel()
    real_positive = real_values[real_values > 0]
    synth_positive = synth_values[synth_values > 0]
    if len(real_positive) and len(synth_positive):
        real_q = np.quantile(real_positive, [0.25, 0.50, 0.99])
        synth_q = np.quantile(synth_positive, [0.25, 0.50, 0.99])
        metrics["quantile_ratio"] = float(np.mean(synth_q / (real_q + 1e-6)))
    else:
        metrics["quantile_ratio"] = np.nan

    if len(common_cols) > 1 and len(real) > 2 and len(synth) > 2:
        real_corr = np.corrcoef(real.T)
        synth_corr = np.corrcoef(synth.T)
        mask = ~np.eye(real_corr.shape[0], dtype=bool)
        metrics["corr_mae"] = float(np.nanmean(np.abs(real_corr[mask] - synth_corr[mask])))
        metrics["corr_spearman"] = float(
            pd.Series(real_corr[mask]).corr(pd.Series(synth_corr[mask]), method="spearman")
        )
    else:
        metrics["corr_mae"] = np.nan
        metrics["corr_spearman"] = np.nan

    metrics["ready_for_training"] = bool(
        (np.isnan(metrics["acf_similarity"]) or metrics["acf_similarity"] > 0.80)
        and metrics["profile_corr"] > 0.85
        and 0.65 < metrics["quantile_ratio"] < 1.50
        and (np.isnan(metrics["corr_mae"]) or metrics["corr_mae"] < 0.20)
    )
    return metrics


def save_generation_config(path, seed, gen_days, validation_rows):
    payload = {
        "seed": seed,
        "gen_days": gen_days,
        "method": "day-block bootstrap + global/cluster magnitude warping + poisson noise",
        "global_warp_sigma": 0.06,
        "cluster_warp_sigma": 0.04,
        "anomaly_probability": 0.10,
        "important_note": "Synthetic data must be generated inside each train fold only.",
        "validation": validation_rows,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)
