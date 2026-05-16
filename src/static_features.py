from pathlib import Path

import numpy as np
import pandas as pd

from . import config


STATIC_CATEGORICAL_COLS = [
    "object_id",
    "transport",
    "cluster",
    "cluster_name",
    "LN_CODE",
    "LN_NAME",
]

STATIC_CONTINUOUS_COLS = [
    "daily_volume",
    "log_daily_volume",
    "morn_eve_ratio",
    "we_wd_ratio",
    "peakiness",
    "night_share",
    "midday_share",
]


def _first_non_null(series):
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else np.nan


def _normalize_transport(value):
    mapping = {"Метро": "Metro", "НГПТ": "NGPT", "МЦД": "MCD", "Другое": "Other"}
    return mapping.get(value, value if pd.notna(value) else "Unknown")


def _profile_rows(hourly):
    hourly = hourly.copy()
    hourly["date_hour"] = pd.to_datetime(hourly["date_hour"])
    if "date" not in hourly.columns:
        hourly["date"] = hourly["date_hour"].dt.normalize()
    if "hour" not in hourly.columns:
        hourly["hour"] = hourly["date_hour"].dt.hour
    if "dow" not in hourly.columns:
        hourly["dow"] = hourly["date_hour"].dt.dayofweek
    hourly["is_wknd"] = (hourly["dow"] >= 5).astype(int)

    rows = []
    for object_id, data in hourly.groupby("object_id"):
        weekdays = data[data["is_wknd"] == 0]
        n_weekdays = weekdays["date"].nunique()
        if n_weekdays == 0:
            continue
        weekday_hourly = weekdays.groupby("hour")["pax"].sum() / n_weekdays
        weekday_total = weekday_hourly.sum()
        if weekday_total <= 0:
            continue

        weekends = data[data["is_wknd"] == 1]
        n_weekends = weekends["date"].nunique()
        weekend_hourly = weekends.groupby("hour")["pax"].sum() / n_weekends if n_weekends > 0 else pd.Series(dtype=float)
        weekend_total = weekend_hourly.sum() if n_weekends > 0 else 0.0

        morning = sum(weekday_hourly.get(hour, 0.0) for hour in [7, 8, 9])
        evening = sum(weekday_hourly.get(hour, 0.0) for hour in [17, 18, 19])
        rows.append(
            {
                "object_id": object_id,
                "daily_volume": float(weekday_total),
                "morn_eve_ratio": float(morning / evening) if evening > 0 else 1.0,
                "we_wd_ratio": float(weekend_total / weekday_total) if weekday_total > 0 else 0.0,
                "peakiness": float(weekday_hourly.max() / (weekday_total / 24)),
                "night_share": float(sum(weekday_hourly.get(hour, 0.0) for hour in [23, 0, 1, 2, 3, 4, 5]) / weekday_total),
                "midday_share": float(sum(weekday_hourly.get(hour, 0.0) for hour in [11, 12, 13, 14]) / weekday_total),
            }
        )
    return pd.DataFrame(rows)


def build_static_covariates(hourly, object_ids=None, clusters_path=None):
    """Build static object-level covariates for TFT/GAT/N-BEATS-style models."""
    hourly = hourly.copy()
    hourly["date_hour"] = pd.to_datetime(hourly["date_hour"])
    if object_ids is not None:
        object_ids = list(object_ids)
        hourly = hourly[hourly["object_id"].isin(object_ids)]

    meta = (
        hourly.groupby("object_id")
        .agg(
            transport=("tcat", _first_non_null),
            ST_NAME=("ST_NAME", _first_non_null),
            ROUTE_NAME=("ROUTE_NAME", _first_non_null),
            LN_CODE=("LN_CODE", _first_non_null),
            LN_NAME=("LN_NAME", _first_non_null),
            TYPE_ID=("TYPE_ID", _first_non_null),
        )
        .reset_index()
    )
    meta["transport"] = meta["transport"].map(_normalize_transport)
    meta["object_name"] = meta["ST_NAME"].fillna(meta["ROUTE_NAME"]).fillna(meta["object_id"])

    profiles = _profile_rows(hourly)
    static_df = meta.merge(profiles, on="object_id", how="left")

    path = Path(clusters_path) if clusters_path is not None else config.CLUSTERS_CSV
    if path.exists():
        clusters = pd.read_csv(path)
        if "object_id_str" in clusters.columns:
            cluster_cols = [
                "object_id_str",
                "cluster",
                "cluster_name",
                "daily_volume",
                "log_volume",
                "morn_eve_ratio",
                "we_wd_ratio",
                "peakiness",
                "night_share",
                "midday_share",
            ]
            cluster_cols = [col for col in cluster_cols if col in clusters.columns]
            clusters = clusters[cluster_cols].rename(columns={"object_id_str": "object_id", "log_volume": "cluster_log_volume"})
            static_df = static_df.merge(clusters, on="object_id", how="left", suffixes=("", "_cluster"))

            for col in ["daily_volume", "morn_eve_ratio", "we_wd_ratio", "peakiness", "night_share", "midday_share"]:
                cluster_col = f"{col}_cluster"
                if cluster_col in static_df.columns:
                    static_df[col] = static_df[cluster_col].combine_first(static_df[col])
                    static_df.drop(columns=cluster_col, inplace=True)
            if "cluster_log_volume" in static_df.columns:
                static_df["log_daily_volume"] = static_df["cluster_log_volume"].combine_first(np.log1p(static_df["daily_volume"]))
                static_df.drop(columns="cluster_log_volume", inplace=True)

    if "cluster" not in static_df.columns:
        static_df["cluster"] = -1
    if "cluster_name" not in static_df.columns:
        static_df["cluster_name"] = "Unknown"
    if "log_daily_volume" not in static_df.columns:
        static_df["log_daily_volume"] = np.log1p(static_df["daily_volume"])

    for col in STATIC_CATEGORICAL_COLS:
        if col not in static_df.columns:
            static_df[col] = "Unknown"
        static_df[col] = static_df[col].fillna("Unknown").astype(str)

    for col in STATIC_CONTINUOUS_COLS:
        if col not in static_df.columns:
            static_df[col] = 0.0
        median = static_df[col].median()
        static_df[col] = static_df[col].fillna(0.0 if pd.isna(median) else median).astype(float)

    static_df = static_df.set_index("object_id", drop=False)
    if object_ids is not None:
        static_df = static_df.reindex(object_ids)
        static_df["object_id"] = static_df.index
        for col in STATIC_CATEGORICAL_COLS:
            static_df[col] = static_df[col].fillna("Unknown").astype(str)
        for col in STATIC_CONTINUOUS_COLS:
            static_df[col] = static_df[col].fillna(0.0).astype(float)
    return static_df


def encode_static_covariates(static_df, categorical_cols=None, continuous_cols=None):
    """Encode static covariates into categorical codes and normalized continuous values."""
    categorical_cols = categorical_cols or STATIC_CATEGORICAL_COLS
    continuous_cols = continuous_cols or STATIC_CONTINUOUS_COLS
    static_df = static_df.copy()

    categorical_arrays = []
    vocabularies = {}
    cardinalities = {}
    for col in categorical_cols:
        values = static_df[col].fillna("Unknown").astype(str)
        categories = sorted(values.unique().tolist())
        mapping = {value: idx for idx, value in enumerate(categories)}
        vocabularies[col] = mapping
        cardinalities[col] = len(mapping)
        categorical_arrays.append(values.map(mapping).astype(np.int64).values)

    cat_matrix = np.stack(categorical_arrays, axis=1) if categorical_arrays else np.empty((len(static_df), 0), dtype=np.int64)

    cont = static_df[continuous_cols].astype(float)
    means = cont.mean(axis=0)
    stds = cont.std(axis=0).replace(0, 1.0).fillna(1.0)
    cont_matrix = ((cont - means) / stds).astype(np.float32).values

    return {
        "object_ids": static_df.index.tolist(),
        "categorical": cat_matrix,
        "continuous": cont_matrix,
        "categorical_cols": categorical_cols,
        "continuous_cols": continuous_cols,
        "vocabularies": vocabularies,
        "cardinalities": cardinalities,
        "continuous_mean": means.to_dict(),
        "continuous_std": stds.to_dict(),
    }


def make_node_feature_matrix(static_df, one_hot_categoricals=("transport", "cluster")):
    """Create a numeric node-feature matrix for GAT baselines."""
    continuous = static_df[STATIC_CONTINUOUS_COLS].astype(float)
    continuous = (continuous - continuous.mean(axis=0)) / continuous.std(axis=0).replace(0, 1.0).fillna(1.0)
    parts = [continuous.fillna(0.0)]
    for col in one_hot_categoricals:
        parts.append(pd.get_dummies(static_df[col].fillna("Unknown").astype(str), prefix=col, dtype=float))
    return pd.concat(parts, axis=1).astype(np.float32)
