import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config
from .data_prep import clean_id, load_hourly


def build_profiles(subset, id_col, name_col):
    subset = subset.copy()
    subset["is_wknd"] = (subset["dow"] >= 5).astype(int)
    rows = []

    for object_id in subset[id_col].dropna().unique():
        data = subset[subset[id_col] == object_id]
        name = data[name_col].iloc[0] if name_col in data.columns else "?"

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
        weekend_total = weekend_hourly.sum() if n_weekends > 0 else 0

        row = {"object_id": object_id, "object_name": name, "daily_volume": weekday_total}
        for hour in range(24):
            row[f"wd_h{hour}"] = weekday_hourly.get(hour, 0) / weekday_total
            row[f"we_h{hour}"] = weekend_hourly.get(hour, 0) / weekend_total if weekend_total > 0 else 0

        morning = sum(weekday_hourly.get(hour, 0) for hour in [7, 8, 9])
        evening = sum(weekday_hourly.get(hour, 0) for hour in [17, 18, 19])
        row["morn_eve_ratio"] = morning / evening if evening > 0 else 1.0
        row["we_wd_ratio"] = weekend_total / weekday_total if weekday_total > 0 else 0
        row["peakiness"] = weekday_hourly.max() / (weekday_total / 24)
        row["night_share"] = sum(weekday_hourly.get(hour, 0) for hour in [23, 0, 1, 2, 3, 4, 5]) / weekday_total
        row["midday_share"] = sum(weekday_hourly.get(hour, 0) for hour in [11, 12, 13, 14]) / weekday_total
        rows.append(row)

    return pd.DataFrame(rows)


def _object_id_str(row):
    prefix = "RT_" if row["transport"] == "NGPT" else "ST_"
    return f"{prefix}{clean_id(row['object_id'])}"


def run_clustering(force=False):
    if config.CLUSTERS_CSV.exists() and not force:
        print(f"{config.CLUSTERS_CSV} already exists. Skipping clustering.")
        return config.CLUSTERS_CSV

    hourly = load_hourly()
    parts = []

    metro = hourly[hourly["tcat"] == "Metro"]
    if len(metro):
        metro_features = build_profiles(metro, "ST_CODE", "ST_NAME")
        metro_features["transport"] = "Metro"
        station_map = metro.drop_duplicates("ST_CODE").set_index("ST_CODE")
        metro_features["LN_CODE"] = metro_features["object_id"].map(station_map["LN_CODE"])
        metro_features["LN_NAME"] = metro_features["object_id"].map(station_map["LN_NAME"])
        parts.append(metro_features)

    ngpt = hourly[hourly["tcat"] == "NGPT"]
    if len(ngpt):
        ngpt_features = build_profiles(ngpt, "BUS_RT_NO", "ROUTE_NAME")
        ngpt_features["transport"] = "NGPT"
        ngpt_features["LN_CODE"] = None
        ngpt_features["LN_NAME"] = None
        parts.append(ngpt_features)

    mcd = hourly[hourly["tcat"] == "MCD"]
    if mcd["ST_CODE"].nunique() > 2:
        mcd_features = build_profiles(mcd, "ST_CODE", "ST_NAME")
        mcd_features["transport"] = "MCD"
        station_map = mcd.drop_duplicates("ST_CODE").set_index("ST_CODE")
        mcd_features["LN_CODE"] = mcd_features["object_id"].map(station_map["LN_CODE"])
        mcd_features["LN_NAME"] = mcd_features["object_id"].map(station_map["LN_NAME"])
        parts.append(mcd_features)

    if not parts:
        raise RuntimeError("No transport objects found for clustering.")

    features = pd.concat(parts, ignore_index=True)
    filtered = features[
        (features["daily_volume"] >= 50)
        & (features["peakiness"] < 15)
        & (features["we_wd_ratio"] > 0)
    ].copy()
    filtered["object_name"] = filtered["object_name"].fillna(
        filtered["transport"] + "_" + filtered["object_id"].astype(str)
    )
    filtered["log_volume"] = np.log1p(filtered["daily_volume"])

    profile_cols = [f"wd_h{hour}" for hour in range(24)]
    feature_cols = profile_cols + [
        "morn_eve_ratio",
        "we_wd_ratio",
        "peakiness",
        "night_share",
        "midday_share",
        "log_volume",
    ]
    x = StandardScaler().fit_transform(filtered[feature_cols].values)

    rows = []
    upper_k = min(14, len(filtered) - 1)
    for k in range(4, upper_k + 1):
        labels = KMeans(k, n_init=30, random_state=42).fit_predict(x)
        sizes = pd.Series(labels).value_counts()
        rows.append(
            {
                "k": k,
                "sil": silhouette_score(x, labels),
                "min_cl": sizes.min(),
                "max_share": sizes.max() / len(labels),
            }
        )
    scores = pd.DataFrame(rows)
    good = scores[(scores["max_share"] < 0.40) & (scores["min_cl"] >= 10)]
    if len(good) == 0:
        good = scores[scores["max_share"] < 0.50]
    if len(good) == 0:
        good = scores
    best_k = int(good.loc[good["sil"].idxmax(), "k"])

    model = KMeans(best_k, n_init=50, random_state=42)
    filtered["cluster"] = model.fit_predict(x)

    sizes = filtered["cluster"].value_counts()
    small_clusters = sizes[sizes < 10].index.tolist()
    if small_clusters:
        centers = model.cluster_centers_
        for small_cluster in small_clusters:
            distances = np.linalg.norm(centers - centers[small_cluster], axis=1)
            distances[small_cluster] = np.inf
            for other_small in small_clusters:
                if other_small != small_cluster:
                    distances[other_small] = np.inf
            filtered.loc[filtered["cluster"] == small_cluster, "cluster"] = int(np.argmin(distances))
        mapping = {old: new for new, old in enumerate(sorted(filtered["cluster"].unique()))}
        filtered["cluster"] = filtered["cluster"].map(mapping)

    cluster_names = {}
    for cluster_id in sorted(filtered["cluster"].unique()):
        subset = filtered[filtered["cluster"] == cluster_id]
        ratio = subset["morn_eve_ratio"].mean()
        volume = subset["daily_volume"].mean()
        name = "Residential" if ratio > 1.3 else ("Business" if ratio < 0.75 else "Mixed")
        name += " large" if volume > 20000 else (" medium" if volume > 5000 else (" small" if volume > 1000 else " tiny"))
        dominant = subset["transport"].value_counts()
        if dominant.iloc[0] / len(subset) > 0.7:
            name += f" ({dominant.index[0]})"
        cluster_names[cluster_id] = name

    filtered["cluster_name"] = filtered["cluster"].map(cluster_names)
    filtered["object_id_str"] = filtered.apply(_object_id_str, axis=1)

    save_cols = [
        "object_id",
        "object_id_str",
        "object_name",
        "transport",
        "cluster",
        "cluster_name",
        "LN_CODE",
        "LN_NAME",
        "daily_volume",
        "log_volume",
        "morn_eve_ratio",
        "we_wd_ratio",
        "peakiness",
        "night_share",
        "midday_share",
    ]
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filtered[save_cols].to_csv(config.CLUSTERS_CSV, index=False)
    print(f"saved: {config.CLUSTERS_CSV} ({filtered['cluster'].nunique()} clusters)")
    return config.CLUSTERS_CSV

