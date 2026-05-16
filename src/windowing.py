import numpy as np
import pandas as pd

from . import config


def make_time_covariates(index, is_synthetic=0):
    index = pd.DatetimeIndex(index)
    return pd.DataFrame(
        {
            "hour": index.hour.astype(float) / 23.0,
            "dow": index.dayofweek.astype(float) / 6.0,
            "hour_sin": np.sin(2 * np.pi * index.hour / 24),
            "hour_cos": np.cos(2 * np.pi * index.hour / 24),
            "dow_sin": np.sin(2 * np.pi * index.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * index.dayofweek / 7),
            "is_weekend": (index.dayofweek >= 5).astype(float),
            "is_holiday": [float((ts.month, ts.day) in config.HOLIDAYS_MD) for ts in index],
            "is_synthetic": float(is_synthetic),
        },
        index=index,
    )


def make_global_windows(pivot_df, past_window=48, horizon=24, step=1, is_synthetic=0):
    """Return arrays for global neural models.

    Shapes:
    - x: samples x past_window x nodes
    - y: samples x horizon x nodes
    - x_cov: samples x past_window x time_features
    - y_cov: samples x horizon x time_features
    """
    pivot_df = pivot_df.sort_index()
    values = pivot_df.values.astype(np.float32)
    covariates = make_time_covariates(pivot_df.index, is_synthetic=is_synthetic).values.astype(np.float32)

    x_rows, y_rows, x_cov_rows, y_cov_rows, starts = [], [], [], [], []
    max_start = len(pivot_df) - past_window - horizon
    for start in range(0, max_start + 1, step):
        x_slice = slice(start, start + past_window)
        y_slice = slice(start + past_window, start + past_window + horizon)
        x_rows.append(values[x_slice])
        y_rows.append(values[y_slice])
        x_cov_rows.append(covariates[x_slice])
        y_cov_rows.append(covariates[y_slice])
        starts.append(pivot_df.index[start + past_window])

    if not x_rows:
        empty_x = np.empty((0, past_window, values.shape[1]), dtype=np.float32)
        empty_y = np.empty((0, horizon, values.shape[1]), dtype=np.float32)
        empty_cov_x = np.empty((0, past_window, covariates.shape[1]), dtype=np.float32)
        empty_cov_y = np.empty((0, horizon, covariates.shape[1]), dtype=np.float32)
        return empty_x, empty_y, empty_cov_x, empty_cov_y, pd.DatetimeIndex([])

    return (
        np.stack(x_rows),
        np.stack(y_rows),
        np.stack(x_cov_rows),
        np.stack(y_cov_rows),
        pd.DatetimeIndex(starts),
    )


def make_correlation_adjacency(pivot_df, top_k=8, min_corr=0.30, include_self=True):
    """Build a sparse correlation graph for GAT-like models."""
    corr = pivot_df.corr().fillna(0.0)
    np.fill_diagonal(corr.values, 0.0)
    adjacency = pd.DataFrame(0.0, index=corr.index, columns=corr.columns)

    for node in corr.columns:
        neighbors = corr[node].sort_values(ascending=False)
        neighbors = neighbors[neighbors >= min_corr].head(top_k)
        adjacency.loc[node, neighbors.index] = neighbors.values

    adjacency = np.maximum(adjacency.values, adjacency.values.T)
    adjacency = pd.DataFrame(adjacency, index=corr.index, columns=corr.columns)
    if include_self:
        np.fill_diagonal(adjacency.values, 1.0)
    return adjacency
