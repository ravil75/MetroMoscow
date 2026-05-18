from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from . import config


def forecast_metrics(y_true, y_pred, model_name, horizon, train_mode, object_id=None, fold=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1.0)) * 100.0
    smape = np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100.0
    if np.sum(np.abs(y_true)) < 100:
        wape = np.nan
    else:    
        wape = np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), 1.0) * 100.0
    return {
        "model": model_name,
        "train_mode": train_mode,
        "horizon": horizon,
        "object_id": object_id,
        "fold": fold,
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4),
        "SMAPE": round(float(smape), 4),
        "WAPE": round(float(wape), 4),
        "y_mean": round(float(np.mean(y_true)), 4),
    }


def _as_array(series):
    return np.asarray(series, dtype=float)


def seasonal_naive(series, horizon, season=24):
    values = _as_array(series)
    if len(values) == 0:
        return np.zeros(horizon)
    if len(values) < season:
        return np.repeat(np.mean(values), horizon)
    return np.array([values[-season + (idx % season)] for idx in range(horizon)], dtype=float)


def mean_profile(series, horizon, season=24):
    values = _as_array(series)
    if len(values) < season:
        return np.repeat(np.mean(values) if len(values) else 0.0, horizon)
    days = len(values) // season
    profile = values[: days * season].reshape(days, season).mean(axis=0)
    return np.tile(profile, horizon // season + 1)[:horizon]


def weighted_profile(series, horizon, season=24):
    values = _as_array(series)
    if len(values) < season:
        return mean_profile(values, horizon, season)
    days = len(values) // season
    daily = values[: days * season].reshape(days, season)
    weights = np.exp(np.linspace(-1.0, 0.0, days))
    weights /= weights.sum()
    profile = np.average(daily, axis=0, weights=weights)
    return np.tile(profile, horizon // season + 1)[:horizon]


def same_type_day(series, target_index, horizon, season=24):
    if not isinstance(series.index, pd.DatetimeIndex):
        return mean_profile(series, horizon, season)
    values = _as_array(series)
    target_is_weekend = target_index[0].dayofweek >= 5
    profiles = []
    for date, group in series.groupby(series.index.normalize()):
        if len(group) != season:
            continue
        if (date.dayofweek >= 5) == target_is_weekend:
            profiles.append(group.values)
    if not profiles:
        return mean_profile(series, horizon, season)
    profile = np.mean(np.vstack(profiles), axis=0)
    return np.array([profile[ts.hour] for ts in target_index[:horizon]], dtype=float)


def holiday_aware_profile(series, target_index, horizon, season=24):
    if (target_index[0].month, target_index[0].day) not in config.HOLIDAYS_MD:
        return same_type_day(series, target_index, horizon, season)
    values = _as_array(series)
    profiles = []
    for date, group in series.groupby(series.index.normalize()):
        if len(group) == season and date.dayofweek == 6:
            profiles.append(group.values)
    if not profiles:
        return mean_profile(values, horizon, season)
    profile = np.mean(np.vstack(profiles), axis=0) * 0.85
    return np.array([profile[ts.hour] for ts in target_index[:horizon]], dtype=float)


class ImprovedETS:
    def __init__(self, seasonal_period=24):
        self.seasonal_period = seasonal_period

    def predict(self, series, target_index, horizon, synthetic_series=None):
        values = _as_array(series)
        sp = self.seasonal_period
        if len(values) < 2 * sp:
            return mean_profile(series, horizon, sp)

        candidates = {
            "mean": mean_profile(series, horizon, sp),
            "weighted": weighted_profile(series, horizon, sp),
            "seasonal": seasonal_naive(series, horizon, sp),
            "same_type": same_type_day(series, target_index, horizon, sp),
            "holiday": holiday_aware_profile(series, target_index, horizon, sp),
        }
        for seasonal in ["add", "mul"]:
            try:
                fit_values = np.maximum(values, 1.0) if seasonal == "mul" else values
                model = ExponentialSmoothing(
                    fit_values,
                    trend=None,
                    seasonal=seasonal,
                    seasonal_periods=sp,
                    initialization_method="estimated",
                ).fit(optimized=True)
                candidates[f"ets_{seasonal}"] = model.forecast(horizon)
            except Exception:
                continue

        validation_true = values[-sp:]
        validation_train = series.iloc[:-sp]
        best_name, best_mae = "weighted", np.inf
        for name in candidates:
            try:
                if name == "mean":
                    pred = mean_profile(validation_train, sp, sp)
                elif name == "weighted":
                    pred = weighted_profile(validation_train, sp, sp)
                elif name == "seasonal":
                    pred = seasonal_naive(validation_train, sp, sp)
                elif name == "same_type":
                    pred = same_type_day(validation_train, series.index[-sp:], sp, sp)
                elif name == "holiday":
                    pred = holiday_aware_profile(validation_train, series.index[-sp:], sp, sp)
                elif name.startswith("ets_"):
                    seasonal = "mul" if name.endswith("mul") else "add"
                    train_values = _as_array(validation_train)
                    fit_values = np.maximum(train_values, 1.0) if seasonal == "mul" else train_values
                    pred = ExponentialSmoothing(
                        fit_values,
                        trend=None,
                        seasonal=seasonal,
                        seasonal_periods=sp,
                        initialization_method="estimated",
                    ).fit(optimized=True).forecast(sp)
                else:
                    continue
                mae = np.mean(np.abs(validation_true - pred[: len(validation_true)]))
                if mae < best_mae:
                    best_name, best_mae = name, mae
            except Exception:
                continue
        return np.maximum(candidates.get(best_name, candidates["weighted"]), 0.0)


class CleanEnsemble:
    def predict(self, series, target_index, horizon, synthetic_series=None):
        sp = 24
        if len(series) < 2 * sp:
            return mean_profile(series, horizon, sp)

        train_sub = series.iloc[:-sp]
        validation = series.iloc[-sp:]
        validation_index = validation.index
        validation_preds = {
            "seasonal": seasonal_naive(train_sub, sp, sp),
            "same_type": same_type_day(train_sub, validation_index, sp, sp),
            "holiday": holiday_aware_profile(train_sub, validation_index, sp, sp),
            "weighted": weighted_profile(train_sub, sp, sp),
            "mean": mean_profile(train_sub, sp, sp),
        }
        scores = {
            name: mean_absolute_error(validation.values, pred[: len(validation)]) + 1e-8
            for name, pred in validation_preds.items()
        }
        inv_total = sum(1.0 / score for score in scores.values())
        weights = {name: (1.0 / score) / inv_total for name, score in scores.items()}

        test_preds = {
            "seasonal": seasonal_naive(series, horizon, sp),
            "same_type": same_type_day(series, target_index, horizon, sp),
            "holiday": holiday_aware_profile(series, target_index, horizon, sp),
            "weighted": weighted_profile(series, horizon, sp),
            "mean": mean_profile(series, horizon, sp),
        }
        return np.maximum(sum(weights[name] * test_preds[name] for name in weights), 0.0)


def _time_features(timestamp):
    return [
        np.sin(2 * np.pi * timestamp.hour / 24),
        np.cos(2 * np.pi * timestamp.hour / 24),
        np.sin(2 * np.pi * timestamp.dayofweek / 7),
        np.cos(2 * np.pi * timestamp.dayofweek / 7),
        float(timestamp.dayofweek >= 5),
        float((timestamp.month, timestamp.day) in config.HOLIDAYS_MD),
    ]


def _lag_features(values, idx, timestamp):
    history = values[:idx]
    fallback = float(np.mean(history)) if len(history) else 0.0

    def lag(offset):
        pos = idx - offset
        return float(values[pos]) if pos >= 0 else fallback

    recent = history[-24:] if len(history) else np.array([fallback])
    short = history[-6:] if len(history) else np.array([fallback])
    return [
        lag(1),
        lag(2),
        lag(24),
        lag(48),
        float(np.mean(short)),
        float(np.mean(recent)),
        float(np.std(recent)),
        * _time_features(timestamp),
    ]


def build_supervised_xy(series_list):
    x_rows, y_rows = [], []
    for series in series_list:
        series = series.sort_index()
        values = _as_array(series)
        if len(values) < 25:
            continue
        for idx in range(24, len(values)):
            x_rows.append(_lag_features(values, idx, series.index[idx]))
            y_rows.append(values[idx])
    if not x_rows:
        return np.empty((0, 13)), np.empty((0,))
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float)


@dataclass
class RecursiveRegressor:
    name: str
    kind: str = "knn"

    def _make_model(self):
        if self.kind == "knn":
            return KNeighborsRegressor(n_neighbors=5, weights="distance")
        raise ValueError(f"Unknown regressor kind: {self.kind}")

    def predict(self, series, target_index, horizon, synthetic_series=None):
        train_series = [series]
        if synthetic_series is not None and len(synthetic_series) >= 25:
            train_series.append(synthetic_series)
        x_train, y_train = build_supervised_xy(train_series)
        if len(y_train) < 5 or np.all(y_train == y_train[0]):
            return weighted_profile(series, horizon)

        model = self._make_model()
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_train)
        if self.kind == "knn":
            model.n_neighbors = min(model.n_neighbors, len(y_train))
        model.fit(x_scaled, y_train)

        history_values = list(_as_array(series))
        predictions = []
        for step in range(horizon):
            timestamp = target_index[step]
            values = np.asarray(history_values, dtype=float)
            features = np.asarray(_lag_features(values, len(values), timestamp), dtype=float).reshape(1, -1)
            pred = max(0.0, float(model.predict(scaler.transform(features))[0]))
            predictions.append(pred)
            history_values.append(pred)
        return np.asarray(predictions, dtype=float)


def make_model_registry():
    return {
        "Seasonal Naive": lambda s, idx, h, syn=None: seasonal_naive(s, h),
        "Mean Profile": lambda s, idx, h, syn=None: mean_profile(s, h),
        "Weighted Profile": lambda s, idx, h, syn=None: weighted_profile(s, h),
        "Same-Type Day": lambda s, idx, h, syn=None: same_type_day(s, idx, h),
        "Holiday Profile": lambda s, idx, h, syn=None: holiday_aware_profile(s, idx, h),
        "ETS": ImprovedETS().predict,
        "Clean Ensemble": CleanEnsemble().predict,
        "kNN Lag": RecursiveRegressor("kNN Lag", kind="knn").predict,
    }
