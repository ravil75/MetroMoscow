import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from datetime import timedelta
from .config import HOLIDAYS_MD

def metrics(yt, yp, name=""):
    yt, yp = np.array(yt, float), np.array(yp, float)
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    smape = np.mean(2*np.abs(yt-yp) / (np.abs(yt)+np.abs(yp)+1e-8)) * 100
    mape = np.mean(np.abs(yt-yp) / np.maximum(np.abs(yt), 1)) * 100
    return {'model':name, 'MAE':round(mae,2), 'RMSE':round(rmse,2), 'SMAPE':round(smape,2), 'MAPE':round(mape,2)}

def check_holiday(dt):
    return int((dt.month, dt.day) in HOLIDAYS_MD)

def seasonal_naive(y, h, s=24):
    return np.array([y[-(s) + i%s] for i in range(h)])

def mean_profile(y, h, s=24):
    days = len(y)//s
    return np.tile(y[:days*s].reshape(days,s).mean(0), h//s+1)[:h]

def weighted_profile(y, h, s=24):
    days = len(y)//s
    d = y[:days*s].reshape(days,s)
    w = np.exp(np.linspace(-1,0,days)); w /= w.sum()
    return np.tile(np.average(d, axis=0, weights=w), h//s+1)[:h]

def same_type_day(y, h, test_dow, s=24):
    days = len(y)//s
    d = y[:days*s].reshape(days,s)
    is_we = test_dow >= 5
    idx = [i for i in range(days) if (i%7 >= 5) == is_we]
    prof = d[idx].mean(0) if idx else d.mean(0)
    return np.tile(prof, h//s+1)[:h]

def holiday_aware_naive(y, h, test_dow, is_hol, s=24):
    eff_dow = 6 if is_hol else test_dow
    pred = same_type_day(y, h, eff_dow, s)
    return np.maximum(pred, 0)

class ImprovedETS:
    def __init__(self, sp=24): self.sp = sp
    def fit_predict(self, y_train, steps, test_dow=None, is_hol=0):
        y = np.array(y_train, float); sp = self.sp
        if len(y) < 2*sp: return mean_profile(y, steps, sp)
        cands = {'mean': mean_profile(y, steps, sp), 'weighted': weighted_profile(y, steps, sp), 'seasonal': seasonal_naive(y, steps, sp)}
        if test_dow is not None:
            cands['same_type'] = same_type_day(y, steps, test_dow, sp)
            cands['hol_aware'] = holiday_aware_naive(y, steps, test_dow, is_hol, sp)
        for stype in ['add', 'mul']:
            try:
                yf = np.maximum(y, 1) if stype=='mul' else y
                m = ExponentialSmoothing(yf, trend=None, seasonal=stype, seasonal_periods=sp, initialization_method='estimated').fit(optimized=True)
                cands[f'ets_{stype}'] = m.forecast(steps)
            except: pass
        y_val, y_sub = y[-sp:], y[:-sp]; val_dow = (len(y_sub)//sp) % 7
        best_name, best_mae = 'mean', np.inf
        for nm in cands:
            try:
                if nm == 'seasonal': vp = seasonal_naive(y_sub, sp, sp)
                elif nm == 'mean': vp = mean_profile(y_sub, sp, sp)
                elif nm == 'weighted': vp = weighted_profile(y_sub, sp, sp)
                elif nm == 'same_type': vp = same_type_day(y_sub, sp, val_dow, sp)
                elif nm == 'hol_aware': vp = holiday_aware_naive(y_sub, sp, val_dow, is_hol, sp)
                elif 'ets' in nm:
                    st = 'mul' if 'mul' in nm else 'add'
                    yf = np.maximum(y_sub, 1) if st=='mul' else y_sub
                    vp = ExponentialSmoothing(yf, trend=None, seasonal=st, seasonal_periods=sp, initialization_method='estimated').fit(optimized=True).forecast(sp)
                else: continue
                mae = np.mean(np.abs(y_val - vp[:len(y_val)]))
                if mae < best_mae: best_mae, best_name = mae, nm
            except: continue
        return np.maximum(cands.get(best_name, cands['mean']), 0)

class CleanEnsemble:
    def fit_predict(self, y_train, steps, test_dow, is_hol=0):
        sp = 24
        if len(y_train) < 2*sp: return mean_profile(y_train, steps, sp)
        y_sub, y_val = y_train[:-sp], y_train[-sp:]; vd = (len(y_sub)//sp) % 7
        models_val = {'seasonal': seasonal_naive(y_sub, sp, sp), 'same_type': same_type_day(y_sub, sp, vd, sp), 'hol_aware': holiday_aware_naive(y_sub, sp, vd, is_hol, sp), 'weighted': weighted_profile(y_sub, sp, sp)}
        scores = {n: mean_absolute_error(y_val, p[:len(y_val)])+1e-8 for n, p in models_val.items()}
        tot = sum(1/s for s in scores.values()); w = {n: (1/s)/tot for n, s in scores.items()}
        models_test = {'seasonal': seasonal_naive(y_train, steps, sp), 'same_type': same_type_day(y_train, steps, test_dow, sp), 'hol_aware': holiday_aware_naive(y_train, steps, test_dow, is_hol, sp), 'weighted': weighted_profile(y_train, steps, sp)}
        return np.maximum(sum(w[n]*models_test[n] for n in w), 0)

class MinimalKNN:
    def __init__(self, k=3): self.k = k; self.scaler = StandardScaler()
    def fit(self, ts):
        v = ts.values; self.mu = np.mean(v[v>0]) if (v>0).any() else 1
        X, y = [], []
        for i in range(24, len(v)):
            h = ts.index[i].hour
            X.append([np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24), v[i-24]/self.mu])
            y.append(v[i]/self.mu)
        X, y = np.array(X), np.array(y); self.scaler.fit(X); k = min(self.k, len(X)-1, 3)
        self.model = KNeighborsRegressor(max(k,1), weights='distance'); self.model.fit(self.scaler.transform(X), y)
        self.vals, self.idx = v, ts.index; return self
    def predict(self, steps):
        preds = []
        for i in range(steps):
            h = (self.idx[-1] + timedelta(hours=i+1)).hour; li = len(self.vals)-24+i
            lag = self.vals[li]/self.mu if 0<=li<len(self.vals) else 1
            x = np.array([[np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24), lag]])
            preds.append(max(0, self.model.predict(self.scaler.transform(x))[0]*self.mu))
        return np.array(preds)
