import pandas as pd
import numpy as np
import os
import json
from scipy.interpolate import CubicSpline
from statsmodels.tsa.stattools import acf
from src import config

def clean_id(v):
    s = str(v)
    return s[:-2] if s.endswith('.0') else s

def generate_global_warp(n_hours=24, n_knots=4, sigma=0.06):
    x = np.linspace(0, n_hours - 1, n_knots)
    y = np.random.normal(1.0, sigma, size=n_knots)
    return CubicSpline(x, y)(np.arange(n_hours))

def generate_cluster_warp(n_hours=24, n_knots=4, sigma=0.04):
    x = np.linspace(0, n_hours - 1, n_knots)
    y = np.random.normal(0.0, sigma, size=n_knots)
    return CubicSpline(x, y)(np.arange(n_hours))

def build_day_blocks(pivot_df):
    """Разбивает реальную матрицу на блоки по дням недели."""
    real_matrix = pivot_df.values
    n_hours, _ = real_matrix.shape
    n_days_real = n_hours // 24
    day_blocks = {i: [] for i in range(7)}
    
    for d in range(n_days_real):
        start_idx = d * 24
        end_idx = start_idx + 24
        if end_idx > n_hours: break
        day_data = real_matrix[start_idx:end_idx, :]
        dow = pivot_df.index[start_idx].dayofweek
        day_blocks[dow].append(day_data)
    return day_blocks

def load_clusters_mapping():
    """Загружает маппинг object_id -> cluster."""
    if not os.path.exists(config.CLUSTERS_CSV):
        print("ВНИМАНИЕ: final_clusters.csv не найден. Локальный варпинг отключен.")
        return {}
    
    clusters_df = pd.read_csv(config.CLUSTERS_CSV)
    
    def format_oid(row):
        oid = clean_id(row['object_id'])
        if row['transport'] == 'НГПТ':
            return f"RT_{oid}"
        else: # Метро, МЦД, Другое
            return f"ST_{oid}"
            
    clusters_df['object_id_str'] = clusters_df.apply(format_oid, axis=1)
    return dict(zip(clusters_df['object_id_str'], clusters_df['cluster']))

def run_synthesis(pivot_df, gen_days=60, seed=42):
    """Основной пайплайн генерации синтетических данных."""
    np.random.seed(seed)
    
    obj_to_cluster = load_clusters_mapping()
    top_objects = pivot_df.columns
    n_objects = len(top_objects)
    
    missing_in_clusters = set(top_objects) - set(obj_to_cluster.keys())
    if missing_in_clusters:
        print(f"!!!{len(missing_in_clusters)} объектов из топ-1500 не найдены в clusters.csv.")
    
    day_blocks = build_day_blocks(pivot_df)
    last_real_date = pivot_df.index[-1]
    unique_clusters = set(obj_to_cluster.get(oid, -1) for oid in top_objects)
    
    # Тренды
    global_trend_direction = np.random.uniform(-0.04, 0.04)
    cluster_trends = {c: np.random.uniform(-0.03, 0.03) for c in unique_clusters}
    object_trend_noise = np.random.normal(0, 0.01, size=n_objects)
    
    print(f"Начало генерации с {last_real_date + pd.Timedelta(hours=1)}...")
    print(f"Глобальный тренд: {'рост' if global_trend_direction > 0 else 'спад'} {abs(global_trend_direction)*100:.1f}%")
    
    synth_days = []
    
    for d in range(gen_days):
        target_date = last_real_date + pd.Timedelta(days=d+1)
        target_dow = target_date.dayofweek
        is_holiday = (target_date.month, target_date.day) in config.HOLIDAYS_MD
        
        # ШАГ 1: Foundation
        if is_holiday:
            available_blocks = day_blocks.get(6, [])
            if not available_blocks:
                all_available = [b for blocks in day_blocks.values() for b in blocks]
                chosen_day = all_available[np.random.randint(len(all_available))]
            else:
                chosen_day = available_blocks[np.random.randint(len(available_blocks))]
            chosen_day = chosen_day * np.random.uniform(0.75, 0.90)
        else:
            available_blocks = day_blocks.get(target_dow, [])
            if not available_blocks:
                all_available = [b for blocks in day_blocks.values() for b in blocks]
                chosen_day = all_available[np.random.randint(len(all_available))]
            else:
                chosen_day = available_blocks[np.random.randint(len(available_blocks))]
        
        # ШАГ 2: Global Morphing
        global_warp = generate_global_warp(sigma=0.06)
        warped_day = chosen_day * global_warp[:, np.newaxis] 
        
        # ШАГ 3: Cluster Morphing
        cluster_warps = {c: generate_cluster_warp(sigma=0.04) for c in unique_clusters}
        for i, oid in enumerate(top_objects):
            c = obj_to_cluster.get(oid, -1)
            warped_day[:, i] *= (1 + cluster_warps[c])
        
        warped_day = np.maximum(warped_day, 0)
        
        # ШАГ 4: Anomaly Spikes
        if np.random.random() < 0.10: 
            spike_hour = np.random.randint(7, 22)
            spike_magnitude = np.random.uniform(1.4, 1.8)
            spike_cluster = np.random.choice(list(unique_clusters))
            for i, oid in enumerate(top_objects):
                if obj_to_cluster.get(oid) == spike_cluster:
                    base = warped_day[spike_hour, i]
                    warped_day[spike_hour, i] = base * spike_magnitude + max(50, 2 * np.sqrt(base))
                    
        # ШАГ 5: Micro-Trend
        for i, oid in enumerate(top_objects):
            c = obj_to_cluster.get(oid, -1)
            local_trend = global_trend_direction + cluster_trends.get(c, 0) + object_trend_noise[i]
            warped_day[:, i] *= (1.0 + local_trend * (d / gen_days))
        
        # ШАГ 6: Physics
        final_day = np.random.poisson(np.maximum(warped_day, 0))
        synth_days.append(final_day)

    synth_matrix = np.concatenate(synth_days, axis=0).astype(np.float32)
    
    # Склейка
    dates_real = pivot_df.index
    dates_synth = pd.date_range(start=last_real_date + pd.Timedelta(hours=1), periods=gen_days * 24, freq='h')
    
    df_synth = pd.DataFrame(synth_matrix, columns=top_objects, index=dates_synth)
    df_real = pivot_df.astype(np.float32)
    df_full = pd.concat([df_real, df_synth])
    
    # Ковариаты
    df_full['hour'] = df_full.index.hour
    df_full['dow'] = df_full.index.dayofweek
    df_full['is_holiday'] = df_full.index.map(lambda x: int((x.month, x.day) in config.HOLIDAYS_MD))
    
    print(f"Сгенерировано: {synth_matrix.shape[0]} часов ({synth_matrix.shape[0]/24:.0f} дней)")
    return df_full

def final_validation(df_full, pivot_df):
    """Проверяет качество сгенерированных данных."""
    metrics = {}
    synth_start_idx = len(pivot_df)
    synth_data = df_full[pivot_df.columns].iloc[synth_start_idx:synth_start_idx+168]
    
    real_acf = acf(pivot_df.mean(axis=1).values, nlags=48, fft=True)[1:25]
    synth_acf = acf(synth_data.mean(axis=1).values, nlags=48, fft=True)[1:25]
    metrics['acf_similarity'] = 1 - np.mean(np.abs(real_acf - synth_acf))
    
    real_profile = pivot_df.groupby(pivot_df.index.hour).mean().mean(axis=1)
    synth_profile = synth_data.groupby(synth_data.index.hour).mean().mean(axis=1)
    metrics['profile_corr'] = real_profile.corr(synth_profile)
    
    real_q = np.quantile(pivot_df.values.ravel(), [0.01, 0.5, 0.99])
    synth_q = np.quantile(synth_data.values.ravel(), [0.01, 0.5, 0.99])
    metrics['quantile_ratio'] = np.mean(synth_q / (real_q + 1e-6))
    
    corr_synth = np.corrcoef(synth_data.T)
    mask = ~np.eye(corr_synth.shape[0], dtype=bool)
    metrics['corr_std'] = corr_synth[mask].std()
    
    metrics['ready_for_training'] = (
        metrics['acf_similarity'] > 0.85 and
        metrics['profile_corr'] > 0.9 and
        0.7 < metrics['quantile_ratio'] < 1.4 and
        metrics['corr_std'] > 0.01
    )
    return metrics

def save_generation_config(seed, gen_days, validation_passed):
    """Сохраняет конфигурацию запуска."""
    cfg = {
        'SEED': seed,
        'GEN_DAYS': gen_days,
        'WARP_SIGMAS': {'global': 0.06, 'cluster': 0.04},
        'ANOMALY_PROB': 0.10,
        'ANOMALY_MAGNITUDE': '1.4-1.8 + additive_buffer',
        'TREND_GLOBAL_RANGE': '-0.04 to 0.04',
        'TREND_CLUSTER_RANGE': '-0.03 to 0.03',
        'TREND_OBJECT_NOISE_STD': 0.01,
        'POISSON_NOISE': True,
        'validation_passed': validation_passed
    }
    with open(config.GENERATION_CONFIG, 'w') as f:
        json.dump(cfg, f, indent=2, default=str)
    print(f"Конфигурация сохранена в {config.GENERATION_CONFIG}")
