"""Иерархия станция → кластер → итог и MinT-согласование прогнозов.

MinT (Wickramasuriya, Athanasopoulos, Hyndman, JASA 2019): оптимальное
trace-minimizing согласование. Базовые прогнозы делаются НЕЗАВИСИМО на всех
уровнях (станции + кластеры + итог); MinT проецирует их в когерентное
подпространство, оптимально блендя прогноз станции с прогнозом агрегата
(агрегат точнее — шум усредняется), снижая ошибку шумных мелких станций.
"""
import numpy as np
import pandas as pd

from ... import config

AGG_PREFIX = "AGG_"


def load_cluster_map(object_ids, clusters_path=None):
    """object_id → cluster (int). Объекты без кластера → -1 (свой узел)."""
    path = clusters_path or config.CLUSTERS_CSV
    cmap = {}
    if path.exists():
        df = pd.read_csv(path)
        key = "object_id_str" if "object_id_str" in df.columns else "object_id"
        for oid, cl in zip(df[key].astype(str), df["cluster"]):
            cmap[oid] = int(cl) if pd.notna(cl) else -1
    return {str(o): cmap.get(str(o), -1) for o in object_ids}


def augment_with_aggregates(pivot, cluster_map):
    """Добавляет к pivot колонки кластер-сумм и итога (агрегатные ряды)."""
    station_ids = [c for c in pivot.columns if not str(c).startswith(AGG_PREFIX)]
    clusters = sorted(set(cluster_map[str(o)] for o in station_ids))
    extra = {}
    for c in clusters:
        members = [o for o in station_ids if cluster_map[str(o)] == c]
        extra[f"{AGG_PREFIX}CL{c}"] = pivot[members].sum(axis=1)
    extra[f"{AGG_PREFIX}TOTAL"] = pivot[station_ids].sum(axis=1)
    return pd.concat([pivot, pd.DataFrame(extra, index=pivot.index)], axis=1)


def build_summing_matrix(station_ids, cluster_map):
    """S: [n_nodes × m_stations]. Порядок узлов: станции, затем кластеры, затем итог."""
    clusters = sorted(set(cluster_map[str(o)] for o in station_ids))
    m = len(station_ids)
    idx = {str(o): i for i, o in enumerate(station_ids)}
    rows, node_ids = [], []
    for o in station_ids:                                   # нижний уровень = I
        r = np.zeros(m); r[idx[str(o)]] = 1.0
        rows.append(r); node_ids.append(str(o))
    for c in clusters:                                      # кластеры
        r = np.array([1.0 if cluster_map[str(o)] == c else 0.0 for o in station_ids])
        rows.append(r); node_ids.append(f"{AGG_PREFIX}CL{c}")
    rows.append(np.ones(m)); node_ids.append(f"{AGG_PREFIX}TOTAL")   # итог
    return np.vstack(rows), node_ids


def mint_projection(S, method="structural"):
    """G = (Sᵀ W⁻¹ S)⁻¹ Sᵀ W⁻¹ — проекция на нижний уровень.
    MinT(Structural): W⁻¹ = 1/(число листьев под узлом) — робастно, без оценки
    ковариации ошибок (важно на коротких данных). method='ols' → W=I."""
    n = S.shape[0]
    w_inv = np.ones(n) if method == "ols" else 1.0 / S.sum(axis=1)
    StWi = S.T * w_inv                                       # [m × n]
    return np.linalg.solve(StWi @ S, StWi)                   # G: [m × n]
