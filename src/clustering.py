import pandas as pd, numpy as np, os, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .config import *

warnings.filterwarnings('ignore')

def clean_id(v):
    s = str(v)
    return s[:-2] if s.endswith('.0') else s

def build_profiles(subset, id_col, name_col):
    subset = subset.copy()
    subset['is_wknd'] = (subset['dow'] >= 5).astype(int)
    rows = []
    for oid in subset[id_col].dropna().unique():
        d = subset[subset[id_col] == oid]
        name = d[name_col].iloc[0] if name_col in d.columns else '?'
        wd = d[d['is_wknd']==0]
        nd_wd = wd['date'].nunique()
        if nd_wd == 0: continue
        wd_h = wd.groupby('hour')['pax'].sum() / nd_wd
        wd_tot = wd_h.sum()
        if wd_tot == 0: continue

        we = d[d['is_wknd']==1]
        nd_we = we['date'].nunique()
        we_h = we.groupby('hour')['pax'].sum() / nd_we if nd_we > 0 else pd.Series(dtype=float)
        we_tot = we_h.sum() if nd_we > 0 else 0

        r = {'object_id': oid, 'object_name': name, 'daily_volume': wd_tot}
        for h in range(24):
            r[f'wd_h{h}'] = wd_h.get(h,0) / wd_tot
            r[f'we_h{h}'] = (we_h.get(h,0) / we_tot) if we_tot > 0 else 0

        morn = sum(wd_h.get(h,0) for h in [7,8,9])
        eve  = sum(wd_h.get(h,0) for h in [17,18,19])
        r['morn_eve_ratio'] = morn/eve if eve > 0 else 1.0
        r['we_wd_ratio'] = we_tot/wd_tot if wd_tot > 0 else 0
        r['peakiness'] = wd_h.max() / (wd_tot/24)
        r['night_share'] = sum(wd_h.get(h,0) for h in [23,0,1,2,3,4,5]) / wd_tot
        r['midday_share'] = sum(wd_h.get(h,0) for h in [11,12,13,14]) / wd_tot
        rows.append(r)
    return pd.DataFrame(rows)

def run_clustering(force_regenerate=False):
    # Проверяем, нужно ли пропускать
    if not force_regenerate and os.path.exists(CLUSTERS_CSV):
        print("final_clusters.csv уже существует. Пропуск.")
        return

    # Убеждаемся, что папка существует
    os.makedirs(os.path.dirname(CLUSTERS_CSV), exist_ok=True)

    hourly = pd.read_parquet(HOURLY_PARQUET)
    
    metro = hourly[hourly['tcat']=='Метро']
    metro_f = build_profiles(metro, 'ST_CODE', 'ST_NAME')
    metro_f['transport'] = 'Метро'
    st_map = metro.drop_duplicates('ST_CODE').set_index('ST_CODE')
    metro_f['LN_CODE'] = metro_f['object_id'].map(st_map['LN_CODE'])
    metro_f['LN_NAME'] = metro_f['object_id'].map(st_map['LN_NAME'])

    ngpt = hourly[hourly['tcat']=='НГПТ']
    ngpt_f = build_profiles(ngpt, 'BUS_RT_NO', 'ROUTE_NAME')
    ngpt_f['transport'] = 'НГПТ'
    ngpt_f['LN_CODE'] = ngpt_f['LN_NAME'] = None

    parts = [metro_f, ngpt_f]
    mcd = hourly[hourly['tcat']=='МЦД']
    if mcd['ST_CODE'].nunique() > 2:
        mcd_f = build_profiles(mcd, 'ST_CODE', 'ST_NAME')
        mcd_f['transport'] = 'МЦД'
        sm = mcd.drop_duplicates('ST_CODE').set_index('ST_CODE')
        mcd_f['LN_CODE'] = mcd_f['object_id'].map(sm['LN_CODE'])
        mcd_f['LN_NAME'] = mcd_f['object_id'].map(sm['LN_NAME'])
        parts.append(mcd_f)

    all_feat = pd.concat(parts, ignore_index=True)
    af = all_feat[(all_feat['daily_volume'] >= 50) & (all_feat['peakiness'] < 15) & (all_feat['we_wd_ratio'] > 0)].copy()
    af['object_name'] = af['object_name'].fillna(af['transport']+'_'+af['object_id'].astype(str))
    af['log_volume'] = np.log1p(af['daily_volume'])

    profile_cols = [f'wd_h{h}' for h in range(24)]
    feat_cols = profile_cols + ['morn_eve_ratio','we_wd_ratio','peakiness','night_share','midday_share','log_volume']
    X = StandardScaler().fit_transform(af[feat_cols].values)

    res = []
    for k in range(4, 15):
        lab = KMeans(k, n_init=30, random_state=42).fit_predict(X)
        sizes = pd.Series(lab).value_counts()
        res.append({'k': k, 'sil': silhouette_score(X, lab), 'min_cl': sizes.min(), 'max_share': sizes.max()/len(lab)})
    rdf = pd.DataFrame(res)

    good = rdf[(rdf.max_share < 0.40) & (rdf.min_cl >= 10)]
    if len(good) == 0: good = rdf[rdf.max_share < 0.50]
    best_k = int(good.loc[good.sil.idxmax(), 'k'])
    
    km = KMeans(best_k, n_init=50, random_state=42)
    af['cluster'] = km.fit_predict(X)

    sizes = af['cluster'].value_counts()
    small = sizes[sizes < 10].index.tolist()
    if small:
        centers = km.cluster_centers_
        for sc in small:
            dists = np.linalg.norm(centers - centers[sc], axis=1)
            dists[sc] = np.inf
            for s2 in small:
                if s2 != sc: dists[s2] = np.inf
            af.loc[af.cluster == sc, 'cluster'] = np.argmin(dists)
        mapping = {old: new for new, old in enumerate(sorted(af.cluster.unique()))}
        af['cluster'] = af.cluster.map(mapping)

    cluster_names = {}
    for c in sorted(af.cluster.unique()):
        s = af[af.cluster==c]
        me, vol = s.morn_eve_ratio.mean(), s.daily_volume.mean()
        nm = "Жилой" if me>1.3 else ("Деловой" if me<0.75 else "Смешанный")
        nm += " крупный" if vol>20000 else (" средний" if vol>5000 else (" малый" if vol>1000 else " мелкий"))
        dom = s.transport.value_counts()
        if dom.iloc[0]/len(s) > 0.7: nm += f" ({dom.index[0]})"
        cluster_names[c] = nm

    af['cluster_name'] = af.cluster.map(cluster_names)
    
    # НОВОЕ:  строковый ID с префиксом (ST_ / RT_)
    def format_oid(row):
        oid = clean_id(row['object_id'])
        if row['transport'] == 'НГПТ': return f"RT_{oid}"
        else: return f"ST_{oid}"
        
    af['object_id_str'] = af.apply(format_oid, axis=1)

    # НОВОЕ: 'object_id_str' в список сохранения
    save_cols = ['object_id', 'object_id_str', 'object_name', 'transport', 'cluster', 'cluster_name',
                 'LN_CODE', 'LN_NAME', 'daily_volume', 'log_volume',
                 'morn_eve_ratio', 'we_wd_ratio', 'peakiness', 'night_share', 'midday_share']
    
    af[save_cols].to_csv(CLUSTERS_CSV, index=False)
    print(f"✅ Сохранено: {CLUSTERS_CSV} (кластеров: {af.cluster.nunique()})")
