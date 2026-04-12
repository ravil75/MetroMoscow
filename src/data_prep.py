import pandas as pd, numpy as np, os, gc, zipfile, re, warnings
import gdown
from src import config

warnings.filterwarnings('ignore')

def download_and_extract():
    if not os.path.exists(config.RAW_ZIP):
        gdown.download('https://drive.google.com/uc?id=1resyBioT2TJTZdLVfuEZdk8PvJAbK5GM', config.RAW_ZIP, quiet=False)
    if not os.path.exists(config.PASS_CSV):
        with zipfile.ZipFile(config.RAW_ZIP, 'r') as z: z.extractall(config.DATA_DIR)
        with zipfile.ZipFile(config.METRO_ZIP, 'r') as z: z.extractall(config.DATA_DIR)

def load_references():
    ref_places = pd.read_csv(os.path.join(config.DATA_DIR, 'REF_PSG_PLACES_202503251822.csv'), sep=';')
    ref_transport = pd.read_csv(os.path.join(config.DATA_DIR, 'REF_TRANSPORT_TYPE_202503251727.csv'), sep=';')
    ref_routes = pd.read_csv(os.path.join(config.DATA_DIR, 'REF_TRANSPORT_WAY_202503251803.csv'), sep=';')
    gds_goods = pd.read_csv(os.path.join(config.DATA_DIR, 'GDS_GOODS_202503251844.csv'), sep=';')
    
    rows = []
    pcr_path = os.path.join(config.DATA_DIR, 'V_PCR_CONTRACTOR_202503251702.csv')
    with open(pcr_path, 'r', encoding='utf-8') as f:
        for line in f:
            for m in re.findall(r'(\d+);;([^0-9]+?)(?=\d+;;|$)', line):
                rows.append({'ID': int(m[0]), 'PARENT_ID': None, 'NAME_SHORT': m[1].strip()})
    pcr_contr = pd.DataFrame(rows)
    return ref_places, ref_transport, ref_routes, gds_goods, pcr_contr

def create_hourly_parquet():
    if os.path.exists(config.HOURLY_PARQUET):
        print("hourly.parquet уже существует. Пропуск агрегации.")
        return
    
    download_and_extract()
    ref_places, ref_transport, ref_routes, gds_goods, pcr_contr = load_references()
    
    USE_COLS = ['TRAN_DATE','PLACE_ID','TRANSPORT_TYPE_ID','BUS_RT_NO',
                'AGENT_ID','GD_ID','TRANSFER_TYPE_ID','VALIDATION_MODE',
                'CPPC_VALIDATION_TYPE','IS_FAIL','CRD_NO']
    GROUP_KEYS = ['date_hour','PLACE_ID','TRANSPORT_TYPE_ID','BUS_RT_NO',
                  'AGENT_ID','GD_ID','TRANSFER_TYPE_ID','VALIDATION_MODE',
                  'CPPC_VALIDATION_TYPE']

    CHUNK = 3_000_000
    agg_parts, n_total = [], 0
    reader = pd.read_csv(config.PASS_CSV, sep=';', usecols=USE_COLS, parse_dates=['TRAN_DATE'], chunksize=CHUNK, low_memory=False)

    for i, ch in enumerate(reader):
        n_total += len(ch)
        ch = ch[ch['IS_FAIL'] != 1].copy()
        ch['date_hour'] = ch['TRAN_DATE'].dt.floor('h')
        agg = ch.groupby(GROUP_KEYS, dropna=False).agg(pax=('TRAN_DATE', 'size'), unique_cards=('CRD_NO', 'nunique')).reset_index()
        agg_parts.append(agg)
        del ch, agg; gc.collect()
        if (i+1) % 10 == 0: print(f"  chunk {i+1}: {n_total:,} строк")

    hourly = pd.concat(agg_parts, ignore_index=True); del agg_parts; gc.collect()
    hourly = hourly.groupby(GROUP_KEYS, dropna=False).agg(pax=('pax','sum'), unique_cards=('unique_cards','sum')).reset_index()

    # Обогащение
    hourly = hourly.merge(ref_places[['PLACE_ID','TYPE_ID','ST_CODE','ST_NAME','LN_CODE','LN_NAME','IS_TEST']], on='PLACE_ID', how='left')
    hourly = hourly[hourly['IS_TEST'] != 1].drop(columns='IS_TEST')
    hourly = hourly.merge(ref_transport.rename(columns={'TRANSPORT_ID':'TRANSPORT_TYPE_ID','NAME':'TRANSPORT_NAME'}), on='TRANSPORT_TYPE_ID', how='left')
    hourly = hourly.merge(ref_routes[['WAY_ID','NAME']].rename(columns={'WAY_ID':'BUS_RT_NO','NAME':'ROUTE_NAME'}), on='BUS_RT_NO', how='left')
    hourly = hourly.merge(pcr_contr[['ID','NAME_SHORT']].rename(columns={'ID':'AGENT_ID','NAME_SHORT':'AGENT_NAME'}), on='AGENT_ID', how='left')
    hourly = hourly.merge(gds_goods[['GD_ID','NAME_SHORT','ARCHITECT_ID']].rename(columns={'NAME_SHORT':'TICKET_NAME'}), on='GD_ID', how='left')

    def transport_cat(row):
        if pd.notna(row['BUS_RT_NO']): return 'НГПТ'
        if row['TYPE_ID'] == 1: return 'Метро'
        if row['TYPE_ID'] == 15: return 'МЦД'
        return 'Другое'

    hourly['tcat'] = hourly.apply(transport_cat, axis=1)
    hourly['hour'] = hourly['date_hour'].dt.hour
    hourly['date'] = hourly['date_hour'].dt.normalize()
    hourly['dow'] = hourly['date_hour'].dt.dayofweek
    hourly['month'] = hourly['date_hour'].dt.month
    hourly['is_wknd'] = (hourly['dow'] >= 5).astype(int)
    
    hourly['md'] = list(zip(hourly['date_hour'].dt.month, hourly['date_hour'].dt.day))
    hourly['is_hol'] = hourly['md'].isin(config.HOLIDAYS_MD).astype(int)
    hourly.drop('md', axis=1, inplace=True)

    # Создаем папку, если ее нет
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    hourly.to_parquet(config.HOURLY_PARQUET, index=False)
    print(f"Сохранено: {config.HOURLY_PARQUET} ({len(hourly):,} строк)")
