import os

# === Пути ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'eda_output')

RAW_ZIP = os.path.join(ROOT_DIR, 'data.zip')
METRO_ZIP = os.path.join(DATA_DIR, 'Метро НБС/pass_10-160324.zip')
PASS_CSV = os.path.join(DATA_DIR, 'PASS_ALL_202503242210.csv')

HOURLY_PARQUET = os.path.join(OUTPUT_DIR, 'hourly.parquet')
CLUSTERS_CSV = os.path.join(OUTPUT_DIR, 'final_clusters.csv')

RESULTS_24H = os.path.join(OUTPUT_DIR, 'results_24h.csv')
RESULTS_1H = os.path.join(OUTPUT_DIR, 'results_1h.csv')
SUMMARY_24H = os.path.join(OUTPUT_DIR, 'summary_24h.csv')
SUMMARY_1H = os.path.join(OUTPUT_DIR, 'summary_1h.csv')

# Синтетика
AUGMENTED_PARQUET = os.path.join(OUTPUT_DIR, 'augmented_full.parquet')
GENERATION_CONFIG = os.path.join(OUTPUT_DIR, 'generation_config.json')
AUGMENTATION_PLOT = os.path.join(OUTPUT_DIR, 'hybrid_augmentation_v3.png')

# === Константы ===
HOLIDAYS_MD = [(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),
               (2,23),(3,8),(5,1),(5,9),(6,12),(11,4)]

HORIZON_24 = 24
HORIZON_1 = 1
N_HOURS = 168
DOW_NAMES = ['Пн','Вт','Ср','Чт','Пт','Сб','Вс']
