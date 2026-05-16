from pathlib import Path
import os

ROOT_DIR = Path(os.getcwd())
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "eda_output"

RAW_ZIP = ROOT_DIR / "data.zip"
METRO_ZIP = DATA_DIR / "Метро НБС" / "pass_10-160324.zip"
PASS_CSV = DATA_DIR / "PASS_ALL_202503242210.csv"

HOURLY_PARQUET = OUTPUT_DIR / "hourly.parquet"
OBJECT_HOURLY_PARQUET = OUTPUT_DIR / "object_hourly.parquet"
CLUSTERS_CSV = OUTPUT_DIR / "final_clusters.csv"
GENERATION_CONFIG = OUTPUT_DIR / "generation_config.json"

RESULTS_TEMPLATE = "results_{horizon}h.csv"
SUMMARY_TEMPLATE = "summary_{horizon}h.csv"
SYNTH_VALIDATION_TEMPLATE = "synth_validation_{horizon}h.csv"

HOLIDAYS_MD = {
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
    (2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4),
}

DEFAULT_TOP_N = 1500
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

VALIDATION_MODE_IN = 1
VALIDATION_RESULT_OK = 1
IS_FAIL_OK = 0
