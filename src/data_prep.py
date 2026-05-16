import gc
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def clean_id(value):
    text = str(value)
    return text[:-2] if text.endswith(".0") else text


def object_id_from_row(row):
    if pd.notna(row.get("BUS_RT_NO")):
        return f"RT_{clean_id(row['BUS_RT_NO'])}"
    if pd.notna(row.get("ST_CODE")):
        return f"ST_{clean_id(row['ST_CODE'])}"
    if pd.notna(row.get("PLACE_ID")):
        return f"PL_{clean_id(row['PLACE_ID'])}"
    return None


def download_and_extract():
    import gdown

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not config.RAW_ZIP.exists():
        gdown.download(
            "https://drive.google.com/uc?id=1resyBioT2TJTZdLVfuEZdk8PvJAbK5GM",
            str(config.RAW_ZIP),
            quiet=False,
        )

    if not config.PASS_CSV.exists():
        with zipfile.ZipFile(config.RAW_ZIP, "r") as archive:
            archive.extractall(config.DATA_DIR)
        with zipfile.ZipFile(config.METRO_ZIP, "r") as archive:
            archive.extractall(config.DATA_DIR)


def load_references():
    ref_places = pd.read_csv(config.DATA_DIR / "REF_PSG_PLACES_202503251822.csv", sep=";")
    ref_transport = pd.read_csv(config.DATA_DIR / "REF_TRANSPORT_TYPE_202503251727.csv", sep=";")
    ref_routes = pd.read_csv(config.DATA_DIR / "REF_TRANSPORT_WAY_202503251803.csv", sep=";")
    gds_goods = pd.read_csv(config.DATA_DIR / "GDS_GOODS_202503251844.csv", sep=";")

    rows = []
    pcr_path = config.DATA_DIR / "V_PCR_CONTRACTOR_202503251702.csv"
    with pcr_path.open("r", encoding="utf-8") as file:
        for line in file:
            for contractor_id, name in re.findall(r"(\d+);;([^0-9]+?)(?=\d+;;|$)", line):
                rows.append({"ID": int(contractor_id), "PARENT_ID": None, "NAME_SHORT": name.strip()})
    pcr_contr = pd.DataFrame(rows)
    return ref_places, ref_transport, ref_routes, gds_goods, pcr_contr


def transport_category(row):
    if pd.notna(row["BUS_RT_NO"]):
        return "NGPT"
    if row["TYPE_ID"] == 1:
        return "Metro"
    if row["TYPE_ID"] == 15:
        return "MCD"
    return "Other"


def create_hourly_parquet(force=False, chunk_size=3_000_000):
    if config.HOURLY_PARQUET.exists() and not force:
        print(f"{config.HOURLY_PARQUET} already exists. Skipping aggregation.")
        return config.HOURLY_PARQUET

    download_and_extract()
    ref_places, ref_transport, ref_routes, gds_goods, pcr_contr = load_references()

    use_cols = [
        "TRAN_DATE",
        "PLACE_ID",
        "TRANSPORT_TYPE_ID",
        "BUS_RT_NO",
        "VALIDATION_MODE",
        "VALIDATION_RESULT",
        "IS_FAIL",
        "CRD_NO",
    ]
    group_keys = [
        "date_hour",
        "PLACE_ID",
        "TRANSPORT_TYPE_ID",
        "BUS_RT_NO",
    ]

    agg_parts = []
    n_total = 0
    reader = pd.read_csv(
        config.PASS_CSV,
        sep=";",
        usecols=use_cols,
        parse_dates=["TRAN_DATE"],
        chunksize=chunk_size,
        low_memory=False,
    )

    for idx, chunk in enumerate(reader, start=1):
        n_total += len(chunk)
        chunk = chunk[
            (chunk["VALIDATION_MODE"] == config.VALIDATION_MODE_IN)
            & (chunk["VALIDATION_RESULT"] == config.VALIDATION_RESULT_OK)
            & (chunk["IS_FAIL"] == config.IS_FAIL_OK)
        ].copy()
        chunk["date_hour"] = chunk["TRAN_DATE"].dt.floor("h")
        agg = (
            chunk.groupby(group_keys, dropna=False)
            .agg(pax=("TRAN_DATE", "size"), unique_cards=("CRD_NO", "nunique"))
            .reset_index()
        )
        agg_parts.append(agg)
        del chunk, agg
        gc.collect()
        if idx % 10 == 0:
            print(f"chunk {idx}: {n_total:,} rows")

    hourly = pd.concat(agg_parts, ignore_index=True)
    del agg_parts
    gc.collect()

    hourly = (
        hourly.groupby(group_keys, dropna=False)
        .agg(pax=("pax", "sum"), unique_cards=("unique_cards", "sum"))
        .reset_index()
    )

    hourly = hourly.merge(
        ref_places[["PLACE_ID", "TYPE_ID", "ST_CODE", "ST_NAME", "LN_CODE", "LN_NAME", "IS_TEST"]],
        on="PLACE_ID",
        how="left",
    )
    hourly = hourly[hourly["IS_TEST"] != 1].drop(columns="IS_TEST")
    hourly = hourly.merge(
        ref_transport.rename(columns={"TRANSPORT_ID": "TRANSPORT_TYPE_ID", "NAME": "TRANSPORT_NAME"}),
        on="TRANSPORT_TYPE_ID",
        how="left",
    )
    hourly = hourly.merge(
        ref_routes[["WAY_ID", "NAME"]].rename(columns={"WAY_ID": "BUS_RT_NO", "NAME": "ROUTE_NAME"}),
        on="BUS_RT_NO",
        how="left",
    )

    hourly["date_hour"] = pd.to_datetime(hourly["date_hour"])
    hourly["tcat"] = hourly.apply(transport_category, axis=1)
    hourly["hour"] = hourly["date_hour"].dt.hour
    hourly["date"] = hourly["date_hour"].dt.normalize()
    hourly["dow"] = hourly["date_hour"].dt.dayofweek
    hourly["month"] = hourly["date_hour"].dt.month
    hourly["is_wknd"] = (hourly["dow"] >= 5).astype(np.int8)
    hourly["is_hol"] = hourly["date_hour"].map(lambda dt: int((dt.month, dt.day) in config.HOLIDAYS_MD))
    hourly["object_id"] = hourly.apply(object_id_from_row, axis=1)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(config.HOURLY_PARQUET, index=False)
    print(f"saved: {config.HOURLY_PARQUET} ({len(hourly):,} rows)")
    create_object_hourly_parquet(force=True, hourly=hourly)
    return config.HOURLY_PARQUET


def create_object_hourly_parquet(force=False, hourly=None):
    if config.OBJECT_HOURLY_PARQUET.exists() and not force:
        print(f"{config.OBJECT_HOURLY_PARQUET} already exists. Skipping object aggregation.")
        return config.OBJECT_HOURLY_PARQUET

    if hourly is None:
        if not config.HOURLY_PARQUET.exists():
            create_hourly_parquet()
        hourly = pd.read_parquet(config.HOURLY_PARQUET)

    hourly = hourly.copy()
    hourly["date_hour"] = pd.to_datetime(hourly["date_hour"])
    if "object_id" not in hourly.columns:
        hourly["object_id"] = hourly.apply(object_id_from_row, axis=1)
    if "tcat" not in hourly.columns:
        hourly["tcat"] = hourly.apply(transport_category, axis=1)

    agg_spec = {
        "pax": ("pax", "sum"),
        "PLACE_ID": ("PLACE_ID", "first"),
        "TRANSPORT_TYPE_ID": ("TRANSPORT_TYPE_ID", "first"),
        "BUS_RT_NO": ("BUS_RT_NO", "first"),
        "TYPE_ID": ("TYPE_ID", "first"),
        "ST_CODE": ("ST_CODE", "first"),
        "ST_NAME": ("ST_NAME", "first"),
        "LN_CODE": ("LN_CODE", "first"),
        "LN_NAME": ("LN_NAME", "first"),
        "TRANSPORT_NAME": ("TRANSPORT_NAME", "first"),
        "ROUTE_NAME": ("ROUTE_NAME", "first"),
        "tcat": ("tcat", "first"),
    }
    if "unique_cards" in hourly.columns:
        agg_spec["unique_cards"] = ("unique_cards", "sum")

    object_hourly = (
        hourly[hourly["object_id"].notna()]
        .groupby(["date_hour", "object_id"], dropna=False)
        .agg(**agg_spec)
        .reset_index()
    )

    object_hourly["date_hour"] = pd.to_datetime(object_hourly["date_hour"])
    object_hourly["hour"] = object_hourly["date_hour"].dt.hour
    object_hourly["date"] = object_hourly["date_hour"].dt.normalize()
    object_hourly["dow"] = object_hourly["date_hour"].dt.dayofweek
    object_hourly["month"] = object_hourly["date_hour"].dt.month
    object_hourly["is_wknd"] = (object_hourly["dow"] >= 5).astype(np.int8)
    object_hourly["is_hol"] = object_hourly["date_hour"].map(lambda dt: int((dt.month, dt.day) in config.HOLIDAYS_MD))

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    object_hourly.to_parquet(config.OBJECT_HOURLY_PARQUET, index=False)
    print(f"saved: {config.OBJECT_HOURLY_PARQUET} ({len(object_hourly):,} rows)")
    return config.OBJECT_HOURLY_PARQUET


def _normalize_loaded_hourly(hourly):
    hourly = hourly.copy()
    hourly["date_hour"] = pd.to_datetime(hourly["date_hour"])
    if "hour" not in hourly.columns:
        hourly["hour"] = hourly["date_hour"].dt.hour
    if "date" not in hourly.columns:
        hourly["date"] = hourly["date_hour"].dt.normalize()
    if "dow" not in hourly.columns:
        hourly["dow"] = hourly["date_hour"].dt.dayofweek
    if "month" not in hourly.columns:
        hourly["month"] = hourly["date_hour"].dt.month
    if "is_wknd" not in hourly.columns:
        hourly["is_wknd"] = (hourly["dow"] >= 5).astype(np.int8)
    if "is_hol" not in hourly.columns:
        hourly["is_hol"] = hourly["date_hour"].map(lambda dt: int((dt.month, dt.day) in config.HOLIDAYS_MD))
    if "tcat" not in hourly.columns:
        hourly["tcat"] = hourly.apply(transport_category, axis=1)
    if "tcat" in hourly.columns:
        hourly["tcat"] = hourly["tcat"].replace({"Метро": "Metro", "НГПТ": "NGPT", "МЦД": "MCD", "Другое": "Other"})
    if "object_id" not in hourly.columns:
        hourly["object_id"] = hourly.apply(object_id_from_row, axis=1)
    return hourly[hourly["object_id"].notna()].copy()


def load_hourly():
    if config.OBJECT_HOURLY_PARQUET.exists():
        return _normalize_loaded_hourly(pd.read_parquet(config.OBJECT_HOURLY_PARQUET))

    if not config.HOURLY_PARQUET.exists():
        create_hourly_parquet()
    create_object_hourly_parquet()
    return _normalize_loaded_hourly(pd.read_parquet(config.OBJECT_HOURLY_PARQUET))


def make_pivot(hourly, top_n=config.DEFAULT_TOP_N, min_total_pax=1):
    grouped = hourly.groupby(["date_hour", "object_id"])["pax"].sum().reset_index()
    pivot = grouped.pivot(index="date_hour", columns="object_id", values="pax").fillna(0.0)
    pivot = pivot.sort_index()

    full_index = pd.date_range(pivot.index.min(), pivot.index.max(), freq="h")
    pivot = pivot.reindex(full_index).fillna(0.0)

    totals = pivot.sum().sort_values(ascending=False)
    keep = totals[totals >= min_total_pax].head(top_n).index
    return pivot.loc[:, keep].astype(np.float32)
