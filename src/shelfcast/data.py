"""Download an attributed public dataset and build an audited weekly panel."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd

SOURCE_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
XLSX_SHA256 = "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980"
FIRST_WEEK = pd.Timestamp("2009-12-07")
LAST_WEEK = pd.Timestamp("2011-11-28")
VALIDATION_START = pd.Timestamp("2011-05-02")
CALIBRATION_START = pd.Timestamp("2011-07-04")
TEST_START = pd.Timestamp("2011-09-05")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "online_retail_II.xlsx"
    if not path.exists():
        archive = root / "online_retail_II.zip"
        temporary = root / "download.part"
        with urllib.request.urlopen(SOURCE_URL, timeout=90) as response, temporary.open("wb") as f:
            while block := response.read(1024 * 1024):
                f.write(block)
        temporary.replace(archive)
        with zipfile.ZipFile(archive) as z:
            # Read one expected member; never extract arbitrary archive paths.
            path.write_bytes(z.read("online_retail_II.xlsx"))
    actual = file_hash(path)
    if actual != XLSX_SHA256:
        raise ValueError(f"Dataset checksum mismatch: {actual}. Review the source before proceeding.")
    return path


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.rename(columns={"Invoice": "InvoiceNo", "Price": "UnitPrice"}).copy()
    needed = {"InvoiceNo", "StockCode", "InvoiceDate", "Quantity", "UnitPrice", "Country"}
    if not needed.issubset(df):
        raise ValueError(f"Missing columns: {sorted(needed - set(df))}")
    audit = {"raw_rows": len(df)}
    # Identical recorded rows are treated as duplicates, including sheet overlap.
    df = df.drop_duplicates()
    audit["exact_duplicate_rows_removed"] = audit["raw_rows"] - len(df)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.upper()
    df["StockCode"] = df["StockCode"].astype(str).str.upper()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    for col in ("Quantity", "UnitPrice"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    filters = [
        ("invalid_numeric_or_date", df["InvoiceDate"].notna() & np.isfinite(df["Quantity"]) & np.isfinite(df["UnitPrice"])),
        ("outside_united_kingdom", df["Country"].eq("United Kingdom")),
        ("cancellation_or_nonpositive_sale", ~df["InvoiceNo"].str.startswith("C") & df["Quantity"].gt(0) & df["UnitPrice"].gt(0)),
        ("non_product_code", df["StockCode"].str.fullmatch(r"\d{5}[A-Z]{0,2}")),
        ("incomplete_boundary_week", df["InvoiceDate"].ge(FIRST_WEEK) & df["InvoiceDate"].lt(LAST_WEEK + pd.Timedelta(days=7))),
    ]
    for name, mask in filters:
        previous = len(df)
        df = df.loc[mask.reindex(df.index).fillna(False)]
        audit[name + "_removed"] = previous - len(df)
    df["week"] = df["InvoiceDate"].dt.normalize() - pd.to_timedelta(df["InvoiceDate"].dt.dayofweek, unit="D")
    audit["clean_rows"] = len(df)
    audit["clean_units"] = float(df["Quantity"].sum())
    audit["quantity_policy"] = "Positive recorded sales; cancellations excluded, not netted. No bulk-sale trimming."
    return df, audit


def build_panel(raw: pd.DataFrame, top_k: int = 50) -> tuple[pd.DataFrame, dict, dict]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    df, audit = clean_transactions(raw)
    # Cohort membership and descriptions never inspect validation or test rows.
    train = df.loc[df["week"] < VALIDATION_START]
    ranking = train.groupby("StockCode").agg(units=("Quantity", "sum"), active_weeks=("week", "nunique"))
    ranking = ranking.loc[ranking["active_weeks"] >= 30]
    ranking = ranking.reset_index().sort_values(["units", "StockCode"], ascending=[False, True]).head(top_k)
    skus = sorted(ranking["StockCode"].tolist())
    if len(skus) < top_k:
        raise ValueError(f"Only {len(skus)} products satisfy the training-only activity threshold")
    weeks = pd.date_range(FIRST_WEEK, LAST_WEEK, freq="W-MON")
    index = pd.MultiIndex.from_product([skus, weeks], names=["sku", "week"])
    totals = df[df["StockCode"].isin(skus)].groupby(["StockCode", "week"])["Quantity"].sum()
    totals.index.names = ["sku", "week"]
    panel = totals.reindex(index, fill_value=0).rename("units").reset_index()
    panel["units"] = panel["units"].astype(float)
    names = {}
    for sku in skus:
        values = train.loc[train["StockCode"].eq(sku), "Description"].dropna() if "Description" in train else pd.Series(dtype=str)
        names[sku] = str(values.mode().iloc[0]) if len(values) else sku
    audit.update({"cohort_products": len(skus), "weeks": len(weeks), "panel_rows": len(panel),
                  "cohort_units": float(panel.units.sum()), "zero_sales_fraction": float(panel.units.eq(0).mean()),
                  "cohort_selection_end_exclusive": str(VALIDATION_START.date()),
                  "first_week": str(FIRST_WEEK.date()), "last_week": str(LAST_WEEK.date())})
    return panel, audit, names


def prepare(root: Path, top_k: int = 50) -> tuple[pd.DataFrame, dict, dict]:
    path = download(root)
    # CSV cache is data, never executable pickle. The source hash guards its provenance.
    cache = root / "transactions.csv.gz"
    stamp = root / "cache.json"
    cache_valid = False
    if cache.exists() and stamp.exists():
        info = json.loads(stamp.read_text())
        cache_valid = info.get("source_sha256") == XLSX_SHA256 and info.get("cache_sha256") == file_hash(cache)
    if cache_valid:
        raw = pd.read_csv(cache, dtype={"Invoice": str, "StockCode": str})
    else:
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl", dtype={"Invoice": str, "StockCode": str})
        raw = pd.concat(sheets.values(), ignore_index=True)
        raw.to_csv(cache, index=False, compression="gzip")
        stamp.write_text(json.dumps({"source_sha256": XLSX_SHA256, "cache_sha256": file_hash(cache)}))
    panel, audit, names = build_panel(raw, top_k)
    audit["source_sha256"] = XLSX_SHA256
    audit["source_url"] = SOURCE_URL
    return panel, audit, names
