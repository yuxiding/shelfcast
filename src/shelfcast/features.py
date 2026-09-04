"""Exactly the same past-only feature builder for offline replay and serving."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["sku_index", "lag_1", "lag_2", "lag_4", "lag_8", "mean_4", "mean_8", "mean_13",
            "std_8", "zero_fraction_8", "trend_4_8", "annual_lag", "annual_available", "week_sin", "week_cos"]
SCHEMA_VERSION = "shelfcast-weekly-v1"


def validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"sku", "week", "units"}
    if not required.issubset(panel):
        raise ValueError("Panel must contain sku, week, units")
    p = panel.copy()
    p["sku"] = p["sku"].astype(str)
    p["week"] = pd.to_datetime(p["week"])
    if p.duplicated(["sku", "week"]).any():
        raise ValueError("Duplicate sku/week rows")
    if p["week"].isna().any() or p["week"].dt.tz is not None or not p["week"].dt.dayofweek.eq(0).all() or not p["week"].eq(p["week"].dt.normalize()).all():
        raise ValueError("Weeks must be timezone-naive Monday dates")
    if not np.isfinite(p["units"]).all() or p["units"].lt(0).any():
        raise ValueError("Units must be finite and nonnegative")
    p = p.sort_values(["sku", "week"]).reset_index(drop=True)
    for _, rows in p.groupby("sku"):
        if not rows["week"].diff().dropna().eq(pd.Timedelta(days=7)).all():
            raise ValueError("Missing weeks must be represented explicitly with units=0")
    return p


def feature_row(history: np.ndarray, week: pd.Timestamp, sku_index: int) -> dict:
    h = np.asarray(history, dtype=float)
    if len(h) < 8 or not np.isfinite(h).all() or (h < 0).any():
        raise ValueError("At least eight finite nonnegative history values are required")
    mean4, mean8 = float(h[-4:].mean()), float(h[-8:].mean())
    angle = 2 * np.pi * float(week.isocalendar().week) / 52.1775
    return dict(zip(FEATURES, [sku_index, h[-1], h[-2], h[-4], h[-8], mean4, mean8,
                h[-13:].mean(), h[-8:].std(), (h[-8:] == 0).mean(), mean4 / max(mean8, 1),
                h[-52] if len(h) >= 52 else mean8, float(len(h) >= 52), np.sin(angle), np.cos(angle)]))


def supervised(panel: pd.DataFrame, sku_order: list[str]) -> pd.DataFrame:
    p = validate_panel(panel)
    mapping = {sku: i for i, sku in enumerate(sku_order)}
    if set(p.sku) - set(mapping):
        raise ValueError("Unknown product")
    rows = []
    for sku, group in p.groupby("sku", sort=True):
        units = group.units.to_numpy()
        for i in range(8, len(group)):
            week = group.iloc[i].week
            row = feature_row(units[:i], week, mapping[sku])
            rows.append({"sku": sku, "week": week, "units": units[i], **row})
    return pd.DataFrame(rows).sort_values(["week", "sku"]).reset_index(drop=True)


def inference_features(sku: str, forecast_week: str, history: pd.DataFrame, sku_order: list[str]) -> pd.DataFrame:
    if sku not in sku_order:
        raise ValueError(f"Unknown product {sku}; retrain to extend the cohort")
    week = pd.Timestamp(forecast_week)
    if week.tz is not None or week.dayofweek != 0 or week != week.normalize():
        raise ValueError("forecast_week must be a Monday date")
    p = validate_panel(history.assign(sku=sku))
    if p.empty or p.week.max() != week - pd.Timedelta(days=7):
        raise ValueError("History must end exactly one week before forecast_week")
    if (p.week >= week).any():
        raise ValueError("History cannot include the forecast week or future weeks")
    return pd.DataFrame([feature_row(p.units.to_numpy(), week, sku_order.index(sku))], columns=FEATURES)
