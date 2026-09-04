"""One command from an attributed weekly panel to frozen forecasts and evidence."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from .data import VALIDATION_START, CALIBRATION_START, TEST_START, file_hash
from .features import supervised
from .models import BASELINES, train, predict_point, wape
from .evaluation import (tune_multiplier, forecast_metrics, interval_metrics, decision_metrics,
                         block_bootstrap_cost_reduction, stock_outcomes)
from .reporting import plots, write_report


def source_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def run(panel_path: Path, output: Path, audit: dict | None = None, names: dict | None = None) -> dict:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(panel_path, dtype={"sku": str}, parse_dates=["week"])
    sku_order = sorted(panel.sku.unique())
    names = names or {s: s for s in sku_order}
    frame = supervised(panel, sku_order)
    train_rows = frame.loc[frame.week < VALIDATION_START].copy()
    validation = frame.loc[frame.week.ge(VALIDATION_START) & frame.week.lt(CALIBRATION_START)].copy()
    calibration = frame.loc[frame.week.ge(CALIBRATION_START) & frame.week.lt(TEST_START)].copy()
    test = frame.loc[frame.week >= TEST_START].copy().reset_index(drop=True)
    if any(x.empty for x in (train_rows, validation, calibration, test)):
        raise ValueError("The panel must cover all four predefined time blocks")
    bundle, (ml_spec, ml_model), val_preds = train(train_rows, validation, calibration, sku_order)
    mean_factor, mean_search = tune_multiplier(validation.units, val_preds["mean_4"])
    point_factor, point_search = tune_multiplier(validation.units, val_preds[bundle.point_spec])
    prediction = bundle.predict(test)
    prediction = pd.concat([test[["sku", "week", "units"]], prediction], axis=1)
    point_predictions = {s: predict_point(test, s) for s in BASELINES}
    point_predictions["selected_ml"] = predict_point(test, ml_spec, ml_model)
    point_predictions["deployed"] = prediction.point.to_numpy()
    metrics = forecast_metrics(test.units.to_numpy(), point_predictions)
    uncertainty = interval_metrics(test.units, prediction)
    targets = {"mean4": test.mean_4.to_numpy(), "mean4_tuned": test.mean_4.to_numpy() * mean_factor,
               "point": prediction.point.to_numpy(), "point_tuned": prediction.point.to_numpy() * point_factor,
               "quantile_80": prediction.q80.to_numpy()}
    costs, details = decision_metrics(test, targets)
    bootstrap = block_bootstrap_cost_reduction(details)
    sensitivity = pd.DataFrame([{"shortage_to_excess_ratio": ratio, "policy": name,
        "total_cost": float(stock_outcomes(test.units.to_numpy(), target, ratio, 1)["cost"].sum())}
        for ratio in (1, 2, 4, 8) for name, target in targets.items()])
    weekly = []
    for week, rows in prediction.groupby("week"):
        weekly.append({"week": week, "wape": wape(rows.units, rows.point),
            "coverage": float(((rows.units >= rows.lower) & (rows.units <= rows.upper)).mean()),
            "recorded_units": float(rows.units.sum()), "predicted_units": float(rows.point.sum())})
    weekly = pd.DataFrame(weekly)
    # Illustration chosen from training sales rank, not holdout fit quality.
    training_volume = panel.loc[panel.week < VALIDATION_START].groupby("sku").units.sum().sort_values()
    sample_sku = str(training_volume.index[len(training_volume) // 2])
    metadata = bundle.metadata
    metadata.update({"panel_sha256": file_hash(panel_path), "source_sha256": source_hash(),
        "product_names": names, "sample_sku": sample_sku,
        "product_training_mean": frame.loc[frame.week < CALIBRATION_START].groupby("sku").units.mean().to_dict(),
        "mean4_stock_multiplier": mean_factor, "point_stock_multiplier": point_factor,
        "mean4_multiplier_search": mean_search, "point_multiplier_search": point_search,
        "seed": 42, "python_version": platform.python_version(),
        "dependency_versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "matplotlib", "joblib", "fastapi", "openpyxl")},
        "splits": {name: {"start": str(f.week.min().date()), "end": str(f.week.max().date()), "weeks": int(f.week.nunique()), "rows": len(f)}
            for name, f in (("train", train_rows), ("validation", validation), ("calibration", calibration), ("test", test))},
        "quantile_settings": {"quantiles": [0.1, 0.5, 0.8, 0.9], "max_iter": 220, "learning_rate": 0.06,
                              "early_stopping": False, "min_samples_leaf": 20, "l2_regularization": 2},
        "source_data_end": str(panel.week.max().date())})
    metadata["model_id"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    metadata["runtime_seconds"] = time.perf_counter() - started
    bundle.save(output / "model.joblib")
    history = panel.loc[panel.sku.eq(sample_sku) & panel.week.lt(TEST_START)].tail(52)
    example = {"sku": sample_sku, "forecast_week": str(TEST_START.date()), "inventory_position": 20,
               "history": [{"week": str(r.week.date()), "units": float(r.units)} for r in history.itertuples()]}
    for filename, value in [("run_metadata.json", metadata), ("interval_metrics.json", uncertainty),
                            ("cost_bootstrap.json", bootstrap), ("example_request.json", example)]:
        (output / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    for filename, value in [("forecast_metrics.csv", metrics), ("inventory_metrics.csv", costs),
        ("predictions.csv", prediction), ("decision_details.csv", details),
        ("weekly_metrics.csv", weekly), ("cost_sensitivity.csv", sensitivity)]:
        value.to_csv(output / filename, index=False, float_format="%.8f")
    plots(output, metrics, costs, prediction, weekly, sample_sku, names)
    audit = audit or {"cohort_products": len(sku_order)}
    write_report(output, metrics, costs, uncertainty, bootstrap, metadata, audit)
    return {"point_model": bundle.point_spec, "ml_model": ml_spec, "test_wape": wape(test.units, prediction.point),
            "interval_coverage": uncertainty["calibrated"]["coverage"], "cost_comparison": bootstrap,
            "runtime_seconds": metadata["runtime_seconds"], "output": str(output)}
