"""Validation-only selection, frozen forecasts, and held-out interval calibration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss
from threadpoolctl import threadpool_limits

from .features import FEATURES, SCHEMA_VERSION

QUANTILES = (0.1, 0.5, 0.8, 0.9)
BASELINES = ("last_week", "mean_4", "annual_52")


def wape(y, prediction) -> float:
    y = np.asarray(y, dtype=float)
    denominator = np.abs(y).sum()
    return float(np.abs(y - prediction).sum() / denominator) if denominator else float("nan")


def fit_point(frame: pd.DataFrame, spec: str):
    if spec in BASELINES:
        return None
    family, leaves = spec.rsplit("_", 1)
    target = frame.units.to_numpy()
    model = HistGradientBoostingRegressor(
        loss="poisson" if family == "poisson" else "squared_error", max_leaf_nodes=int(leaves),
        max_iter=220, learning_rate=0.06, min_samples_leaf=20, l2_regularization=2,
        categorical_features=["sku_index"], early_stopping=False, random_state=42)
    with threadpool_limits(limits=1):
        model.fit(frame[FEATURES], np.log1p(target) if family == "log" else target)
    return model


def predict_point(frame: pd.DataFrame, spec: str, model=None) -> np.ndarray:
    if spec in BASELINES:
        return frame[{"last_week": "lag_1", "mean_4": "mean_4", "annual_52": "annual_lag"}[spec]].to_numpy(float)
    with threadpool_limits(limits=1):
        result = model.predict(frame[FEATURES])
    return np.maximum(0, np.expm1(result) if spec.startswith("log_") else result)


def fit_quantiles(frame: pd.DataFrame, leaves: int) -> dict:
    models = {}
    with threadpool_limits(limits=1):
        for q in QUANTILES:
            model = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_leaf_nodes=leaves,
                max_iter=220, learning_rate=0.06, min_samples_leaf=20, l2_regularization=2,
                categorical_features=["sku_index"], early_stopping=False, random_state=42)
            models[q] = model.fit(frame[FEATURES], frame.units)
    return models


def predict_quantiles(frame: pd.DataFrame, models: dict) -> np.ndarray:
    with threadpool_limits(limits=1):
        result = np.column_stack([models[q].predict(frame[FEATURES]) for q in QUANTILES])
    # Monotone rearrangement is specified before validation/calibration/test.
    return np.sort(np.maximum(result, 0), axis=1)


def conformal_adjustment(y, lower, upper, scale, alpha: float = 0.2) -> float:
    y, lower, upper, scale = map(lambda x: np.asarray(x, dtype=float), (y, lower, upper, scale))
    if not 0 < alpha < 1 or len(y) == 0 or any(x.shape != y.shape for x in (lower, upper, scale)):
        raise ValueError("Invalid calibration inputs")
    if not all(np.isfinite(x).all() for x in (y, lower, upper, scale)) or (scale <= 0).any() or (upper < lower).any():
        raise ValueError("Calibration inputs must be finite, ordered, with positive scales")
    scores = np.maximum.reduce([lower - y, y - upper, np.zeros_like(y)]) / scale
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    if rank > len(scores):
        raise ValueError("Not enough calibration rows for the requested alpha")
    return float(np.sort(scores)[rank - 1])


@dataclass
class ForecastBundle:
    sku_order: list[str]
    point_spec: str
    point_model: object
    quantile_models: dict
    interval_adjustment: float
    metadata: dict
    feature_names: tuple = tuple(FEATURES)
    schema_version: str = SCHEMA_VERSION

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.schema_version != SCHEMA_VERSION or self.feature_names != tuple(FEATURES):
            raise ValueError("Incompatible feature schema")
        x = frame[FEATURES]
        point = predict_point(x, self.point_spec, self.point_model)
        q = predict_quantiles(x, self.quantile_models)
        scale = np.maximum(x.mean_8.to_numpy(), 1)
        correction = self.interval_adjustment * scale
        return pd.DataFrame({"point": point, "q10": q[:, 0], "q50": q[:, 1], "q80": q[:, 2], "q90": q[:, 3],
            "lower": np.maximum(0, q[:, 0] - correction), "upper": q[:, 3] + correction}, index=frame.index)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


def load_bundle(path: Path) -> ForecastBundle:
    if not path.is_file():
        raise FileNotFoundError(f"No trained model at {path}; run shelfcast reproduce first")
    # joblib is for locally trained/trusted artifacts only, never request uploads.
    bundle = joblib.load(path)
    if not isinstance(bundle, ForecastBundle) or bundle.schema_version != SCHEMA_VERSION or bundle.feature_names != tuple(FEATURES):
        raise ValueError("Incompatible model artifact")
    if bundle.metadata.get("sklearn_version", "").split(".")[:2] != sklearn.__version__.split(".")[:2]:
        raise ValueError("scikit-learn version differs from training; reproduce the model in this environment")
    return bundle


def train(train: pd.DataFrame, validation: pd.DataFrame, calibration: pd.DataFrame, sku_order: list[str]):
    if not (train.week.max() < validation.week.min() <= validation.week.max() < calibration.week.min()):
        raise ValueError("Training, validation, and calibration must be disjoint chronological blocks")
    specs = [*BASELINES, "poisson_7", "poisson_15", "log_7", "log_15"]
    scores, validation_predictions = [], {}
    for spec in specs:
        fitted = fit_point(train, spec)
        prediction = predict_point(validation, spec, fitted)
        validation_predictions[spec] = prediction
        scores.append({"model": spec, "validation_wape": wape(validation.units.to_numpy(), prediction)})
    winner = min(scores, key=lambda x: x["validation_wape"])["model"]
    ml_winner = min((x for x in scores if x["model"] not in BASELINES), key=lambda x: x["validation_wape"])["model"]
    quantile_scores = []
    for leaves in (7, 15):
        fitted = fit_quantiles(train, leaves)
        preds = predict_quantiles(validation, fitted)
        loss = float(np.mean([mean_pinball_loss(validation.units, preds[:, i], alpha=q) for i, q in enumerate(QUANTILES)]))
        quantile_scores.append({"max_leaf_nodes": leaves, "validation_mean_pinball_loss": loss})
    quantile_leaves = min(quantile_scores, key=lambda x: x["validation_mean_pinball_loss"])["max_leaf_nodes"]
    refit = pd.concat([train, validation], ignore_index=True)
    point_model = fit_point(refit, winner)
    ml_model = point_model if ml_winner == winner else fit_point(refit, ml_winner)
    quantile_models = fit_quantiles(refit, quantile_leaves)
    q = predict_quantiles(calibration, quantile_models)
    adjustment = conformal_adjustment(calibration.units, q[:, 0], q[:, 3], np.maximum(calibration.mean_8, 1))
    metadata = {"selected_point_model": winner, "selected_ml_model": ml_winner,
        "point_candidates": scores, "quantile_candidates": quantile_scores, "quantile_leaves": quantile_leaves,
        "fit_end_exclusive": str(calibration.week.min().date()), "calibration_end": str(calibration.week.max().date()),
        "sklearn_version": sklearn.__version__, "calibration_rows": len(calibration),
        "interval_adjustment": adjustment, "nominal_interval_coverage": 0.8,
        "interval_note": "Empirical calibration only: temporal and cross-product dependence prevent an exchangeability guarantee."}
    bundle = ForecastBundle(sku_order, winner, point_model, quantile_models, adjustment, metadata)
    return bundle, (ml_winner, ml_model), validation_predictions
