"""Forecast metrics and explicitly hypothetical single-period stock decisions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_pinball_loss

from .models import wape


def stock_outcomes(demand, target, underage_cost=4.0, overage_cost=1.0) -> dict:
    demand, target = np.asarray(demand, dtype=float), np.asarray(target, dtype=float)
    if demand.shape != target.shape or not np.isfinite(demand).all() or not np.isfinite(target).all() or (demand < 0).any():
        raise ValueError("Demand and stock targets must be finite arrays of equal shape; demand cannot be negative")
    if not np.isfinite([underage_cost, overage_cost]).all() or min(underage_cost, overage_cost) <= 0:
        raise ValueError("Costs must be finite and positive")
    target = np.ceil(np.maximum(target, 0))
    short = np.maximum(demand - target, 0)
    excess = np.maximum(target - demand, 0)
    return {"target": target, "shortage": short, "excess": excess,
            "cost": underage_cost * short + overage_cost * excess,
            "filled": demand - short}


def tune_multiplier(demand, point) -> tuple[float, list]:
    scores = []
    for factor in (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
        result = stock_outcomes(demand, np.asarray(point) * factor)
        scores.append({"multiplier": factor, "validation_cost": float(result["cost"].sum())})
    return min(scores, key=lambda x: x["validation_cost"])["multiplier"], scores


def forecast_metrics(y, predictions: dict) -> pd.DataFrame:
    return pd.DataFrame([{"model": name, "mae": mean_absolute_error(y, pred), "wape": wape(y, pred),
                          "bias_units": float(np.mean(np.asarray(pred) - np.asarray(y)))} for name, pred in predictions.items()])


def interval_metrics(y, prediction: pd.DataFrame) -> dict:
    y = np.asarray(y)
    def metrics(lo, hi):
        return {"coverage": float(((y >= lo) & (y <= hi)).mean()), "mean_width": float(np.mean(hi - lo)),
                "below_lower": float((y < lo).mean()), "above_upper": float((y > hi).mean())}
    return {"nominal_coverage": 0.8, "raw": metrics(prediction.q10, prediction.q90),
            "calibrated": metrics(prediction.lower, prediction.upper),
            "q80_pinball_loss": float(mean_pinball_loss(y, prediction.q80, alpha=0.8)),
            "q80_empirical_coverage": float((y <= prediction.q80).mean())}


def decision_metrics(test: pd.DataFrame, targets: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate, details = [], []
    demand = test.units.to_numpy()
    for name, target in targets.items():
        outcome = stock_outcomes(demand, target)
        aggregate.append({"policy": name, "total_cost": float(outcome["cost"].sum()),
            "shortage_units": float(outcome["shortage"].sum()), "excess_units": float(outcome["excess"].sum()),
            "fill_rate": float(outcome["filled"].sum() / demand.sum()),
            "shortage_week_fraction": float((outcome["shortage"] > 0).mean()),
            "mean_stock_target": float(outcome["target"].mean())})
        details.append(test[["sku", "week", "units"]].reset_index(drop=True).assign(policy=name, **outcome))
    return pd.DataFrame(aggregate), pd.concat(details, ignore_index=True)


def block_bootstrap_cost_reduction(details: pd.DataFrame, baseline="mean4_tuned", candidate="quantile_80", draws=2000, block_length=3) -> dict:
    weekly = details.groupby(["week", "policy"]).cost.sum().unstack()
    base, cand = weekly[baseline].to_numpy(), weekly[candidate].to_numpy()
    n = len(base)
    if n < block_length or (base.sum() <= 0):
        raise ValueError("Insufficient nonzero baseline weeks")
    rng = np.random.default_rng(42)
    values = []
    for _ in range(draws):
        starts = rng.integers(0, n - block_length + 1, size=int(np.ceil(n / block_length)))
        idx = np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]
        values.append(1 - cand[idx].sum() / base[idx].sum())
    return {"baseline": baseline, "candidate": candidate,
            "relative_cost_reduction": float(1 - cand.sum() / base.sum()),
            "bootstrap_95_percentile_interval": [float(x) for x in np.quantile(values, [0.025, 0.975])],
            "draws": draws, "block_length_weeks": block_length, "test_weeks": n,
            "interpretation": "Descriptive paired moving-block bootstrap; 13 weeks do not establish generalization to other seasons or retailers."}
