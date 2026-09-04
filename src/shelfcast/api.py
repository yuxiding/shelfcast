"""Serve reproducible weekly forecasts from a required, trusted local artifact."""
from contextlib import asynccontextmanager
from datetime import date
import math
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .features import inference_features
from .models import load_bundle


class HistoryWeek(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    week: date
    units: float = Field(ge=0)


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    sku: str = Field(min_length=1, max_length=20)
    forecast_week: date
    history: list[HistoryWeek] = Field(min_length=8, max_length=156)
    inventory_position: int = Field(default=0, ge=0)


def create_app(model_path: str | Path | None = None) -> FastAPI:
    path = Path(model_path or os.environ.get("SHELFCAST_MODEL", "outputs/model.joblib"))

    @asynccontextmanager
    async def lifespan(app):
        app.state.bundle = load_bundle(path)
        yield

    app = FastAPI(title="ShelfCast", version="0.1.0", lifespan=lifespan,
                  description="One-week forecasts for the documented product cohort; inventory costs are hypothetical.")

    @app.get("/health")
    def health():
        bundle = app.state.bundle
        return {"status": "ok", "schema": bundle.schema_version, "products": len(bundle.sku_order),
                "model": bundle.point_spec, "model_id": bundle.metadata.get("model_id")}

    @app.get("/products")
    def products():
        bundle = app.state.bundle
        return {"products": [{"sku": s, "description": bundle.metadata.get("product_names", {}).get(s, s)} for s in bundle.sku_order]}

    @app.post("/forecast")
    def forecast(request: ForecastRequest):
        bundle = app.state.bundle
        if pd.Timestamp(request.forecast_week) <= pd.Timestamp(bundle.metadata["calibration_end"]):
            raise HTTPException(422, "Forecast week must be after the artifact's calibration period")
        try:
            frame = inference_features(request.sku, str(request.forecast_week),
                pd.DataFrame([h.model_dump() for h in request.history]), bundle.sku_order)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        values = bundle.predict(frame).iloc[0]
        target = math.ceil(values.q80)
        training_mean = bundle.metadata.get("product_training_mean", {}).get(request.sku)
        ratio = float(frame.mean_4.iloc[0] / max(training_mean, 1)) if training_mean is not None else None
        shift = ratio is not None and (ratio < 0.5 or ratio > 2)
        return {"sku": request.sku, "forecast_week": str(request.forecast_week),
            "point_forecast_units": float(values.point), "median_units": float(values.q50),
            "interval_80": {"lower": float(values.lower), "upper": float(values.upper),
                            "interpretation": "Empirically calibrated range; future coverage is not guaranteed."},
            "stock_target_units": target, "suggested_order_units": max(0, target - request.inventory_position),
            "stock_rule": "80th percentile; illustrative underage:overage cost ratio 4:1, zero lead time",
            "recent_to_training_mean_ratio": ratio, "history_shift_warning": shift,
            "model_id": bundle.metadata.get("model_id")}

    return app


app = create_app()
