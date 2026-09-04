from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from shelfcast.api import create_app
from shelfcast.data import VALIDATION_START, CALIBRATION_START, TEST_START
from shelfcast.features import FEATURES, supervised, inference_features
from shelfcast.models import train, load_bundle


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    weeks = pd.date_range("2009-12-07", periods=104, freq="W-MON")
    rng = np.random.default_rng(91)
    p = pd.concat([pd.DataFrame({"sku": sku, "week": weeks, "units": rng.poisson(mean, len(weeks)).astype(float)})
                   for sku, mean in (("A", 20), ("B", 70))], ignore_index=True)
    frame = supervised(p, ["A", "B"])
    tr = frame.loc[frame.week < VALIDATION_START]
    val = frame.loc[frame.week.ge(VALIDATION_START) & frame.week.lt(CALIBRATION_START)]
    cal = frame.loc[frame.week.ge(CALIBRATION_START) & frame.week.lt(TEST_START)]
    bundle, _, _ = train(tr, val, cal, ["A", "B"])
    path = tmp_path_factory.mktemp("artifact") / "model.joblib"
    bundle.save(path)
    h = p.loc[p.sku.eq("A") & p.week.lt(TEST_START)].tail(52)
    request = {"sku": "A", "forecast_week": str(TEST_START.date()), "inventory_position": 5,
               "history": [{"week": str(r.week.date()), "units": r.units} for r in h.itertuples()]}
    return path, bundle, request, h


def test_actual_training_save_load_and_endpoint_parity(trained):
    path, original, request, history = trained
    x = inference_features("A", request["forecast_week"], history, ["A", "B"])
    expected = original.predict(x).iloc[0]
    loaded = load_bundle(path)
    pd.testing.assert_frame_equal(original.predict(x), loaded.predict(x[FEATURES[::-1]]))
    assert expected.lower <= expected.q10 <= expected.q50 <= expected.q80 <= expected.q90 <= expected.upper
    with TestClient(create_app(path)) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/forecast", json=request)
        assert response.status_code == 200
        body = response.json()
        assert body["point_forecast_units"] == pytest.approx(expected.point)
        assert body["suggested_order_units"] == max(0, body["stock_target_units"] - 5)
        assert client.post("/forecast", json={**request, "inventory_position": 100000}).json()["suggested_order_units"] == 0


def test_endpoint_rejects_unknown_products_and_invalid_time(trained):
    path, _, request, _ = trained
    with TestClient(create_app(path)) as client:
        for update in ({"sku": "missing"}, {"forecast_week": "2011-09-06"},
                       {"forecast_week": "2011-01-03"}, {"inventory_position": -1}, {"future_sales": 100}):
            assert client.post("/forecast", json={**request, **update}).status_code == 422
        assert client.post("/forecast", json={**request, "history": request["history"][:-1]}).status_code == 422


def test_missing_and_incompatible_models_fail_explicitly(trained, tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(tmp_path / "absent.joblib")):
            pass
    path, _, _, _ = trained
    invalid = load_bundle(path)
    invalid.schema_version = "outdated"
    invalid.save(tmp_path / "invalid.joblib")
    with pytest.raises(ValueError, match="Incompatible"):
        load_bundle(tmp_path / "invalid.joblib")
