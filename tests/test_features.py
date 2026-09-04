import numpy as np
import pandas as pd
import pytest

from shelfcast.features import FEATURES, inference_features, supervised, validate_panel


def panel():
    return pd.DataFrame({"sku": "A", "week": pd.date_range("2010-01-04", periods=65, freq="W-MON"),
                         "units": np.arange(65, dtype=float)})


def test_current_and_future_targets_cannot_change_forecast_features():
    p = panel()
    before = supervised(p, ["A"])
    cutoff = p.week.iloc[55]
    changed = p.copy()
    changed.loc[changed.week >= cutoff, "units"] = 100000
    after = supervised(changed, ["A"])
    pd.testing.assert_frame_equal(before.loc[before.week <= cutoff, FEATURES], after.loc[after.week <= cutoff, FEATURES])
    row = before.loc[before.week == cutoff].iloc[0]
    assert row.lag_1 == 54
    assert row.annual_lag == 3
    assert row.mean_4 == np.mean([51, 52, 53, 54])


def test_batch_and_api_feature_parity_even_when_history_is_reordered():
    p = panel()
    frame = supervised(p, ["A"])
    for i in (8, 20, 52, 64):
        week = p.week.iloc[i]
        online = inference_features("A", str(week.date()), p.iloc[:i].sample(frac=1, random_state=9), ["A"])
        offline = frame.loc[frame.week == week, FEATURES].reset_index(drop=True)
        pd.testing.assert_frame_equal(online, offline, check_dtype=False)


def test_missing_duplicate_non_monday_and_invalid_history_are_rejected():
    p = panel()
    cases = [p.drop(index=10), pd.concat([p, p.iloc[[3]]]), p.assign(units=-1),
             p.assign(week=p.week + pd.Timedelta(days=1))]
    for bad in cases:
        with pytest.raises(ValueError):
            validate_panel(bad)
    with pytest.raises(ValueError, match="Unknown"):
        inference_features("B", "2011-04-04", p, ["A"])
    with pytest.raises(ValueError, match="end exactly"):
        inference_features("A", "2012-04-02", p, ["A"])
