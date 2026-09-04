import numpy as np
import pytest

from shelfcast.evaluation import stock_outcomes
from shelfcast.models import conformal_adjustment


def test_stock_costs_rounding_and_zero_demand():
    result = stock_outcomes([10, 0, 4], [7.1, 3, 5])
    np.testing.assert_equal(result["target"], [8, 3, 5])
    np.testing.assert_equal(result["shortage"], [2, 0, 0])
    np.testing.assert_equal(result["excess"], [0, 3, 1])
    np.testing.assert_equal(result["cost"], [8, 3, 1])
    np.testing.assert_equal(result["filled"], [8, 0, 4])
    with pytest.raises(ValueError):
        stock_outcomes([10], [float("nan")])


def test_eightieth_percentile_minimizes_four_to_one_single_period_cost():
    # Independent enumeration verifies the decision rule, not a copy of its code.
    demand = np.array([0, 1, 2, 3, 4])
    candidates = range(7)
    costs = [stock_outcomes(demand, np.repeat(s, len(demand)))["cost"].sum() for s in candidates]
    assert candidates[int(np.argmin(costs))] in (3, 4)
    assert costs[0] > costs[3] and costs[6] > costs[4]


def test_calibration_uses_finite_sample_rank_and_nonshrinking_intervals():
    # n=9, alpha=.2 -> ceil(10*.8)=8th order statistic, not the usual interpolated quantile.
    y = np.arange(1, 10, dtype=float)
    adjustment = conformal_adjustment(y, np.zeros(9), np.zeros(9), np.ones(9))
    assert adjustment == 8
    assert conformal_adjustment(np.ones(9), np.zeros(9), np.repeat(2, 9), np.ones(9)) == 0
    with pytest.raises(ValueError):
        conformal_adjustment([1], [0], [1], [1], alpha=0.01)
