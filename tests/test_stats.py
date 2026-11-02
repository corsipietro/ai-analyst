"""Tests of the pure numerical helpers (no database needed)."""
import pytest
from scipy import stats as scipy_stats

from sim_analyst.stats import (bootstrap_difference, clean, nice_step, sample_points, shape_of,
                               round_significant, trend_summary, turning_points, welch_interval)


def test_clean_rounds_and_turns_whole_floats_into_int():
    """40.0 -> 40, 1.23456 -> 1.235, NaN -> None, text unchanged."""
    assert clean(40.0) == 40 and isinstance(clean(40.0), int)
    assert clean(1.23456) == 1.235
    assert clean(float("nan")) is None
    assert clean("a") == "a"


def test_nice_step_is_round():
    """About 12 points with a step of 1, 2 or 5 times a power of ten."""
    assert nice_step(60) == 5
    assert nice_step(10) == 1
    assert nice_step(200) == 20


def test_turning_points_and_shape():
    """A series that rises, falls and rises again has a peak and a trough; a flat top counts at its first time."""
    times = [0, 1, 2, 3, 4, 5]
    means = [0, 2, 2, 1, 3, 4]
    assert turning_points(times, means) == [{"kind": "peak", "time": 1, "mean": 2},
                                            {"kind": "trough", "time": 3, "mean": 1}]
    assert shape_of(means) == "not monotonic"
    assert shape_of([1, 1, 2]) == "increasing"
    assert shape_of([3, 3, 3]) == "constant"


def test_sample_points_keeps_the_last():
    """Points at start + k*step, plus the last one."""
    points = [(t, 1, t, t, t) for t in range(0, 11)]
    assert [p[0] for p in sample_points(points, 3, 0)] == [0, 3, 6, 9, 10]


def test_trend_summary_of_a_peak():
    """A rise then a fall: maximum and its time, decreasing last part, one peak."""
    points = [(t, 3, float(m), float(m) - 1, float(m) + 1) for t, m in enumerate([0, 5, 10, 8, 6, 4])]
    s = trend_summary(points, step=1)
    assert s["max_of_mean"] == {"time": 2, "mean": 10.0}
    assert s["slope_last_part"]["slope_per_time_unit"] < 0
    assert s["turning_points_sampled"] == [{"kind": "peak", "time": 2, "mean": 10.0}]
    assert s["final"]["spread"] == 2.0


def test_bootstrap_is_reproducible_and_brackets_the_difference():
    """Same seed, same interval; the interval contains the observed difference."""
    a, b = [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]
    first = bootstrap_difference(a, b, 2000, 7, 0.95)
    second = bootstrap_difference(a, b, 2000, 7, 0.95)
    assert first == second
    assert first["difference"] == 4.0
    assert first["interval"][0] < 4.0 < first["interval"][1] and first["interval"][0] > 0


def test_welch_interval_matches_scipy():
    """Difference, interval and degrees of freedom agree with scipy's Welch t-test."""
    a, b = [1.0, 2.0, 3.5, 4.0], [5.0, 6.5, 7.0, 9.0, 8.0]
    ours = welch_interval(a, b, 0.95)
    reference = scipy_stats.ttest_ind(b, a, equal_var=False)
    interval = reference.confidence_interval(0.95)
    assert ours["difference"] == pytest.approx(sum(b) / 5 - sum(a) / 4)
    assert ours["interval"] == pytest.approx([interval.low, interval.high])
    assert ours["degrees_of_freedom"] == pytest.approx(reference.df)


def test_welch_interval_without_variability_has_no_interval():
    """Both variances zero: the standard error is 0, so there is no interval (and no degrees of freedom)."""
    r = welch_interval([3.0, 3.0, 3.0], [5.0, 5.0], 0.95)
    assert r["difference"] == 2.0
    assert r["interval"] is None and r["degrees_of_freedom"] is None


def test_round_significant_removes_floating_point_noise():
    """1/(100*1e-6*25) is 400.00000000000006 in floating point: 12 significant digits give 400.0."""
    assert round_significant(1 / (100 * 1e-6 * 25)) == 400.0
    assert round_significant(0.1 + 0.2) == 0.3
    assert round_significant(123456789.123456789) == 123456789.123
    assert round_significant(5) == 5 and round_significant(0.0) == 0.0
