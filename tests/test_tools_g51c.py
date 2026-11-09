"""Tests of g5.1-c: metric label in the groups, warning on mixed categories, shorter sampled series."""
from __future__ import annotations

import pytest

from sim_analyst.stats import trend_summary
from test_tools_g5 import AMOUNT, make_toolbox

EXP_A = {"series.quantity": "x", "experiment": "exp_a"}


def after(out: dict, key: str) -> str:
    """The name of the field that follows `key` in an output."""
    names = list(out)
    return names[names.index(key) + 1]


def test_single_value_filters_label_every_group(toolbox):
    """A scalar or one-element list filter enters the head of `group`; operators do not."""
    for quantity in ("x", ["x"]):
        r = toolbox.describe_trends(value="value", x="step", group_by=AMOUNT, filters={**EXP_A, "series.quantity": quantity})
        for g in r["groups"]:
            assert list(g["group"])[:2] == ["series.quantity", "runs.experiment"]
            assert g["group"]["series.quantity"] == "x" and AMOUNT in g["group"]
    r = toolbox.describe_trends(value="value", x="step", group_by=AMOUNT,
                                filters={**EXP_A, "step": {"between": [0, 5]}})
    assert "series.step" not in r["groups"][0]["group"] and "step" not in r["groups"][0]["group"]


def test_group_holds_only_the_filters_without_group_by(toolbox):
    """Without group_by `group` is the single-value filters."""
    r = toolbox.describe_trends(value="value", x="step", filters={**EXP_A, AMOUNT: 10})
    assert r["groups"][0]["group"] == {"series.quantity": "x", "runs.experiment": "exp_a", f"runs.{AMOUNT}": 10}


MIXED = {"series.quantity": ["x", "y"], "experiment": "exp_a", AMOUNT: 10}
CALLS = {
    "summarize_results": lambda tb, f, g: tb.summarize_results(value="value", group_by=g, x="step", at=[10], filters=f),
    "describe_trends": lambda tb, f, g: tb.describe_trends(value="value", x="step", group_by=g, filters=f),
    "compare_groups": lambda tb, f, g: tb.compare_groups(value="value", group_by=g or "step", a=9, b=10, filters=f),
}


@pytest.mark.parametrize("name", list(CALLS))
def test_mixed_categories_give_a_warning_right_after_group_by(toolbox, name):
    """Two values of a filter column that is not in group_by: warning, same numbers as before."""
    r = CALLS[name](toolbox, MIXED, None)
    assert "error" not in r, r
    w = r["warning"]
    assert all(s in w for s in ("series.quantity", "x", "y", "group_by", "some groups hold more than one"))
    assert after(r, "group_by") == "warning"
    if name == "summarize_results":
        row = dict(zip(r["columns"], r["rows"][0]))
        assert (row["n"], row["mean"], row["min"], row["max"]) == (6, 50.5, -99, 101)
    elif name == "describe_trends":
        assert r["groups"][0]["final"] == {"x": 10, "mean": 50.5, "min": -99, "max": 101, "spread": 200}
    else:
        assert (r["a"]["mean"], r["b"]["mean"]) == (45.5, 50.5)


@pytest.mark.parametrize("name", list(CALLS))
def test_no_warning_when_separated_or_single_value(toolbox, name):
    """quantity in group_by, or a single value: no warning."""
    if name == "compare_groups":
        separated = toolbox.compare_groups(value="value", group_by="series.quantity", a="x", b="y",
                                           filters={"series.quantity": ["x", "y"], "experiment": "exp_a", AMOUNT: 10})
    else:
        separated = CALLS[name](toolbox, MIXED, "series.quantity")
    assert "error" not in separated and "warning" not in separated
    single = CALLS[name](toolbox, {**MIXED, "series.quantity": ["x"]}, None)
    assert "error" not in single and "warning" not in single


def test_warning_on_a_date_trend(tmp_path):
    """date_trends warns too."""
    import datetime as dt
    rows = [[dt.date(2026, 1, 1 + i), m, float(i)] for i in range(5) for m in ("a", "b")]
    tb, source = make_toolbox(tmp_path, ["CREATE TABLE t (d DATE, m VARCHAR, v DOUBLE)"],
                              [("INSERT INTO t VALUES (?, ?, ?)", rows)])
    try:
        r = tb.describe_trends(value="v", x="d", filters={"m": ["a", "b"]})
    finally:
        source.close()
    assert "error" not in r and after(r, "group_by") == "warning" and "t.m" in r["warning"]


def test_series_sampled_has_x_and_mean_only(toolbox, tmp_path):
    """Two columns per row, for numeric and date times."""
    r = toolbox.describe_trends(value="value", x="step", group_by=AMOUNT, filters=EXP_A)
    s = r["groups"][0]["series_sampled"]
    assert s["columns"] == ["x", "mean"] and all(len(row) == 2 for row in s["rows"])
    import datetime as dt
    tb, source = make_toolbox(tmp_path, ["CREATE TABLE t (d DATE, v DOUBLE)"],
                              [("INSERT INTO t VALUES (?, ?)", [[dt.date(2026, 1, 1 + i), float(i)] for i in range(9)])])
    try:
        s = tb.describe_trends(value="v", x="d")["groups"][0]["series_sampled"]
    finally:
        source.close()
    assert s["columns"] == ["x", "mean"] and all(len(row) == 2 for row in s["rows"])


def many_groups(tmp_path, n_groups):
    """A table with n_groups groups, each with 61 times."""
    rows = [[g, t, float(g * t)] for g in range(n_groups) for t in range(61)]
    return make_toolbox(tmp_path, ["CREATE TABLE m (g INTEGER, t INTEGER, v DOUBLE)"],
                        [("INSERT INTO m VALUES (?, ?, ?)", rows)])


def test_points_per_group_shrink_with_the_number_of_groups(tmp_path):
    """7 groups: at most 80 // 7 = 11 points and at least 6; 1 group: at most max_points."""
    tb, source = many_groups(tmp_path, 7)
    try:
        r = tb.describe_trends(value="v", x="t", group_by="g")
        lengths = [len(g["series_sampled"]["rows"]) for g in r["groups"]]
    finally:
        source.close()
    assert len(lengths) == 7 and all(6 <= n <= 11 for n in lengths)
    tb, source = many_groups(tmp_path, 1)
    try:
        r = tb.describe_trends(value="v", x="t", step=1)
        n = len(r["groups"][0]["series_sampled"]["rows"])
    finally:
        source.close()
    assert 40 >= n > 11


def test_trend_summary_keeps_its_statistics():
    """initial, final, max/min of the mean and slope are the same as before (computed by hand)."""
    points = [(t, 3, float(t), float(t) - 1, float(t) + 1) for t in range(11)]
    s = trend_summary(points, 5)
    assert s["initial"] == {"x": 0, "mean": 0.0}
    assert s["final"] == {"x": 10, "mean": 10.0, "min": 9.0, "max": 11.0, "spread": 2.0}
    assert s["max_of_mean"] == {"x": 10, "mean": 10.0} and s["min_of_mean"] == {"x": 0, "mean": 0.0}
    assert s["slope_last_part"]["slope_per_unit_of_x"] == pytest.approx(1.0)
    assert s["series_sampled"] == {"columns": ["x", "mean"], "rows": [[0, 0.0], [5, 5.0], [10, 10.0]]}


def test_explanation_is_generic_and_says_where_min_and_max_are(toolbox, tmp_path):
    """Both branches: min and max only in final; numeric branch names the x column and never says time."""
    import datetime as dt
    import re
    r = toolbox.describe_trends(value="value", x="step", group_by=AMOUNT, filters=EXP_A)
    text = r["explanation"]
    assert "only in final" in text and "mean, min and max" not in text
    assert "step" in text and not re.search(r"\btime\b", text)
    tb, source = make_toolbox(tmp_path, ["CREATE TABLE t (d DATE, v DOUBLE)"],
                              [("INSERT INTO t VALUES (?, ?)", [[dt.date(2026, 1, 1 + i), float(i)] for i in range(9)])])
    try:
        text = tb.describe_trends(value="v", x="d")["explanation"]
    finally:
        source.close()
    assert "only in final" in text and "mean, min and max" not in text
