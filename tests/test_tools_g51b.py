"""Tests of g5.1-b: tool outputs repeat filters and grouping; floats keep 3 significant digits, data values 12."""
from __future__ import annotations

import json

import pytest

from sim_analyst.tools import deep_clean
from test_tools_g5 import AMOUNT, make_toolbox

CALLS = {
    "summarize_results": lambda tb, f: tb.summarize_results(value="value", group_by=AMOUNT, x="step", at=[10], filters=f),
    "describe_trends": lambda tb, f: tb.describe_trends(value="value", x="step", group_by=AMOUNT, filters=f),
    "compare_groups": lambda tb, f: tb.compare_groups(value="value", group_by=AMOUNT, a=10, b=20, x="step", at=10,
                                                      filters=f),
    "check_quality": lambda tb, f: tb.check_quality("series", group_by="quantity", filters=f),
}
PLAIN = ["CREATE TABLE s (grp VARCHAR, t INTEGER, v DOUBLE)"]
PLAIN_ROWS = [("INSERT INTO s VALUES (?, ?, ?)", [["p", 0, 1.0], ["p", 1, 2.0], ["p", 2, 3.0],
                                                   ["q", 0, 4.0], ["q", 1, 5.0], ["q", 2, 7.0]])]
PLAIN_CALLS = {
    "summarize_results": lambda tb: tb.summarize_results(value="v", group_by="grp"),
    "describe_trends": lambda tb: tb.describe_trends(value="v", x="t", group_by="grp"),
    "compare_groups": lambda tb: tb.compare_groups(value="v", group_by="grp", a="p", b="q"),
    "check_quality": lambda tb: tb.check_quality("s", group_by="grp"),
}


def filters_for(name: str, quantity: str) -> dict:
    """Filters that keep the fake database unambiguous (check_quality sees one table only)."""
    filters = {"series.quantity": quantity}
    if name != "check_quality":
        filters["runs.experiment"] = "exp_a"
    return filters


def test_deep_clean_uses_key_digits_for_keys_and_three_digits_for_the_rest():
    """group, filters, at, x, from_x, to_x and leading `x` columns keep 12 digits; the rest 3."""
    assert deep_clean({"group": {"g": 12.3456}, "mean": 12.3456}) == {"group": {"g": 12.3456}, "mean": 12.3}
    table = {"columns": ["x", "n", "mean"], "rows": [[0.125, 3, 977.969]]}
    assert deep_clean(table)["rows"] == [[0.125, 3, 978]]
    assert deep_clean({"at": 0.12345, "x": 0.12345}) == {"at": 0.12345, "x": 0.12345}
    assert deep_clean({"a": 12.3456}, digits=12) == {"a": 12.3456}


@pytest.mark.parametrize("name", list(CALLS))
def test_two_calls_differing_only_in_filters_have_different_outputs(toolbox, name):
    """The output echoes the filters, right after the first field."""
    x = CALLS[name](toolbox, filters_for(name, "x"))
    y = CALLS[name](toolbox, filters_for(name, "y"))
    assert x["filters"] == filters_for(name, "x")
    assert y["filters"] == filters_for(name, "y")
    assert x != y
    assert list(x).index("filters") == 1


@pytest.mark.parametrize("name", list(PLAIN_CALLS))
def test_no_filters_gives_an_empty_object_and_group_by_is_resolved(tmp_path, name):
    """`filters` is always present; `group_by` lists the resolved names (a string in compare_groups)."""
    tb, source = make_toolbox(tmp_path, PLAIN, PLAIN_ROWS)
    try:
        out = PLAIN_CALLS[name](tb)
    finally:
        source.close()
    assert out["filters"] == {}
    assert out["group_by"] == ("s.grp" if name == "compare_groups" else ["s.grp"])


def test_group_by_is_an_empty_list_without_grouping(tmp_path):
    """No group_by -> []."""
    tb, source = make_toolbox(tmp_path, PLAIN, PLAIN_ROWS)
    try:
        assert tb.summarize_results(value="v", x="t", at=[0], filters={"grp": "p"})["group_by"] == []
        assert tb.describe_trends(value="v", x="t", filters={"grp": "p"})["group_by"] == []
        assert tb.check_quality("s")["group_by"] == []
    finally:
        source.close()


def test_a_short_filter_name_is_resolved_in_the_output(toolbox):
    """{"quantity": "x"} is shown as {"series.quantity": "x"}."""
    out = toolbox.summarize_results(value="value", group_by=AMOUNT, x="step", at=[10],
                                    filters={"quantity": "x", "experiment": "exp_a"})
    assert out["filters"] == {"series.quantity": "x", "runs.experiment": "exp_a"}


def test_small_numbers_and_float_groups_survive_cleaning(tmp_path):
    """1e-06 in a JSON column and a float group 0.12345 are not rounded away; the mean has 3 digits."""
    doc = json.dumps({"coeff": 1e-06})
    tb, source = make_toolbox(
        tmp_path,
        ["CREATE TABLE t (id INTEGER, doc VARCHAR, g DOUBLE, v DOUBLE)"],
        rows=[("INSERT INTO t VALUES (?, ?, ?, ?)", [[1, doc, 0.12345, 1.0], [2, doc, 0.12345, 2.2346]])])
    try:
        assert "1e-06" in str(tb.describe_database()["json_paths"])
        out = tb.summarize_results(value="v", group_by="g")
        assert out["rows"][0][0] == 0.12345
        assert out["rows"][0][2] == 1.62
    finally:
        source.close()
