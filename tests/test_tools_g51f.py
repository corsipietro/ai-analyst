"""Tests of g5.1-f: corrected variants (list positions with a dot, x/at inside filters, equals) and self-describing groups.

The fake database is the one of conftest.py. Pooling runs with different parameters is refused by the tools
(require_not_mixed), so the calls below keep experiment exp_a (or group by what varies) where the spec pools.
"""
from __future__ import annotations

import datetime as dt

import pytest

from test_tools_g5 import AMOUNT, make_toolbox

BASE = {"quantity": "x", "experiment": "exp_a"}
AMOUNT_Q = f"runs.{AMOUNT}"


def without_read_as(out: dict) -> dict:
    """The result without its `read_as` field."""
    return {k: v for k, v in out.items() if k != "read_as"}


# ---------------------------------------------------------------- 1. list positions written with a dot

def test_a_dotted_list_position_is_read_with_brackets(toolbox):
    """a.0.b gives the same result as a[0].b; the only difference is read_as, which is the first field."""
    dotted = toolbox.describe_trends(value="value", x="step", group_by="config.strategies.0.amount", filters=BASE)
    bracket = toolbox.describe_trends(value="value", x="step", group_by=AMOUNT, filters=BASE)
    assert "error" not in dotted, dotted
    assert list(dotted)[0] == "read_as" and "read_as" not in bracket
    assert dotted["read_as"] == ("'config.strategies.0.amount' read as 'config.strategies[0].amount': "
                                 "list positions go in brackets.")
    assert without_read_as(dotted) == bracket
    assert [g["group"][AMOUNT] for g in dotted["groups"]] == [10, 20]
    assert [g["final"]["mean"] for g in dotted["groups"]] == [100, 200]


def test_the_dotted_position_is_read_in_every_place_a_reference_is_written(toolbox):
    """value, x, unit, group_by and the keys of filters."""
    out = toolbox.summarize_results(value="value", group_by="quantity", x="step", at=[10], unit="run_id",
                                    filters={"experiment": "exp_a", "config.strategies.0.amount": 10})
    assert "error" not in out, out
    assert out["filters"] == {"runs.experiment": "exp_a", f"runs.{AMOUNT}": 10}
    assert "'config.strategies.0.amount' read as 'config.strategies[0].amount'" in out["read_as"]
    # a position written with the dot also at the end of the reference, and a decimal-looking text is not touched
    assert toolbox.parse("config.strategies.0", "runs").text == "config.strategies[0]"
    assert toolbox.parse("config.strategies[0].amount", "runs").text == AMOUNT


def test_no_change_no_read_as_and_garbage_is_still_refused(toolbox):
    """A reference that needed no correction adds nothing; a bad one keeps its error."""
    out = toolbox.summarize_results(value="value", group_by=AMOUNT, filters=BASE)
    assert "read_as" not in out
    bad = toolbox.summarize_results(value="value", group_by="config..amount", filters=BASE)
    assert bad["error_type"] == "column" and "not a column reference" in bad["error"]


def test_read_as_is_also_on_error_results(toolbox):
    """The first field of an error result is read_as, when something was corrected before the error."""
    out = toolbox.describe_trends(value="value", x="step", group_by="config.nothing.0.z", filters=BASE)
    assert "error" in out and list(out)[0] == "read_as" and "config.nothing[0].z" in out["read_as"]


def test_read_as_is_reset_at_every_call(toolbox):
    """A sentence of one call does not leak into the next."""
    first = toolbox.summarize_results(value="value", group_by="config.strategies.0.amount", filters=BASE)
    second = toolbox.summarize_results(value="value", group_by=AMOUNT, filters=BASE)
    assert "read_as" in first and "read_as" not in second


# ---------------------------------------------------------------- 2a. x and at inside filters

def test_x_and_at_inside_filters_become_the_arguments(toolbox):
    """compare_groups: same result as with the arguments, plus read_as; the filters echo has no x and at."""
    moved = toolbox.compare_groups(value="value", group_by=AMOUNT, a=10, b=20, unit="run_id",
                                   filters={**BASE, "x": "step", "at": 10})
    plain = toolbox.compare_groups(value="value", group_by=AMOUNT, a=10, b=20, unit="run_id", x="step", at=10,
                                   filters=BASE)
    assert "error" not in moved, moved
    assert moved["read_as"] == "'x' and 'at' are arguments, not filters: read as x='step', at=10."
    assert without_read_as(moved) == plain
    assert moved["filters"] == {"series.quantity": "x", "runs.experiment": "exp_a"}
    assert (moved["a"]["mean"], moved["a"]["n_units"], moved["b"]["mean"], moved["b"]["n_units"]) == (100, 3, 200, 3)


def test_x_and_at_inside_filters_summarize_a_list_of_values(toolbox):
    """summarize_results with at=[0, 10] in filters."""
    filters = {"experiment": "exp_a", AMOUNT: 10}
    moved = toolbox.summarize_results(value="value", group_by="quantity", filters={**filters, "x": "step", "at": [0, 10]})
    plain = toolbox.summarize_results(value="value", group_by="quantity", x="step", at=[0, 10], filters=filters)
    assert "error" not in moved, moved
    assert moved["read_as"] == "'x' and 'at' are arguments, not filters: read as x='step', at=[0, 10]."
    assert without_read_as(moved) == plain


def test_only_the_moved_keys_are_named(toolbox):
    """x in filters and at as an argument: the sentence names x only."""
    out = toolbox.summarize_results(value="value", at=[10], filters={**BASE, AMOUNT: 10, "x": "step"})
    assert out["read_as"] == "'x' is an argument, not a filter: read as x='step'."
    assert out["filters"] == {"series.quantity": "x", "runs.experiment": "exp_a", f"runs.{AMOUNT}": 10}


def test_the_same_value_is_compared_on_the_resolved_reference(toolbox):
    """x='series.step' in filters and x='step' as an argument are the same column."""
    out = toolbox.summarize_results(value="value", x="step", at=10, filters={**BASE, AMOUNT: 10, "x": "series.step"})
    assert "error" not in out and "x='series.step'" in out["read_as"]


def test_a_different_value_is_refused_with_a_clear_message(toolbox):
    """x or at given twice with different values."""
    out = toolbox.summarize_results(value="value", x="step", at=10, filters={**BASE, "x": "run_id"})
    assert out == {"error": "x is given twice: as an argument ('step') and inside filters ('run_id'). "
                            "Give it once, as an argument.", "error_type": "argument"}
    out = toolbox.summarize_results(value="value", x="step", at=10, filters={**BASE, "at": 9})
    assert out == {"error": "at is given twice: as an argument (10) and inside filters (9). "
                            "Give it once, as an argument.", "error_type": "argument"}


def test_describe_trends_has_no_at(toolbox):
    """at in the filters of describe_trends is refused and the message points to summarize_results."""
    out = toolbox.describe_trends(value="value", x="step", filters={**BASE, "at": 10})
    assert out == {"error": "describe_trends has no 'at': it describes the whole series along x. "
                            "For the values at one x use summarize_results with x and at.", "error_type": "argument"}
    out = toolbox.describe_trends(value="value", x="step", group_by=AMOUNT, filters={**BASE, "x": "step"})
    assert "error" not in out and out["read_as"] == "'x' is an argument, not a filter: read as x='step'."


def test_a_column_called_x_stays_a_filter(tmp_path):
    """If some table has a column x, filters={"x": 1} filters it, as before."""
    tb, source = make_toolbox(tmp_path, ["CREATE TABLE m (x INTEGER, v DOUBLE)"],
                              [("INSERT INTO m VALUES (?, ?)", [[1, 5.0], [2, 6.0], [3, 7.0]])])
    try:
        out = tb.summarize_results(value="v", filters={"x": 1})
    finally:
        source.close()
    assert "read_as" not in out and out["filters"] == {"m.x": 1}
    assert dict(zip(out["columns"], out["rows"][0]))["mean"] == 5


# ---------------------------------------------------------------- 2b. {"equals": v}

def test_equals_is_a_plain_value(toolbox):
    """{"equals": "x"} gives the result of "x", plus read_as; the echo shows the value read."""
    filters = {"experiment": "exp_a", AMOUNT: 10, "step": 10}
    equals = toolbox.summarize_results(value="value", filters={"quantity": {"equals": "x"}, **filters})
    plain = toolbox.summarize_results(value="value", filters={"quantity": "x", **filters})
    assert "error" not in equals, equals
    assert without_read_as(equals) == plain
    assert equals["read_as"] == '{"equals": "x"} read as the value "x".'
    assert equals["filters"]["series.quantity"] == "x"
    row = dict(zip(equals["columns"], equals["rows"][0]))
    assert (row["n"], row["mean"]) == (3, 100)


def test_equals_with_a_list_counts_as_a_list_for_the_warning(toolbox):
    """{"equals": ["x", "y"]} keeps two values: the mixed-categories warning applies."""
    out = toolbox.summarize_results(value="value", filters={"quantity": {"equals": ["x", "y"]},
                                                            "experiment": "exp_a", AMOUNT: 10})
    assert "error" not in out and "series.quantity" in out["warning"]
    assert out["filters"]["series.quantity"] == ["x", "y"]


def test_equals_with_a_single_value_labels_the_group(toolbox):
    """A one-value equals is a single-value filter for the label of describe_trends."""
    out = toolbox.describe_trends(value="value", x="step", group_by=AMOUNT,
                                  filters={"quantity": {"equals": "x"}, "experiment": "exp_a"})
    assert out["groups"][0]["group"]["series.quantity"] == "x"


def test_equals_cannot_be_combined(toolbox):
    """equals with another operator."""
    out = toolbox.summarize_results(value="value", filters={"quantity": {"equals": "x", "is_null": False}})
    assert out == {"error": "filter 'quantity': equals cannot be combined with other operators.",
                   "error_type": "argument"}


# ---------------------------------------------------------------- 3. self-describing groups

def test_groups_by_run_say_which_parameters_each_run_has(toolbox):
    """(c) the filtered column with two values and (d) the parameters of the run table that vary between the runs."""
    out = toolbox.describe_trends(value="value", x="step", group_by="run_id",
                                  filters={"quantity": "x", AMOUNT: [10, 20]})
    assert "error" not in out, out
    assert "warning" not in out and len(out["groups"]) == 8
    first, seventh = out["groups"][0]["group"], out["groups"][6]["group"]
    assert first == {"series.quantity": "x", "run_id": 1, AMOUNT_Q: 10, "runs.experiment": "exp_a",
                     "runs.config.model.rate": 0.5, "runs.seed": 101}
    assert list(first) == ["series.quantity", "run_id", AMOUNT_Q, "runs.experiment", "runs.config.model.rate",
                           "runs.seed"]
    assert seventh == {"series.quantity": "x", "run_id": 7, AMOUNT_Q: 10, "runs.experiment": "exp_b",
                       "runs.config.model.rate": 0.9, "runs.seed": 107}
    assert not any("strategies[0].type" in name for g in out["groups"] for name in g["group"])


def test_a_group_that_holds_several_values_lists_them_and_the_warning_appears(toolbox):
    """group_by experiment with amount [10, 20]: exp_a holds both, exp_b one; no parameter (experiment is not a key)."""
    out = toolbox.describe_trends(value="value", x="step", group_by="experiment",
                                  filters={"quantity": "x", AMOUNT: [10, 20]})
    assert "error" not in out, out
    by_name = {g["group"]["experiment"]: g["group"] for g in out["groups"]}
    assert by_name["exp_a"] == {"series.quantity": "x", "experiment": "exp_a", AMOUNT_Q: [10, 20]}
    assert by_name["exp_b"] == {"series.quantity": "x", "experiment": "exp_b", AMOUNT_Q: 10}
    assert "some groups hold more than one" in out["warning"] and AMOUNT_Q in out["warning"]
    assert out["warning"].startswith(f"filter {AMOUNT_Q} keeps 2 values (10, 20) and some groups hold more than one: "
                                     f"their numbers mix them (see {AMOUNT_Q} in each group). To see them apart, add "
                                     f"{AMOUNT_Q} to group_by.")


def test_operator_filters_label_the_groups_but_never_warn(toolbox):
    """{">=": 10} on the amount: exp_a [10, 20], exp_b 10 in a column of the table; no warning."""
    out = toolbox.summarize_results(value="value", group_by="experiment",
                                    filters={"quantity": "x", AMOUNT: {">=": 10}})
    assert "error" not in out, out
    assert "warning" not in out
    assert out["columns"][:3] == ["experiment", AMOUNT_Q, "n"]
    rows = {r[0]: r[1] for r in out["rows"]}
    assert rows == {"exp_a": [10, 20], "exp_b": 10}


def test_summarize_puts_the_parameters_after_the_group_by_columns(toolbox):
    """Columns: group_by, then (c) and (d) labels, then x and the statistics."""
    out = toolbox.summarize_results(value="value", x="step", at=10, group_by="run_id", filters={"quantity": "x"})
    assert "error" not in out, out
    assert out["columns"][:7] == ["run_id", "runs.experiment", "runs.config.model.rate", AMOUNT_Q, "runs.seed",
                                  "step", "n"]
    assert out["rows"][0][:7] == [1, "exp_a", 0.5, 10, 101, 10, 1]
    assert out["rows"][6][:7] == [7, "exp_b", 0.9, 10, 107, 10, 1]
    assert len(out["rows"]) == 8


def test_compare_groups_labels_the_two_groups_with_what_differs(toolbox):
    """a.labels and b.labels, right after `group`: only the parameters that differ between 1 and 4."""
    out = toolbox.compare_groups(value="value", group_by="run_id", a=1, b=4, x="step", at=10,
                                 filters={"quantity": "x"})
    assert "error" not in out, out
    assert out["a"]["labels"] == {AMOUNT_Q: 10, "runs.seed": 101}
    assert out["b"]["labels"] == {AMOUNT_Q: 20, "runs.seed": 104}
    assert list(out["a"])[:2] == ["group", "labels"]
    assert "warning" not in out


def test_compare_groups_warns_when_a_group_holds_two_values(toolbox):
    """Both experiments, amount [10, 20]: exp_a holds two amounts."""
    out = toolbox.compare_groups(value="value", group_by="experiment", a="exp_a", b="exp_b",
                                 filters={"quantity": "x", AMOUNT: [10, 20]})
    assert "error" not in out, out
    assert out["a"]["labels"] == {AMOUNT_Q: [10, 20]} and out["b"]["labels"] == {AMOUNT_Q: 10}
    assert "some groups hold more than one" in out["warning"]


def test_without_group_by_the_group_is_the_single_value_filters_only(toolbox):
    """No grouping: nothing to tell apart, so no extra label and the warning is the one of the filter."""
    out = toolbox.describe_trends(value="value", x="step",
                                  filters={"quantity": "x", "experiment": "exp_a", AMOUNT: 10})
    assert out["groups"][0]["group"] == {"series.quantity": "x", "runs.experiment": "exp_a", AMOUNT_Q: 10}
    out = toolbox.summarize_results(value="value", filters={"quantity": ["x", "y"], "experiment": "exp_a", AMOUNT: 10})
    assert "some groups hold more than one" in out["warning"]
    assert out["columns"][0] == "n"


def test_a_filter_on_the_grouped_column_is_not_repeated(toolbox):
    """The column in group_by is not a label too."""
    out = toolbox.describe_trends(value="value", x="step", group_by=AMOUNT,
                                  filters={"quantity": "x", "experiment": "exp_a", AMOUNT: [10, 20]})
    assert "error" not in out, out
    assert all(list(g["group"]) == ["series.quantity", "runs.experiment", AMOUNT] for g in out["groups"])
    assert "warning" not in out


@pytest.fixture()
def parameters_db(tmp_path):
    """s(uid, t, v) and u(uid, c0..c11): twelve parameters of the unit, all different between the three units."""
    columns = ", ".join(f"c{j} INTEGER" for j in range(12))
    units = [[i] + [i + j for j in range(12)] for i in range(3)]
    series = [[i, t, float(10 * i + t)] for i in range(3) for t in range(4)]
    tb, source = make_toolbox(
        tmp_path, [f"CREATE TABLE u (uid INTEGER, {columns})", "CREATE TABLE s (uid INTEGER, t INTEGER, v DOUBLE)"],
        [(f"INSERT INTO u VALUES (?{', ?' * 12})", units), ("INSERT INTO s VALUES (?, ?, ?)", series)])
    yield tb
    source.close()


def test_at_most_max_items_parameters_are_shown_and_the_rest_is_named(parameters_db):
    """Twelve varying parameters: ten in the group, the other two named in a note."""
    out = parameters_db.describe_trends(value="v", x="t", group_by="uid")
    assert "error" not in out, out
    group = out["groups"][1]["group"]
    assert list(group) == ["uid"] + [f"u.c{j}" for j in range(10)]
    assert group["u.c0"] == 1 and group["u.c9"] == 10
    assert "unit parameters cut to 10: u.c10, u.c11." in out["notes"]


@pytest.fixture()
def spread_db(tmp_path):
    """k(g, n, t, v): group 0 has n = 1..8, group 1 has n = 2 and 3, group 2 has n = 5."""
    rows = [[0, n, n, float(n)] for n in range(1, 9)] + [[1, 2, 2, 1.0], [1, 3, 3, 2.0], [2, 5, 5, 3.0]]
    tb, source = make_toolbox(tmp_path, ["CREATE TABLE k (g INTEGER, n INTEGER, t INTEGER, v DOUBLE)"],
                              [("INSERT INTO k VALUES (?, ?, ?, ?)", rows)])
    yield tb
    source.close()


def test_a_filtered_column_shows_one_value_a_short_list_or_a_range(spread_db):
    """1 value -> the value; 2 to 5 -> the sorted list; more -> {"between": [min, max]}."""
    out = spread_db.describe_trends(value="v", x="t", group_by="g", filters={"n": {">=": 1}})
    assert "error" not in out, out
    groups = {g["group"]["g"]: g["group"] for g in out["groups"]}
    assert groups[0]["k.n"] == {"between": [1, 8]}
    assert groups[1]["k.n"] == [2, 3]
    assert groups[2]["k.n"] == 5
    assert "warning" not in out


def test_dates_trends_describe_the_groups_too(tmp_path):
    """The date branch of describe_trends: a list filter, group 1 holds both values, group 2 one."""
    rows = [[dt.date(2026, 1, 1 + i), 1, m, float(i)] for i in range(3) for m in ("a", "b")]
    rows += [[dt.date(2026, 1, 1 + i), 2, "a", float(i)] for i in range(3)]
    tb, source = make_toolbox(tmp_path, ["CREATE TABLE t (d DATE, g INTEGER, m VARCHAR, v DOUBLE)"],
                              [("INSERT INTO t VALUES (?, ?, ?, ?)", rows)])
    try:
        out = tb.describe_trends(value="v", x="d", group_by="g", filters={"m": ["a", "b"]})
    finally:
        source.close()
    assert "error" not in out, out
    groups = {g["group"]["g"]: g["group"] for g in out["groups"]}
    assert groups[1]["t.m"] == ["a", "b"] and groups[2]["t.m"] == "a"
    assert "some groups hold more than one" in out["warning"]
