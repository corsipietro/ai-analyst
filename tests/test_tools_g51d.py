"""Tests of g5.1-d: the independent variable is called x (not time) in summarize_results, describe_trends, compare_groups."""
from __future__ import annotations

import datetime as dt
import inspect
import json

import pytest

from bench import replay_tool_calls
from sim_analyst.tool_schemas import ARGS_SCHEMAS, make_langchain_tools
from sim_analyst.tools import Toolbox
from test_tools_g5 import AMOUNT, make_toolbox

TOOLS = ("summarize_results", "describe_trends", "compare_groups")
EXP_A = {"series.quantity": "x", "experiment": "exp_a"}


def test_the_three_tools_and_their_schemas_take_x_not_time():
    """Argument x in the functions and in the Pydantic models; check_quality takes x too (g5.1-e)."""
    for name in TOOLS:
        assert "x" in inspect.signature(getattr(Toolbox, name)).parameters
        assert "time" not in inspect.signature(getattr(Toolbox, name)).parameters
        assert "x" in ARGS_SCHEMAS[name].model_fields and "time" not in ARGS_SCHEMAS[name].model_fields
    assert "x" in ARGS_SCHEMAS["check_quality"].model_fields and "time" not in ARGS_SCHEMAS["check_quality"].model_fields
    assert "x" in inspect.signature(Toolbox.check_quality).parameters


def test_outputs_never_say_time(toolbox, tmp_path):
    """No '"time"' key or value in the JSON of the three tools (numeric x), nor in a date trend."""
    outputs = [
        toolbox.summarize_results(value="value", group_by=AMOUNT, x="step", at=[10], filters=EXP_A),
        toolbox.describe_trends(value="value", x="step", group_by=AMOUNT, filters=EXP_A),
        toolbox.compare_groups(value="value", group_by=AMOUNT, a=10, b=20, x="step", at=10, filters=EXP_A),
    ]
    tb, source = make_toolbox(tmp_path, ["CREATE TABLE t (d DATE, v DOUBLE)"],
                              [("INSERT INTO t VALUES (?, ?)", [[dt.date(2026, 1, 1 + i), float(i)] for i in range(9)])])
    try:
        outputs.append(tb.describe_trends(value="v", x="d"))
    finally:
        source.close()
    for out in outputs:
        assert "error" not in out, out
        assert '"time"' not in json.dumps(out)
        assert "time" not in json.dumps(out).replace("timestamp", "")


def test_trend_fields_are_named_x(toolbox):
    """x in the head, in initial/final/max/min, in the slope and in the sampled series."""
    r = toolbox.describe_trends(value="value", x="step", group_by=AMOUNT, filters=EXP_A)
    g = r["groups"][0]
    assert r["x"] == "step" and g["initial"]["x"] == 0 and g["final"]["x"] == 10
    assert "x" in g["max_of_mean"] and "x" in g["min_of_mean"]
    assert {"from_x", "to_x", "slope_per_unit_of_x"} <= set(g["slope_last_part"])
    assert g["series_sampled"]["columns"] == ["x", "mean"]


def test_date_x_keeps_slope_per_day(tmp_path):
    """With a date x the slope is per day and the values of x are ISO dates."""
    tb, source = make_toolbox(tmp_path, ["CREATE TABLE t (d DATE, v DOUBLE)"],
                              [("INSERT INTO t VALUES (?, ?)", [[dt.date(2026, 1, 1 + i), float(i)] for i in range(9)])])
    try:
        g = tb.describe_trends(value="v", x="d")["groups"][0]
    finally:
        source.close()
    assert "slope_per_day" in g["slope_last_part"] and g["initial"]["x"] == "2026-01-01"
    assert {"from_x", "to_x"} <= set(g["slope_last_part"])


def test_large_x_values_are_not_rounded(tmp_path):
    """x = 123456, 123457, ... stays exact in describe_trends and summarize_results."""
    rows = [[123456.0 + i, float(i % 7)] for i in range(61)]
    tb, source = make_toolbox(tmp_path, ["CREATE TABLE r (pos DOUBLE, y DOUBLE)"], [("INSERT INTO r VALUES (?, ?)", rows)])
    try:
        g = tb.describe_trends(value="y", x="pos")["groups"][0]
        s = tb.summarize_results(value="y", x="pos", at=[123470.0])
    finally:
        source.close()
    assert g["initial"]["x"] == 123456 and g["final"]["x"] == 123516
    assert g["slope_last_part"]["from_x"] == int(g["slope_last_part"]["from_x"]) > 123000
    assert g["slope_last_part"]["to_x"] == 123516
    assert all(row[0] == int(row[0]) and row[0] > 123000 for row in g["series_sampled"]["rows"])
    assert g["series_sampled"]["rows"][0][0] == 123456
    assert s["rows"][0][0] == 123470 and s["columns"][0] == "pos"


def test_schemas_have_no_time_property_and_x_is_required(toolbox):
    """JSON schemas of the three tools: no `time`; x required in describe_trends."""
    tools = {t.name: t for t in make_langchain_tools(toolbox)}
    for name in TOOLS:
        schema = tools[name].args_schema.model_json_schema()
        assert "time" not in schema["properties"] and "x" in schema["properties"]
    assert "x" in tools["describe_trends"].args_schema.model_json_schema()["required"]


def test_messages_call_the_column_x(toolbox):
    """The notes and errors of the three tools say x."""
    r = toolbox.summarize_results(value="value", group_by=AMOUNT, x="step", filters=EXP_A)
    assert "last value of x" in r["notes"][0]
    assert "'x'" in toolbox.summarize_results(value="value", at=[1])["error"]
    r = toolbox.compare_groups(value="value", group_by=AMOUNT, a=10, b=20, x="step", filters=EXP_A)
    assert any("last value of x" in n for n in r["notes"])
    r = toolbox.compare_groups(value="value", group_by=AMOUNT, a=10, b=20, x="step", at=[1, 2], filters=EXP_A)
    assert "ONE value of x" in r["error"]
    r = toolbox.summarize_results(value="value", group_by=AMOUNT, x="step", at=[10, 99], filters=EXP_A)
    assert any("no rows at values of x [99]" in n for n in r["notes"])
    assert "x 'quantity'" in toolbox.describe_trends(value="value", x="quantity", filters=EXP_A)["error"]
    assert "x + at" in Toolbox.repetition_note((["k"], 3, ["t.x"]), with_x=True)


def test_replay_renames_time_to_x(toolbox):
    """A call recorded before g5.1-d, with `time`, still runs."""
    tools = {t.name: t for t in make_langchain_tools(toolbox)}
    args = {"value": "value", "time": "step", "group_by": AMOUNT, "filters": EXP_A}
    assert "error" not in replay_tool_calls.run_call(tools, {"tool": "describe_trends", "args": args})
    assert "time" in args    # the recorded call is not changed
    summary = {"value": "value", "time": "step", "at": [10], "group_by": AMOUNT, "filters": EXP_A}
    assert "error" not in replay_tool_calls.run_call(tools, {"tool": "summarize_results", "args": summary})
