"""Tests of g5.1-e: the tools work for any x (integer, decimal, irregular, date, timestamp) and for any domain.

The fake database has obs(run_id, step, tt, d, ts, metric, region, value) and runs(run_id, alpha, L, seed):
6 runs with L = 20, 50, 80, 120, 200, 400 (one run each) and alpha 0.1, 0.1, 0.2, 0.2, 0.3 (with float noise), 0.3;
per run, metric a and b, steps 0..10 (tt = step / 4, d = 2026-01-01 + step days, ts = d + step hours).
region alternates n, s from one row to the next (the rows are written run by run, metric by metric, step by step).
value = 100 * (metric is b) + 10 * run_id + step, so the mean of metric a at one step is 35 + step.
"""
from __future__ import annotations

import datetime as dt
import inspect
import json

import duckdb
import pytest
from pydantic import ValidationError

from conftest import SETTINGS
from sim_analyst import tools
from sim_analyst.config import DataSpec, load_settings
from sim_analyst.data_access import DataSource
from sim_analyst.stats import sample_points, trend_summary
from sim_analyst.tool_schemas import ARGS_SCHEMAS, CheckQualityArgs
from sim_analyst.tools import Toolbox
from test_tools_g5 import make_toolbox

L_VALUES = [20, 50, 80, 120, 200, 400]
ALPHAS = [0.1, 0.1, 0.2, 0.2, 0.30000000000000004, 0.30000000000000004]
A = {"metric": "a"}


def fill(con) -> None:
    """Create and fill obs and runs."""
    con.execute("CREATE TABLE obs (run_id INTEGER, step INTEGER, tt DOUBLE, d DATE, ts TIMESTAMP, "
                "metric VARCHAR, region VARCHAR, value DOUBLE)")
    con.execute("CREATE TABLE runs (run_id INTEGER, alpha DOUBLE, L INTEGER, seed INTEGER)")
    counter = 0
    for run_id in range(1, 7):
        con.execute("INSERT INTO runs VALUES (?, ?, ?, ?)", [run_id, ALPHAS[run_id - 1], L_VALUES[run_id - 1], run_id])
        for metric in ("a", "b"):
            for step in range(11):
                day = dt.date(2026, 1, 1) + dt.timedelta(days=step)
                stamp = dt.datetime(day.year, day.month, day.day) + dt.timedelta(hours=step)
                value = float(100 * (metric == "b") + 10 * run_id + step)
                con.execute("INSERT INTO obs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            [run_id, step, step * 0.25, day, stamp, metric, "n" if counter % 2 == 0 else "s", value])
                counter += 1


@pytest.fixture(scope="session")
def gx_path(tmp_path_factory):
    """Path of the fake DuckDB file."""
    path = tmp_path_factory.mktemp("generic") / "generic.duckdb"
    con = duckdb.connect(str(path))
    fill(con)
    con.close()
    return path


@pytest.fixture()
def gx(gx_path):
    """A Toolbox on the fake database."""
    source = DataSource(DataSpec(database=gx_path))
    yield Toolbox(source, SETTINGS)
    source.close()


def notes_of(result: dict) -> list:
    """The notes of a result ([] if none), after checking there is no error."""
    assert "error" not in result, result
    return result.get("notes", [])


# ---------------------------------------------------------------- B1 sampling with irregular x

def test_irregular_x_keeps_every_point(gx):
    """6 values of x (20 ... 400), at most per_group: all kept, step null, no step note."""
    r = gx.describe_trends(value="value", x="L", filters={**A, "step": 10})
    assert "error" not in r, r
    group = r["groups"][0]
    assert group["series_sampled"]["rows"] == [[20, 20], [50, 30], [80, 40], [120, 50], [200, 60], [400, 70]]
    assert r["step"] is None
    assert not any("step" in n for n in r.get("notes", []))
    assert group["shape_sampled"] == "increasing"


def test_regular_x_with_many_values_is_sampled_as_before(tmp_path):
    """61 regular values: step 5, 13 points (0, 5 ... 60)."""
    tb, source = make_toolbox(tmp_path, ["CREATE TABLE m (t INTEGER, v DOUBLE)"],
                              [("INSERT INTO m VALUES (?, ?)", [[t, float(t)] for t in range(61)])])
    try:
        r = tb.describe_trends(value="v", x="t")
    finally:
        source.close()
    assert r["step"] == 5
    assert [row[0] for row in r["groups"][0]["series_sampled"]["rows"]] == list(range(0, 61, 5))


def test_sample_points_takes_the_first_point_of_each_interval():
    """per_group lets everything through; otherwise the first point of every non-empty interval and the last."""
    points = [(x, 1, float(x), float(x), float(x)) for x in L_VALUES]
    assert [p[0] for p in sample_points(points, 50, 20, per_group=6)] == L_VALUES
    assert [p[0] for p in sample_points(points, 50, 20, per_group=3)] == [20, 80, 120, 200, 400]
    regular = [(t, 1, 1.0, 1.0, 1.0) for t in range(11)]
    assert [p[0] for p in sample_points(regular, 3, 0)] == [0, 3, 6, 9, 10]
    assert [p[0] for p in sample_points(regular, 3, 0, per_group=40)] == list(range(11))


# ---------------------------------------------------------------- B2 shape on the numbers the model sees

def test_noise_of_1e_12_is_not_a_turning_point():
    """A constant series with noise 1e-12: constant, no peaks."""
    points = [(t, 3, 5.0 + (1e-12 if t % 2 else 0.0), 5.0, 5.0) for t in range(10)]
    s = trend_summary(points, 1)
    assert s["shape_sampled"] == "constant" and s["turning_points_sampled"] == []


# ---------------------------------------------------------------- B3 repeated rows in describe_trends

def test_describe_trends_notes_rows_that_repeat(gx):
    """x = L from runs: each point mixes the 11 steps of its run; the note names obs.step."""
    notes = notes_of(gx.describe_trends(value="value", x="L", filters=A))
    assert any("rows repeat" in n and "obs.step" in n and "x + at" not in n for n in notes), notes


def test_no_repetition_note_with_one_row_per_run_and_step(gx):
    """x = step, group_by run_id: one row per cell."""
    notes = notes_of(gx.describe_trends(value="value", x="step", group_by="run_id", filters=A))
    assert not any("rows repeat" in n for n in notes), notes


def test_date_trends_note_rows_that_repeat(gx):
    """The date branch has the note too (one year = the 11 days of each run)."""
    notes = notes_of(gx.describe_trends(value="value", x="d", period="year", filters=A))
    assert any("rows repeat" in n for n in notes), notes


# ---------------------------------------------------------------- B4 equality of decimal numbers

def test_decimal_filters_and_group_values_match_with_a_tolerance(gx):
    """0.3 matches 0.30000000000000004 in a filter, in compare_groups and in 'at'."""
    r = gx.summarize_results(value="value", group_by="alpha", filters={"alpha": 0.3, **A})
    assert "error" not in r, r
    assert r["rows"][0][0] == 0.3 and r["rows"][0][1] == 22
    c = gx.compare_groups(value="value", group_by="alpha", a=0.1, b=0.3, filters=A)
    assert "error" not in c, c
    assert c["a"]["n"] == 22 and c["b"]["n"] == 22
    s = gx.summarize_results(value="value", x="tt", at=0.75, filters=A)
    assert "error" not in s, s
    assert s["rows"][0][0] == 0.75 and s["rows"][0][1] == 6 and s["rows"][0][2] == 38
    s = gx.summarize_results(value="value", x="tt", at=0.7500000000001, filters=A)
    assert s["rows"][0][1] == 6 and not any("no rows at" in n for n in s.get("notes", []))
    c = gx.compare_groups(value="value", group_by="metric", a="a", b="b", x="tt", at=0.75)
    assert "error" not in c, c


def test_a_missing_decimal_is_still_reported(gx):
    """A value of x that is not in the data: the note says so."""
    s = gx.summarize_results(value="value", x="tt", at=[0.75, 0.8], filters=A)
    assert any("no rows at values of x [0.8]" in n for n in s["notes"]), s


# ---------------------------------------------------------------- B5 x is a date or a timestamp

def test_at_compares_whole_days_when_x_is_a_timestamp(gx):
    """at='2026-01-05' finds the row stamped 2026-01-05 04:00, with the note."""
    r = gx.summarize_results(value="value", x="ts", at="2026-01-05", filters=A)
    assert "error" not in r, r
    assert r["rows"][0][1] == 6 and r["rows"][0][2] == 39
    assert "at compares whole days of x." in r["notes"]
    c = gx.compare_groups(value="value", group_by="metric", a="a", b="b", x="ts", at="2026-01-05")
    assert "error" not in c and "at compares whole days of x." in c["notes"]


def test_no_whole_day_note_for_a_date(gx):
    """The note is only for TIMESTAMP."""
    r = gx.summarize_results(value="value", x="d", at="2026-01-05", filters=A)
    assert "error" not in r and "at compares whole days of x." not in r.get("notes", [])
    assert r["rows"][0][1] == 6


# ---------------------------------------------------------------- B6 numeric parameters in other tables

def test_numeric_parameter_of_a_joined_table_is_noted(gx):
    """alpha (3 values, in runs) varies inside the cell: note, not refusal. L and seed are unique per run."""
    r = gx.summarize_results(value="value", x="step", at=10, filters={**A, "region": "n"})
    notes = notes_of(r)
    assert any(n.startswith("aggregated over") and "alpha" in n for n in notes), notes
    assert not any("seed" in n or " L " in n for n in notes if n.startswith("aggregated over")), notes


def test_numeric_columns_of_the_main_table_are_measures(gx):
    """tt, value and step (main table) never enter the 'aggregated over' note."""
    notes = notes_of(gx.summarize_results(value="value", group_by="run_id", filters=A))
    joined = " ".join(n for n in notes if n.startswith("aggregated over"))
    assert "tt" not in joined and "step" not in joined


# ---------------------------------------------------------------- B7 measure name with a decimal x

def test_a_category_is_not_a_measure_name_when_x_has_decimals(tmp_path):
    """Three units x 8 values of tt; region per unit: a note, no error."""
    rows = [[u, 0.25 * k, region, float(u + k)] for u, region in ((1, "n"), (2, "s"), (3, "n")) for k in range(1, 9)]
    tb, source = make_toolbox(tmp_path, ["CREATE TABLE fl (unit INTEGER, tt DOUBLE, region VARCHAR, value DOUBLE)"],
                              [("INSERT INTO fl VALUES (?, ?, ?, ?)", rows)])
    try:
        r = tb.describe_trends(value="value", x="tt")
    finally:
        source.close()
    assert "error" not in r, r
    assert any("aggregated over region" in n for n in r["notes"]), r["notes"]


# ---------------------------------------------------------------- A2 periods

def test_hourly_trend_of_a_timestamp(gx):
    """period hour: x as YYYY-MM-DDTHH:00, slope per hour, the explanation says per hour."""
    r = gx.describe_trends(value="value", x="ts", period="hour", filters=A)
    assert "error" not in r, r
    group = r["groups"][0]
    assert r["period"] == "hour"
    assert group["series_sampled"]["rows"][0][0] == "2026-01-01T00:00"
    assert group["series_sampled"]["rows"][1][0] == "2026-01-02T01:00"
    slope = group["slope_last_part"]
    assert slope["slope_per_hour"] == pytest.approx(0.04) and "slope_per_day" not in slope
    assert slope["from_x"] == "2026-01-09T08:00"
    assert "per hour" in r["explanation"] and "per day" not in r["explanation"]


def test_hour_needs_a_timestamp(gx):
    """x a DATE with period hour: an 'argument' error."""
    r = gx.describe_trends(value="value", x="d", period="hour", filters=A)
    assert r["error_type"] == "argument" and "period hour needs a TIMESTAMP x; x is a DATE." in r["error"]
    r = gx.check_quality(table="obs", x="d", period="hour")
    assert r["error_type"] == "argument"


def test_year_period_and_automatic_choice(gx):
    """year: one point; automatic: the finest allowed (hour for a timestamp, day for a date)."""
    r = gx.describe_trends(value="value", x="d", period="year", filters=A)
    assert "error" not in r, r
    assert r["period"] == "year" and r["groups"][0]["series_sampled"]["rows"] == [["2026-01-01", 40]]
    assert gx.describe_trends(value="value", x="ts", filters=A)["period"] == "hour"
    assert gx.describe_trends(value="value", x="d", filters=A)["period"] == "day"
    assert tools.PERIODS == ("hour", "day", "week", "month", "year")


def test_period_help_lists_the_five_periods():
    """The text the model reads."""
    from sim_analyst.tool_schemas import PERIOD_HELP
    assert PERIOD_HELP == ("hour, day, week, month or year: the size of the bucket when x is a date or timestamp "
                           "column (hour only for a timestamp)")


# ---------------------------------------------------------------- A1 check_quality takes x

def test_check_quality_takes_x_not_time(gx):
    """x works (numeric, date, timestamp); time is refused by the schema and the method."""
    for x in ("step", "tt", "d", "ts"):
        assert "error" not in gx.check_quality(table="obs", x=x), x
    r = gx.check_quality(table="obs", x="step", period="day")
    assert "period ignored: x is numeric." in r["notes"]
    r = gx.check_quality(table="obs", x="metric")
    assert r["error_type"] == "column" and "x must be a numeric, DATE or TIMESTAMP column" in r["error"]
    with pytest.raises(ValidationError):
        CheckQualityArgs(table="obs", time="step")
    assert "x" in ARGS_SCHEMAS["check_quality"].model_fields and "time" not in ARGS_SCHEMAS["check_quality"].model_fields
    assert "time" not in inspect.signature(Toolbox.check_quality).parameters


# ---------------------------------------------------------------- A3 thresholds in settings.yaml

def test_last_fraction_comes_from_the_settings(gx_path):
    """analysis.trend_last_fraction 0.5: from_x and the text change."""
    results = {}
    for fraction in (0.2, 0.5):
        source = DataSource(DataSpec(database=gx_path))
        try:
            toolbox = Toolbox(source, {**SETTINGS, "analysis": {"trend_last_fraction": fraction}})
            results[fraction] = toolbox.describe_trends(value="value", x="step", group_by="run_id", filters=A)
        finally:
            source.close()
    assert results[0.2]["groups"][0]["slope_last_part"]["from_x"] == 8
    assert results[0.5]["groups"][0]["slope_last_part"]["from_x"] == 5
    assert "last 20%" in results[0.2]["explanation"] and "last 50%" in results[0.5]["explanation"]


def test_analysis_section_of_settings_yaml_has_todays_values():
    """The thresholds live in settings.yaml and the old constants of tools.py are gone."""
    analysis = load_settings()["analysis"]
    assert analysis == {"category_max_values": 20, "measure_name_share": 0.5, "link_share": 0.95, "number_share": 0.8,
                        "max_groups": 20, "max_at_values": 20, "max_items": 10, "max_listed_values": 20,
                        "trend_last_fraction": 0.2, "min_points_per_group": 6, "default_points": 12,
                        "x_tolerance": 1.0e-9, "grid_max_share": 0.5, "identifier_min_share": 0.95}
    assert analysis == tools.ANALYSIS_DEFAULTS
    for name in ("MAX_GROUPS", "MAX_ITEMS", "MAX_TIMES", "MAX_LISTED_VALUES", "CATEGORY_MAX_VALUES",
                 "MEASURE_NAME_SHARE", "LINK_SHARE", "NUMBER_SHARE", "IDENTIFIER_NAME"):
        assert not hasattr(tools, name), name


# ---------------------------------------------------------------- A5 identifiers found from the values

def test_identifier_duplicates_are_found_from_the_values(tmp_path):
    """code: unique but for 2 rows -> reported; idx: always unique -> no; kind: 3 repeated values -> no."""
    rows = [[i, f"c{10 if i == 11 else i}", "pqr"[i % 3], float(i)] for i in range(50)]
    tb, source = make_toolbox(tmp_path, ["CREATE TABLE ident (idx INTEGER, code VARCHAR, kind VARCHAR, value DOUBLE)"],
                              [("INSERT INTO ident VALUES (?, ?, ?, ?)", rows)])
    try:
        r = tb.check_quality(table="ident")
    finally:
        source.close()
    found = {d["column"]: d for d in r["identifier_duplicates"]}
    assert set(found) == {"code"}
    assert found["code"]["duplicated_values"] == 1 and found["code"]["extra_rows"] == 1
    assert found["code"]["examples"] == ["c10"]


# ---------------------------------------------------------------- A4 wording

def test_the_source_has_no_domain_example():
    """The example of a customer id is gone from the comments and docstrings of tools.py."""
    source = inspect.getsource(tools)
    assert "customer id" not in source and "a customer" not in source


# ---------------------------------------------------------------- C1 keep every point only if sampling would keep as many

def test_a_regular_series_of_31_values_is_still_sampled(tmp_path):
    """0..30 in one group: step 5 and 7 points, as before g5.1-e (not 31)."""
    tb, source = make_toolbox(tmp_path, ["CREATE TABLE m (t INTEGER, v DOUBLE)"],
                              [("INSERT INTO m VALUES (?, ?)", [[t, float(t)] for t in range(31)])])
    try:
        r = tb.describe_trends(value="v", x="t")
    finally:
        source.close()
    assert r["step"] == 5
    assert [row[0] for row in r["groups"][0]["series_sampled"]["rows"]] == [0, 5, 10, 15, 20, 25, 30]


def test_eleven_regular_values_are_all_kept(tmp_path):
    """0..10: step 1 would keep all 11 anyway, so step is null."""
    tb, source = make_toolbox(tmp_path, ["CREATE TABLE m (t INTEGER, v DOUBLE)"],
                              [("INSERT INTO m VALUES (?, ?)", [[t, float(t)] for t in range(11)])])
    try:
        r = tb.describe_trends(value="v", x="t")
    finally:
        source.close()
    assert r["step"] is None and len(r["groups"][0]["series_sampled"]["rows"]) == 11


# ---------------------------------------------------------------- C2 one row per day when x is a timestamp

def test_summarize_groups_a_timestamp_x_by_day(tmp_path):
    """Two instants (08:00, 17:00) of one day, 3 rows each: one row, n = 6; same without at."""
    rows = [[dt.datetime(2026, 1, 5, hour), float(i)] for hour in (8, 17) for i in range(3)]
    tb, source = make_toolbox(tmp_path, ["CREATE TABLE t (ts TIMESTAMP, v DOUBLE)"],
                              [("INSERT INTO t VALUES (?, ?)", rows)])
    try:
        with_at = tb.summarize_results(value="v", x="ts", at="2026-01-05")
        without = tb.summarize_results(value="v", x="ts")
    finally:
        source.close()
    for r in (with_at, without):
        assert "error" not in r, r
        assert len(r["rows"]) == 1 and r["rows"][0][0] == "2026-01-05" and r["rows"][0][1] == 6


# ---------------------------------------------------------------- C3 compare with a date or timestamp group_by

def test_compare_groups_by_a_timestamp_column_with_days(tmp_path):
    """a and b are days; the rows of any instant of the day are in the group."""
    rows = [[dt.datetime(2026, 1, 5 + day, hour), float(10 * day + i)]
            for day in (0, 1) for hour, i in ((8, 1), (12, 2), (17, 3))]
    tb, source = make_toolbox(tmp_path, ["CREATE TABLE t (ts TIMESTAMP, v DOUBLE)"],
                              [("INSERT INTO t VALUES (?, ?)", rows)])
    try:
        r = tb.compare_groups(value="v", group_by="ts", a="2026-01-05", b="2026-01-06")
        bad = tb.compare_groups(value="v", group_by="ts", a="not a day", b="2026-01-06")
    finally:
        source.close()
    assert "error" not in r, r
    assert r["a"]["n"] == 3 and r["b"]["n"] == 3
    assert r["a"]["mean"] == 2 and r["b"]["mean"] == 12 and r["a"]["group"] == "2026-01-05"
    assert bad["error_type"] == "argument"


# ---------------------------------------------------------------- matrix: the numbers, computed by hand
# metric a: value = 10 * run_id + step. At step 10 (6 runs): mean 45. Run 6 alone (L = 400): mean 65 over 11 steps.

X_AT = {"step": 10, "tt": 2.5, "L": 400, "d": "2026-01-11", "ts": "2026-01-11"}
SLOPE = {"step": ("slope_per_unit_of_x", 1), "tt": ("slope_per_unit_of_x", 4), "d": ("slope_per_day", 1),
         "ts": ("slope_per_hour", 0.04), "L": ("slope_per_unit_of_x", None)}


@pytest.mark.parametrize("x", list(X_AT))
def test_every_tool_gives_the_right_numbers_for_every_kind_of_x(gx, x):
    """summarize (with at), describe_trends, compare (with at) and check_quality: the expected values, no word 'time'."""
    at = X_AT[x]
    s = gx.summarize_results(value="value", x=x, at=at, filters=A)
    t = gx.describe_trends(value="value", x=x, filters=A)
    c = gx.compare_groups(value="value", group_by="metric", a="a", b="b", x=x, at=at)
    q = gx.check_quality(table="runs", x=x) if x == "L" else gx.check_quality(table="obs", x=x)
    for name, out in (("summarize_results", s), ("describe_trends", t), ("compare_groups", c), ("check_quality", q)):
        assert "error" not in out, (name, x, out)
        assert "time" not in json.dumps(out), (name, x)

    assert len(s["rows"]) == 1
    n, mean = s["rows"][0][1], s["rows"][0][2]
    if x == "L":
        assert (n, mean) == (11, 65)
        assert any("rows repeat" in note for note in s["notes"]), s["notes"]
    else:
        assert (n, mean) == (6, 45)

    group = t["groups"][0]
    key, expected = SLOPE[x]
    if x == "L":
        assert group["initial"]["mean"] == 15 and group["final"]["mean"] == 65
        assert group["shape_sampled"] == "increasing" and len(group["series_sampled"]["rows"]) == 6
    else:
        assert group["slope_last_part"][key] == pytest.approx(expected)

    assert c["b"]["mean"] - c["a"]["mean"] == pytest.approx(100)

    assert "identifier_duplicates" not in q and "duplicate_rows" not in q and "nulls" not in q
