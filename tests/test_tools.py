"""Tests of the generic tools on the fake database of conftest.py (known answers)."""
from __future__ import annotations

import json

import duckdb
import pytest

from conftest import SETTINGS, fill
from sim_analyst.config import DataSpec, sha256_of
from sim_analyst.data_access import DataSource
from sim_analyst.tools import Toolbox

AMOUNT = "config.strategies[0].amount"
EXP_A = {"quantity": "x", "experiment": "exp_a"}


# ---------------------------------------------------------------- describe_database

def test_describe_lists_tables_values_and_json_paths(toolbox):
    """Row counts, distinct values of short text columns, JSON paths with their values, shared columns."""
    d = toolbox.describe_database()
    assert d["tables"]["runs"]["rows"] == 8
    assert d["tables"]["series"]["rows"] == 8 * 11 * 2
    assert d["tables"]["series"]["columns"]["quantity"]["values"] == ["x", "y"]
    assert d["tables"]["series"]["columns"]["step"]["min"] == 0
    assert d["tables"]["series"]["columns"]["step"]["max"] == 10
    assert d["tables"]["runs"]["columns"]["config"]["json"] is True
    assert d["json_paths"][AMOUNT]["values"] == [10, 20]
    assert d["json_paths"]["config.model.rate"]["values"] == [0.5, 0.9]
    assert d["shared_columns"] == {"runs & series": ["run_id"]}


# ---------------------------------------------------------------- query_results

def test_query_refuses_anything_but_one_select(toolbox):
    """Writes and multiple statements are refused before reaching the database."""
    assert toolbox.query_results("DELETE FROM runs")["error_type"] == "sql"
    assert toolbox.query_results("SELECT 1; SELECT 2")["error_type"] == "sql"


def test_query_cannot_read_other_files(toolbox):
    """External access is off: a query cannot read files of the computer."""
    result = toolbox.query_results("SELECT * FROM read_csv('/etc/hosts')")
    assert result["error_type"] == "sql"


def test_query_warns_when_cut(db_path):
    """A result longer than max_rows is cut and says so."""
    settings = {**SETTINGS, "limits": {**SETTINGS["limits"], "max_rows": 5}}
    source = DataSource(DataSpec(database=db_path))
    result = Toolbox(source, settings).query_results("SELECT * FROM series")
    source.close()
    assert result["truncated"] is True and len(result["rows"]) == 5 and "CUT" in result["warning"]


def test_tools_leave_the_database_unchanged(db_path, toolbox):
    """The database file has the same fingerprint before and after using every tool."""
    before = sha256_of(db_path)
    toolbox.describe_database()
    toolbox.summarize_results("value", AMOUNT, "step", [10], EXP_A)
    toolbox.describe_trends("value", "step", AMOUNT, EXP_A)
    toolbox.compare_groups("value", AMOUNT, 10, 20, 10, "step", EXP_A)
    toolbox.check_quality("series")
    assert sha256_of(db_path) == before


# ---------------------------------------------------------------- summarize_results

def test_summarize_known_means(toolbox):
    """Mean, min, max and n per group at the requested time, with a JSON path from another table."""
    r = toolbox.summarize_results(value="value", group_by=AMOUNT, x="step", at=[10], filters=EXP_A)
    assert r["columns"] == [AMOUNT, "step", "n", "mean", "std", "min", "max", "n_rows", "n_null", "sum"]
    assert r["rows"] == [[10, 10, 3, 100, 1, 99, 101, 3, 0, 300], [20, 10, 3, 200, 1, 199, 201, 3, 0, 600]]
    assert "warning" not in r


def test_summarize_defaults_to_last_time(toolbox):
    """Without 'at', the last time is used and a note says so."""
    r = toolbox.summarize_results(value="value", group_by=AMOUNT, x="step", filters=EXP_A)
    assert [row[1] for row in r["rows"]] == [10, 10]
    assert "last value of x" in r["notes"][0]


def test_summarize_refuses_mixing_quantities(toolbox):
    """Without filtering the quantity, x and y would be averaged together: error, no numbers."""
    r = toolbox.summarize_results(value="value", group_by=AMOUNT, x="step", at=[10],
                                  filters={"experiment": "exp_a"})
    assert r["error_type"] == "column" and "rows" not in r
    assert "series.quantity (values ['x', 'y'])" in r["error"]


def test_summarize_refuses_mixing_parameters(toolbox):
    """Grouping by amount only, runs of exp_b (another rate) fall in the group amount = 10: error."""
    r = toolbox.summarize_results(value="value", group_by=AMOUNT, x="step", at=[10], filters={"quantity": "x"})
    assert r["error_type"] == "column" and "runs.config (JSON" in r["error"]


def test_trends_and_compare_refuse_mixing(toolbox):
    """The same refusal in describe_trends and compare_groups."""
    r = toolbox.describe_trends(value="value", x="step", group_by=AMOUNT, filters={"experiment": "exp_a"})
    assert r["error_type"] == "column" and "series.quantity" in r["error"]
    r = toolbox.compare_groups(value="value", group_by=AMOUNT, a=10, b=20, x="step", at=10,
                               filters={"experiment": "exp_a"})
    assert r["error_type"] == "column" and "series.quantity" in r["error"]


def test_unknown_column_and_wrong_json_path_are_column_errors(toolbox):
    """Wrong names give a 'column' error that lists what exists."""
    r = toolbox.summarize_results(value="valu", x="step")
    assert r["error_type"] == "column" and "series: run_id, step, quantity, value" in r["error"]
    r = toolbox.summarize_results(value="value", group_by="config.strategies[0].amnt", x="step", filters=EXP_A)
    assert r["error_type"] == "column" and AMOUNT in r["error"]


def test_filter_on_missing_value_lists_existing_values(toolbox):
    """A filter that selects nothing gives a 'value' error with the values that exist."""
    r = toolbox.summarize_results(value="value", x="step", filters={"quantity": "z"})
    assert r["error_type"] == "value" and "['x', 'y']" in r["error"]


# ---------------------------------------------------------------- describe_trends

def test_trends_of_a_linear_series(toolbox):
    """x = amount * step: initial 0, final 10 * amount, slope = amount, increasing, no turning point."""
    r = toolbox.describe_trends(value="value", x="step", group_by=AMOUNT, filters=EXP_A)
    assert r["step"] is None and "warning" not in r      # 11 values of x: all kept, no step
    first = r["groups"][0]
    assert first["group"] == {"series.quantity": "x", "runs.experiment": "exp_a", AMOUNT: 10}
    assert first["initial"] == {"x": 0, "mean": 0}
    assert first["final"] == {"x": 10, "mean": 100, "min": 99, "max": 101, "spread": 2}
    assert first["slope_last_part"]["slope_per_unit_of_x"] == pytest.approx(10)
    assert first["shape_sampled"] == "increasing" and first["turning_points_sampled"] == []
    assert first["observations"] == [3]
    assert len(first["series_sampled"]["rows"]) == 11


def test_trends_step_and_range(toolbox):
    """step and start/end select the sampled points; the last point is always kept."""
    r = toolbox.describe_trends(value="value", x="step", group_by=AMOUNT, filters=EXP_A, step=4, start=0, end=10)
    times = [row[0] for row in r["groups"][0]["series_sampled"]["rows"]]
    assert times == [0, 4, 8, 10]


# ---------------------------------------------------------------- compare_groups

def test_compare_clear_difference(toolbox):
    """x at step 10: 200 - 100 = 100, ranges far apart: established."""
    r = toolbox.compare_groups(value="value", group_by=AMOUNT, a=10, b=20, x="step", at=10, filters=EXP_A)
    assert r["difference_b_minus_a"] == 100
    assert r["verdict"] == "established" and r["ranges_overlap"] is False
    assert r["a"]["n"] == 3 and "notes" not in r


def test_compare_noise_is_not_established(toolbox):
    """y: groups differ by 1, repetitions by 100: not established, ranges overlap."""
    r = toolbox.compare_groups(value="value", group_by=AMOUNT, a=10, b=20, x="step", at=10,
                               filters={"quantity": "y", "experiment": "exp_a"})
    assert r["difference_b_minus_a"] == 1
    assert r["verdict"] == "not established" and r["ranges_overlap"] is True


def test_compare_uses_welch_and_no_bootstrap_with_few_replicates(toolbox):
    """3 replicates per group: the verdict comes from Welch's interval; no bootstrap is reported."""
    r = toolbox.compare_groups(value="value", group_by=AMOUNT, a=10, b=20, x="step", at=10, filters=EXP_A)
    assert r["method"] == "welch"
    assert r["degrees_of_freedom"] == pytest.approx(4)
    assert "bootstrap_interval" not in r
    assert "normal" in r["assumption"]
    assert r["interval"][0] < 100 < r["interval"][1]


def test_compare_with_one_value_in_a_group_has_no_verdict(toolbox):
    """exp_b has a single run per (rate) value here: one value in a group -> no verdict, with the reason."""
    r = toolbox.compare_groups(value="value", group_by="runs.run_id", a=1, b=2, x="step", at=10,
                               filters={"quantity": "x"})
    assert "verdict" in r and r["verdict"].startswith("cannot be assessed")
    assert "observations" in r["verdict"] and "replicate" not in r["verdict"]
    assert "interval" not in r


def test_compare_adds_bootstrap_from_min_n(tmp_path):
    """With at least bootstrap.min_n values per group the bootstrap is reported too, but the verdict is Welch's."""
    path = tmp_path / "big.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE obs (grp VARCHAR, value DOUBLE)")
    for i in range(30):
        con.execute("INSERT INTO obs VALUES ('a', ?), ('b', ?)", [float(i % 5), float(i % 5) + 10])
    con.close()
    source = DataSource(DataSpec(database=path))
    r = Toolbox(source, SETTINGS).compare_groups(value="value", group_by="grp", a="a", b="b")
    source.close()
    assert r["method"] == "welch" and r["verdict"] == "established"
    low, high = r["bootstrap_interval"]
    assert low < 10 < high


def test_compare_without_variability_has_no_interval(tmp_path):
    """Identical values in both groups: no interval can be computed, said explicitly, no verdict invented."""
    path = tmp_path / "flat.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE obs (grp VARCHAR, value DOUBLE)")
    con.execute("INSERT INTO obs VALUES ('a', 1), ('a', 1), ('b', 2), ('b', 2)")
    con.close()
    source = DataSource(DataSpec(database=path))
    r = Toolbox(source, SETTINGS).compare_groups(value="value", group_by="grp", a="a", b="b")
    source.close()
    assert "interval" not in r and "no variability" in r["verdict"]


def test_compare_on_pooled_times_gives_a_note_and_no_verdict(toolbox):
    """Without time + at, the 11 steps of each run fall in the same group: numbers, a note naming run_id, no verdict."""
    r = toolbox.compare_groups(value="value", group_by=AMOUNT, a=10, b=20, filters=EXP_A)
    assert "error" not in r and r["verdict"] == "none" and "interval" not in r
    note = " ".join(r["notes"])
    assert "run_id" in note and "step" in note and "unit" in note        # the key, the varying column, the remedy


def test_compare_still_refuses_a_varying_category(toolbox):
    """A category column (quantity: x, y) that varies inside the group is still refused."""
    r = toolbox.compare_groups(value="value", group_by=AMOUNT, a=10, b=20, filters={"experiment": "exp_a"})
    assert r["error_type"] == "column" and "quantity" in r["error"]


def test_compare_accepts_a_fixed_time(toolbox):
    """With time and at (and quantity fixed) each run has one row: accepted."""
    r = toolbox.compare_groups(value="value", group_by=AMOUNT, a=10, b=20, x="step", at=10, filters=EXP_A)
    assert "error" not in r and r["a"]["n"] == 3


def test_compare_accepts_one_row_per_run(toolbox):
    """Fixing step and quantity by filters is as good as time + at."""
    r = toolbox.compare_groups(value="value", group_by=AMOUNT, a=10, b=20,
                               filters={**EXP_A, "step": 10})
    assert "error" not in r and r["verdict"] == "established"


def test_summarize_on_pooled_times_gives_a_note(toolbox):
    """Without time, the 11 steps of each run fall in one group: numbers over rows plus a note naming run_id and unit."""
    r = toolbox.summarize_results(value="value", group_by=AMOUNT, filters={**EXP_A})
    assert "error" not in r and r["rows"][0][1] == 33
    note = " ".join(r["notes"])
    assert "run_id" in note and "step" in note and "unit" in note and "filters" in note and "at" in note


def test_summarize_accepts_time_with_at(toolbox):
    """time + at: one row per run in each group."""
    r = toolbox.summarize_results(value="value", group_by=AMOUNT, x="step", at=[5, 10], filters=EXP_A)
    assert "error" not in r and r["rows"][0][2] == 3


def test_summarize_accepts_time_without_at(toolbox):
    """time without at summarizes the last time: accepted."""
    r = toolbox.summarize_results(value="value", group_by=AMOUNT, x="step", filters=EXP_A)
    assert "error" not in r and r["rows"][0][2] == 3


def test_summarize_without_linked_tables_is_not_checked(tmp_path):
    """No linked table, no unit: repeated rows are plain observations."""
    path = tmp_path / "single.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE obs (grp VARCHAR, value DOUBLE)")
    con.execute("INSERT INTO obs VALUES ('a', 1), ('a', 2), ('b', 5)")
    con.close()
    source = DataSource(DataSpec(database=path))
    r = Toolbox(source, SETTINGS).summarize_results(value="value", group_by="grp")
    source.close()
    assert "error" not in r


def test_compare_without_linked_tables_is_not_checked(tmp_path):
    """A table that shares no column with another has no unit to repeat: repeated rows are plain observations."""
    path = tmp_path / "single.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE obs (grp VARCHAR, value DOUBLE)")
    con.execute("INSERT INTO obs VALUES ('a', 1), ('a', 2), ('b', 5), ('b', 7)")
    con.close()
    source = DataSource(DataSpec(database=path))
    r = Toolbox(source, SETTINGS).compare_groups(value="value", group_by="grp", a="a", b="b")
    source.close()
    assert "error" not in r


def test_summarize_refuses_more_than_20_times(toolbox):
    """The limit written in the description is applied: 21 values in 'at' give an 'argument' error."""
    r = toolbox.summarize_results(value="value", group_by=AMOUNT, x="step", at=list(range(21)), filters=EXP_A)
    assert r["error_type"] == "argument" and "20" in r["error"]
    ok = toolbox.summarize_results(value="value", group_by=AMOUNT, x="step", at=list(range(20)), filters=EXP_A)
    assert "error" not in ok


def test_all_null_warning_points_to_json_paths(toolbox):
    """A column that is all NULL gets the warning that suggests a wrong column or JSON path."""
    r = toolbox.query_results("SELECT json_extract_string(config, '$.nothing') AS v FROM runs")
    assert "possibly a wrong column or JSON path" in r["warning"]


def test_compare_needs_one_column(toolbox):
    """Two group_by columns are refused with an explanation."""
    r = toolbox.compare_groups(value="value", group_by=[AMOUNT, "experiment"], a=10, b=20)
    assert r["error_type"] == "argument"


# ---------------------------------------------------------------- other sources

def test_csv_tables_work_like_a_database(tmp_path):
    """The same tools on CSV files (one per table) loaded read-only into memory."""
    con = duckdb.connect(":memory:")
    fill(con)
    for table in ("runs", "series"):
        con.execute(f"COPY {table} TO '{tmp_path / (table + '.csv')}' (HEADER)")
    con.close()
    spec = DataSpec(tables={"runs": tmp_path / "runs.csv", "series": tmp_path / "series.csv"})
    source = DataSource(spec)
    r = Toolbox(source, SETTINGS).summarize_results("value", AMOUNT, "step", [10], EXP_A)
    source.close()
    assert [row[:7] for row in r["rows"]] == [[10, 10, 3, 100, 1, 99, 101], [20, 10, 3, 200, 1, 199, 201]]


# ---------------------------------------------------------------- what the model sees

def test_langchain_tools_have_schemas_and_run(toolbox):
    """The seven tools are wrapped with their argument schemas and return the same results."""
    from sim_analyst.tool_schemas import make_langchain_tools
    tools = {t.name: t for t in make_langchain_tools(toolbox)}
    assert set(tools) == {"describe_database", "query_results", "summarize_results",
                          "describe_trends", "compare_groups", "calculate", "check_quality"}
    assert {"value", "group_by", "x", "at", "filters", "table", "unit"} <= set(tools["summarize_results"].args)
    for tool in tools.values():
        assert "Each row of the data is one replicate" not in tool.description
    assert "comparable observations of one condition" in tools["summarize_results"].description
    assert "Welch" in tools["compare_groups"].description
    assert "replicate" not in tools["compare_groups"].description and "observations" in tools["compare_groups"].description
    r = tools["summarize_results"].invoke({"value": "value", "group_by": AMOUNT, "x": "step",
                                           "at": [10], "filters": EXP_A})
    assert r["rows"][0][:7] == [10, 10, 3, 100, 1, 99, 101]


def strings_in(value):
    """Every key and every text inside a result (nested dicts and lists)."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from strings_in(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from strings_in(v)
    elif isinstance(value, str):
        yield value


def test_no_replicate_in_what_the_model_reads(toolbox):
    """Descriptions and results of summarize_results, describe_trends and compare_groups say 'observations'."""
    from sim_analyst.tool_schemas import DESCRIPTIONS
    texts = list(DESCRIPTIONS.values())
    texts += strings_in(toolbox.summarize_results("value", AMOUNT, "step", [10], EXP_A))
    texts += strings_in(toolbox.describe_trends("value", "step", AMOUNT, EXP_A))
    texts += strings_in(toolbox.compare_groups("value", AMOUNT, 10, 20, "step", 10, EXP_A))
    texts += strings_in(toolbox.calculate("1+1"))
    texts += strings_in(toolbox.compare_groups("value", "runs.run_id", 1, 2, "step", 10, {"quantity": "x"}))
    assert [t for t in texts if "replicate" in t.lower()] == []


# ---------------------------------------------------------------- calculate

def test_calculate_returns_the_expression_and_a_rounded_result(toolbox):
    """The result has 12 significant digits, so floating point noise does not reach the model."""
    r = toolbox.calculate("1/(100*1e-6*25)")
    assert r == {"expression": "1/(100*1e-6*25)", "result": 400.0}


def test_calculate_refusals_are_argument_errors(toolbox):
    """Anything that is not arithmetic is refused with the calculator's message, as an 'argument' error."""
    for bad in ["__import__('os')", "2 +", "1/0", "x + 1", ""]:
        r = toolbox.calculate(bad)
        assert r["error_type"] == "argument" and r["error"], bad


def test_calculate_is_a_langchain_tool(toolbox):
    """The model sees 'calculate' with a description and one argument; it runs and refuses like the method."""
    from sim_analyst.tool_schemas import DESCRIPTIONS, make_langchain_tools
    tool = {t.name: t for t in make_langchain_tools(toolbox)}["calculate"]
    assert set(tool.args) == {"expression"}
    assert "arithmetic expression" in DESCRIPTIONS["calculate"]
    assert tool.invoke({"expression": "(3.5 - 1.2) / 4"})["result"] == 0.575
    assert tool.invoke({"expression": "abs(-1)"})["error_type"] == "argument"


# ---------------------------------------------------------------- serialization (g4.1)

@pytest.fixture()
def dated_toolbox(tmp_path):
    """A small table with DATE, TIMESTAMP, TIME, DECIMAL, INTERVAL and UUID columns."""
    path = tmp_path / "dated.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE events (step INTEGER, day DATE, ts TIMESTAMP, at_time TIME, price DECIMAL(10,2), "
                "span INTERVAL, uid UUID, grp VARCHAR)")
    for i in range(6):
        con.execute("INSERT INTO events VALUES (?, DATE '2026-01-01' + ?, TIMESTAMP '2026-01-01 10:00:00' + "
                    "INTERVAL (?) HOUR, TIME '10:30:00', ?, INTERVAL (?) DAY, uuid(), ?)",
                    [i % 3, i, i, 1.25 * (i + 1), i, "a" if i < 3 else "b"])
    con.close()
    source = DataSource(DataSpec(database=path))
    yield Toolbox(source, SETTINGS)
    source.close()


def test_describe_database_is_json_with_dates(dated_toolbox):
    """Dates, timestamps, times, decimals, intervals and uuids become strings or numbers; date columns get min/max."""
    r = dated_toolbox.describe_database()
    text = json.dumps(r)                              # must not raise
    columns = r["tables"]["events"]["columns"]
    assert columns["day"]["min"] == "2026-01-01" and columns["day"]["max"] == "2026-01-06"
    assert columns["ts"]["min"] == "2026-01-01T10:00:00" and columns["ts"]["max"] == "2026-01-01T15:00:00"
    assert "min" not in columns["grp"] and "min" not in columns["at_time"]
    assert columns["price"]["max"] == 7.5
    assert isinstance(json.loads(text)["tables"]["events"]["sample_rows"][0][1], str)


def test_query_results_is_json_with_dates(dated_toolbox):
    """A query returning every kind of column gives valid JSON."""
    r = dated_toolbox.query_results("SELECT * FROM events ORDER BY day")
    assert "error" not in r
    json.dumps(r)
    row = r["rows"][0]
    assert row[1] == "2026-01-01" and row[2] == "2026-01-01T10:00:00" and row[3] == "10:30:00"
    assert row[4] == 1.25 and isinstance(row[5], str) and isinstance(row[6], str)


def test_summarize_results_is_json_with_dates(dated_toolbox):
    """Grouping by a date column and summarizing a decimal column gives valid JSON."""
    r = dated_toolbox.summarize_results(value="price", group_by="day")
    assert "error" not in r
    json.dumps(r)
    assert r["rows"][0][0] == "2026-01-01"


def test_describe_trends_is_json_with_decimals(dated_toolbox):
    """A decimal value column and a date group give valid JSON."""
    r = dated_toolbox.describe_trends(value="price", x="step", group_by="grp")
    assert "error" not in r
    json.dumps(r)


def test_clean_converts_every_non_json_type():
    """stats.clean is the one place that makes values JSON-safe."""
    import datetime as dt
    import decimal
    import uuid
    from sim_analyst import stats
    assert stats.clean(dt.date(2026, 1, 2)) == "2026-01-02"
    assert stats.clean(dt.datetime(2026, 1, 2, 3, 4, 5)) == "2026-01-02T03:04:05"
    assert stats.clean(dt.time(3, 4)) == "03:04:00"
    assert stats.clean(dt.timedelta(days=1, hours=2)) == "1 day, 2:00:00"
    assert stats.clean(decimal.Decimal("2.50")) == 2.5 and isinstance(stats.clean(decimal.Decimal("2.5")), float)
    assert stats.clean(uuid.UUID(int=1)) == "00000000-0000-0000-0000-000000000001"
    assert stats.clean(b"\x01\xff") == "01ff"
