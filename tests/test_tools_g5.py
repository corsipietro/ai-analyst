"""Tests of the g5 tools: roles and links, categories vs identifiers, `unit`, date trends, check_quality, new counts.

Each test builds a small DuckDB file with known contents, so every number can be checked by hand.
"""
from __future__ import annotations

import datetime as dt
import json
import statistics
import uuid

import duckdb
import pytest

from conftest import SETTINGS
from sim_analyst.config import DataSpec
from sim_analyst.data_access import DataSource
from sim_analyst.tools import Toolbox

AMOUNT = "config.strategies[0].amount"
EXP_A = {"quantity": "x", "experiment": "exp_a"}


def make_toolbox(tmp_path, statements, rows=(), settings=SETTINGS, name=None):
    """A Toolbox on a new DuckDB file: statements are SQL texts, rows are (sql, list of parameter lists)."""
    path = tmp_path / (name or f"{uuid.uuid4().hex}.duckdb")
    con = duckdb.connect(str(path))
    for sql in statements:
        con.execute(sql)
    for sql, many in rows:
        con.executemany(sql, many)
    con.close()
    source = DataSource(DataSpec(database=path))
    return Toolbox(source, settings), source


@pytest.fixture()
def stations_db(tmp_path):
    """stations (40 unique ids) and rentals whose start_station holds the 40 ids plus ONE id that is not a station."""
    stations = [(f"ST{i:02d}", f"city{i % 4}") for i in range(40)]
    rentals = [(f"R{i:03d}", f"ST{i % 40:02d}", None if i % 10 == 0 else float(i)) for i in range(80)]
    rentals.append(("R999", "ST99", 5.0))                   # the orphan
    toolbox, source = make_toolbox(
        tmp_path,
        ["CREATE TABLE stations (station_id VARCHAR, city VARCHAR)",
         "CREATE TABLE rentals (rental_id VARCHAR, start_station VARCHAR, price DOUBLE)"],
        [("INSERT INTO stations VALUES (?, ?)", [list(s) for s in stations]),
         ("INSERT INTO rentals VALUES (?, ?, ?)", [list(r) for r in rentals])])
    yield toolbox
    source.close()


# customers c01..c25: 12 'basic', 13 'pro'; customer i has 1 + i % 3 rentals priced base + 10*j
def customer_rows():
    """(customer_id, plan) and (rental_id, customer_id, start_station, price) of the shop database."""
    customers, rentals = [], []
    for i in range(1, 26):
        plan = "basic" if i <= 12 else "pro"
        customers.append((f"c{i:02d}", plan))
        base = 10 if plan == "basic" else 30
        for j in range(1 + i % 3):
            rentals.append((f"r{i:02d}_{j}", f"c{i:02d}", f"s{(i * 3 + j) % 30}", float(base + 10 * j + (i % 2))))
    return customers, rentals


@pytest.fixture()
def shop(tmp_path):
    """customers(customer_id, plan) and rentals(rental_id, customer_id, start_station, price)."""
    customers, rentals = customer_rows()
    toolbox, source = make_toolbox(
        tmp_path,
        ["CREATE TABLE customers (customer_id VARCHAR, plan VARCHAR)",
         "CREATE TABLE rentals (rental_id VARCHAR, customer_id VARCHAR, start_station VARCHAR, price DOUBLE)"],
        [("INSERT INTO customers VALUES (?, ?)", [list(c) for c in customers]),
         ("INSERT INTO rentals VALUES (?, ?, ?, ?)", [list(r) for r in rentals])])
    yield toolbox
    source.close()


def per_plan(plan):
    """(unit means, raw prices) of the rentals of one plan, computed in Python."""
    customers, rentals = customer_rows()
    ids = {c for c, p in customers if p == plan}
    raw = {}
    for _, customer, _, price in rentals:
        if customer in ids:
            raw.setdefault(customer, []).append(price)
    return [statistics.mean(v) for v in raw.values()], [p for v in raw.values() for p in v], raw


# ---------------------------------------------------------------- describe_database: nulls, role, links

def test_describe_gives_nulls_role_and_links_with_different_names(stations_db):
    """nulls and role per column; start_station links to stations.station_id although the names differ."""
    d = stations_db.describe_database()
    rentals = d["tables"]["rentals"]["columns"]
    assert rentals["price"]["nulls"] == 8 and rentals["rental_id"]["nulls"] == 0
    assert rentals["rental_id"]["role"] == "unique per row"
    assert rentals["start_station"]["role"] == "repeats"
    assert d["tables"]["stations"]["columns"]["station_id"]["role"] == "unique per row"
    assert rentals["start_station"]["links"] == [{"to": "stations.station_id", "orphans": 1}]
    assert "links" not in rentals["price"] and "links" not in rentals["rental_id"]
    assert d["shared_columns"] == {}                      # unchanged: the names differ
    json.dumps(d)


def test_describe_links_follow_shared_names_and_report_no_orphans(toolbox):
    """In the toy-like database series.run_id links to runs.run_id (no orphan); runs.run_id has no link."""
    d = toolbox.describe_database()
    assert d["tables"]["series"]["columns"]["run_id"]["links"] == [{"to": "runs.run_id", "orphans": 0}]
    assert "links" not in d["tables"]["runs"]["columns"]["run_id"]
    assert d["tables"]["series"]["columns"]["step"]["nulls"] == 0
    assert d["tables"]["series"]["columns"]["step"]["role"] == "repeats"


# ---------------------------------------------------------------- mixing: only categories count

def test_an_identifier_text_column_that_varies_does_not_refuse(tmp_path):
    """obs_id is unique per row and varies in every group: it is an identifier, so no refusal."""
    rows = [[f"o{i}", "a" if i < 15 else "b", float(i)] for i in range(30)]
    toolbox, source = make_toolbox(tmp_path, ["CREATE TABLE obs (obs_id VARCHAR, grp VARCHAR, value DOUBLE)"],
                                   [("INSERT INTO obs VALUES (?, ?, ?)", rows)])
    r = toolbox.summarize_results(value="value", group_by="grp")
    source.close()
    assert "error" not in r and [row[:2] for row in r["rows"]] == [["a", 15], ["b", 15]]


def test_an_identifier_with_many_values_that_repeats_does_not_refuse(shop):
    """rentals.customer_id repeats but has 25 values (more than 20): not a category, so no refusal."""
    r = shop.summarize_results(value="price", group_by="plan")
    assert "error" not in r


def test_a_category_text_column_still_refuses(tmp_path):
    """kind has 2 values, repeats, and varies inside the group: refused as before, message unchanged."""
    rows = [[f"o{i}", "a" if i < 15 else "b", "x" if i % 2 else "y", float(i)] for i in range(30)]
    toolbox, source = make_toolbox(
        tmp_path, ["CREATE TABLE obs (obs_id VARCHAR, grp VARCHAR, kind VARCHAR, value DOUBLE)"],
        [("INSERT INTO obs VALUES (?, ?, ?, ?)", rows)])
    r = toolbox.summarize_results(value="value", group_by="grp")
    source.close()
    assert r["error_type"] == "column" and "NO RESULT" in r["error"]
    assert "obs.kind (values ['x', 'y'])" in r["error"] and "obs_id" not in r["error"]


# ---------------------------------------------------------------- summarize_results: counts

def test_summarize_reports_n_rows_n_null_and_sum(tmp_path):
    """n counts the values, n_rows the rows including NULL, n_null the NULL values, sum the sum of the values."""
    toolbox, source = make_toolbox(
        tmp_path, ["CREATE TABLE obs (grp VARCHAR, value DOUBLE)"],
        [("INSERT INTO obs VALUES (?, ?)", [["a", 1.0], ["a", 2.0], ["a", None], ["b", 5.0], ["c", None]])])
    r = toolbox.summarize_results(value="value", group_by="grp")
    source.close()
    assert r["columns"] == ["grp", "n", "mean", "std", "min", "max", "n_rows", "n_null", "sum"]
    assert r["rows"][0] == ["a", 2, 1.5, 0.707, 1, 2, 3, 1, 3]
    assert r["rows"][1] == ["b", 1, 5, None, 5, 5, 1, 0, 5]
    assert r["rows"][2] == ["c", 0, None, None, None, None, 1, 1, None]


# ---------------------------------------------------------------- unit

def test_summarize_with_unit_averages_per_unit_first(shop):
    """Rows per customer are averaged first; the statistics are over customers; n_units and rows per unit reported."""
    r = shop.summarize_results(value="price", group_by="plan", unit="customer_id")
    assert "error" not in r and r["unit"] == ["customer_id"]
    assert r["columns"] == ["plan", "n_units", "mean", "std", "min", "max", "n_rows", "n_null", "sum",
                            "rows_per_unit_min", "rows_per_unit_max"]
    means, raw, per_customer = per_plan("basic")
    row = r["rows"][0]
    assert row[0] == "basic" and row[1] == len(means) == 12
    assert row[2] == pytest.approx(statistics.mean(means), abs=1e-3)
    assert row[3] == pytest.approx(statistics.stdev(means), abs=1e-3)
    assert (row[4], row[5]) == (pytest.approx(min(means), abs=1e-3), pytest.approx(max(means), abs=1e-3))
    assert row[6] == len(raw) and row[7] == 0 and row[8] == pytest.approx(sum(raw))
    assert (row[9], row[10]) == (min(map(len, per_customer.values())), max(map(len, per_customer.values())))
    assert row[2] != pytest.approx(statistics.mean(raw), abs=1e-3)       # not the mean over rows


def test_summarize_without_unit_on_repeated_rows_gives_a_note_not_a_refusal(shop):
    """Customers have several rentals: numbers over rows are returned, with a note naming the column and unit."""
    r = shop.summarize_results(value="price", group_by="plan")
    assert "error" not in r
    _, raw, _ = per_plan("basic")
    assert r["rows"][0][:3] == ["basic", len(raw), pytest.approx(statistics.mean(raw), abs=1e-3)]
    note = " ".join(r["notes"])
    assert "customer_id" in note and "unit" in note and "3" in note        # max rows per customer is 3


def test_summarize_with_unit_has_no_repetition_note(shop):
    """Once the unit is given, the repetition check is not needed."""
    r = shop.summarize_results(value="price", group_by="plan", unit=["customer_id"])
    assert "error" not in r and not any("unit" in n for n in r.get("notes", []))


def test_compare_without_unit_on_repeated_rows_has_no_verdict(shop):
    """Means and spreads are returned; the verdict is 'none', with the reason; no interval."""
    r = shop.compare_groups(value="price", group_by="plan", a="basic", b="pro")
    assert "error" not in r and r["verdict"] == "none"
    assert "independent" in r["verdict_reason"] and "unit" in r["verdict_reason"]
    assert "interval" not in r and "difference_b_minus_a" not in r
    assert r["a"]["n"] == len(per_plan("basic")[1]) and "mean" in r["a"] and "min" in r["b"]
    assert "customer_id" in " ".join(r["notes"])


def test_compare_with_unit_is_welch_over_units(shop):
    """The verdict comes from Welch's interval over the unit means."""
    from sim_analyst import stats
    r = shop.compare_groups(value="price", group_by="plan", a="basic", b="pro", unit="customer_id")
    assert "error" not in r and r["unit"] == ["customer_id"] and r["method"] == "welch"
    a_means, b_means = per_plan("basic")[0], per_plan("pro")[0]
    assert (r["a"]["n_units"], r["b"]["n_units"]) == (12, 13)
    assert r["a"]["mean"] == pytest.approx(statistics.mean(a_means), abs=1e-3)
    welch = stats.welch_interval(a_means, b_means, SETTINGS["bootstrap"]["confidence"])
    assert r["difference_b_minus_a"] == pytest.approx(welch["difference"], abs=1e-3)
    assert r["degrees_of_freedom"] == pytest.approx(welch["degrees_of_freedom"], abs=1e-3)
    assert r["verdict"] == "established"
    assert r["a"]["rows_per_unit"] == [1, 3] and r["b"]["rows_per_unit"] == [1, 3]


def test_toy_shape_compare_without_at_has_no_verdict_and_names_run_id(toolbox):
    """Runs + yearly series, compare without time/at: all the steps of a run are pooled: no verdict, note names run_id."""
    r = toolbox.compare_groups(value="value", group_by=AMOUNT, a=10, b=20, filters=EXP_A)
    assert "error" not in r and r["verdict"] == "none"
    assert "run_id" in " ".join(r["notes"]) and "step" in " ".join(r["notes"])
    assert "interval" not in r and r["a"]["n"] == 33


def test_toy_shape_summarize_pooled_steps_gives_a_note(toolbox):
    """Same shape in summarize_results: the numbers are returned with a note that names run_id and the unit argument."""
    r = toolbox.summarize_results(value="value", group_by=AMOUNT, filters=EXP_A)
    assert "error" not in r and "run_id" in " ".join(r["notes"]) and "unit" in " ".join(r["notes"])
    r = toolbox.summarize_results(value="value", group_by=AMOUNT, filters=EXP_A, unit="run_id")
    assert "error" not in r and r["rows"][0][1] == 3        # 3 runs, each averaged over its 11 steps


def test_compare_with_unit_and_a_fixed_time(toolbox):
    """With time + at each run has one row; unit='run_id' gives the same numbers as without unit."""
    plain = toolbox.compare_groups(value="value", group_by=AMOUNT, a=10, b=20, time="step", at=10, filters=EXP_A)
    unit = toolbox.compare_groups(value="value", group_by=AMOUNT, a=10, b=20, time="step", at=10, filters=EXP_A,
                                  unit="run_id")
    assert "notes" not in plain or not any("unit" in n for n in plain["notes"])
    assert unit["difference_b_minus_a"] == plain["difference_b_minus_a"] and unit["verdict"] == plain["verdict"]


def test_unit_must_be_a_known_column(shop):
    """A wrong unit is a 'column' error that lists the columns."""
    assert shop.summarize_results(value="price", unit="no_such")["error_type"] == "column"


# ---------------------------------------------------------------- describe_trends with dates

@pytest.fixture()
def daily(tmp_path):
    """84 days from Monday 2026-01-05 (12 whole weeks); v = days since the start; one site."""
    start = dt.date(2026, 1, 5)
    rows = [[start + dt.timedelta(days=i), "s1", float(i)] for i in range(84)]
    toolbox, source = make_toolbox(tmp_path, ["CREATE TABLE daily (day DATE, site VARCHAR, v DOUBLE)"],
                                   [("INSERT INTO daily VALUES (?, ?, ?)", rows)])
    yield toolbox
    source.close()


def test_trends_with_a_date_default_period_keeps_at_most_max_points(daily):
    """84 days > 40 points, 12 weeks <= 40: the default period is the week; times are ISO strings; slope per day."""
    r = daily.describe_trends(value="v", time="day")
    assert "error" not in r and r["period"] == "week"
    g = r["groups"][0]
    assert g["initial"] == {"time": "2026-01-05", "mean": 3}
    assert g["final"]["time"] == "2026-03-23"
    assert g["slope_last_part"]["slope_per_day"] == pytest.approx(1.0)
    assert "slope_per_time_unit" not in g["slope_last_part"]
    assert len(g["series_sampled"]["rows"]) == 12 and g["series_sampled"]["rows"][1][0] == "2026-01-12"
    assert g["shape_sampled"] == "increasing"
    json.dumps(r)


def test_trends_with_a_date_day_period_is_thinned_over_max_points(daily):
    """Explicit day period: 84 points exceed 40, the sampled series is thinned and a note says so."""
    r = daily.describe_trends(value="v", time="day", period="day")
    g = r["groups"][0]
    assert r["period"] == "day" and g["slope_last_part"]["slope_per_day"] == pytest.approx(1.0)
    rows = g["series_sampled"]["rows"]
    assert len(rows) <= 40 and rows[0][0] == "2026-01-05" and rows[-1][0] == "2026-03-29"
    assert any("every" in n for n in r["notes"])


def test_trends_with_a_date_month_period(daily):
    """Months: Jan, Feb, Mar (week 1 starts in January)."""
    r = daily.describe_trends(value="v", time="day", period="month")
    times = [row[0] for row in r["groups"][0]["series_sampled"]["rows"]]
    assert times == ["2026-01-01", "2026-02-01", "2026-03-01"]


def test_trends_with_a_date_start_and_end(daily):
    """start and end may be ISO dates for a date time column."""
    r = daily.describe_trends(value="v", time="day", period="day", start="2026-02-01", end="2026-02-10")
    rows = r["groups"][0]["series_sampled"]["rows"]
    assert rows[0][0] == "2026-02-01" and rows[-1][0] == "2026-02-10" and len(rows) == 10


def test_trends_with_a_timestamp_time(tmp_path):
    """A TIMESTAMP column is cut to days; two readings in a day are one point."""
    rows = [[dt.datetime(2026, 1, 1 + i // 2, 6 * (i % 2)), float(i)] for i in range(20)]
    toolbox, source = make_toolbox(tmp_path, ["CREATE TABLE t (ts TIMESTAMP, v DOUBLE)"],
                                   [("INSERT INTO t VALUES (?, ?)", rows)])
    r = toolbox.describe_trends(value="v", time="ts", period="day")
    source.close()
    g = r["groups"][0]
    assert g["series_sampled"]["rows"][0][:2] == ["2026-01-01", 2] and len(g["series_sampled"]["rows"]) == 10
    assert g["slope_last_part"]["slope_per_day"] == pytest.approx(2.0)


def test_trends_numeric_time_is_unchanged(toolbox):
    """A numeric time still gives step and slope_per_time_unit, no period."""
    r = toolbox.describe_trends(value="value", time="step", group_by=AMOUNT, filters=EXP_A)
    assert "period" not in r and r["step"] == 1
    assert "slope_per_time_unit" in r["groups"][0]["slope_last_part"]


def test_trends_time_must_be_numeric_or_date(daily):
    """A text column as time is refused with a 'column' error."""
    assert daily.describe_trends(value="v", time="site")["error_type"] == "column"


# ---------------------------------------------------------------- check_quality

@pytest.fixture()
def readings(tmp_path):
    """60 days x 2 sites. temp is NULL for site A on days 10-19 (a 10-day run) and for site B on days 3 and 40."""
    start = dt.date(2026, 1, 1)
    rows = []
    for i in range(60):
        for site in ("A", "B"):
            temp = None if (site == "A" and 10 <= i < 20) or (site == "B" and i in (3, 40)) else float(i)
            city = {"A": "Rome", "B": "Milan"}[site]
            if site == "A" and i == 5:
                city = "rome "
            if site == "A" and i == 6:
                city = "ROME"
            amount = ["12,5", "3.4", "7"][i % 3] if not (site == "A" and i == 7) else "n/a"
            rows.append([f"R{i}{site}", start + dt.timedelta(days=i), site, temp, city, amount])
    rows.append(list(rows[0]))                               # an exact copy of the first row
    rows.append(["R5A", start, "B", 1.0, "Milan", "1"])      # an id used twice by different rows
    toolbox, source = make_toolbox(
        tmp_path, ["CREATE TABLE readings (reading_id VARCHAR, day DATE, site VARCHAR, temp DOUBLE, "
                   "city VARCHAR, amount VARCHAR)"],
        [("INSERT INTO readings VALUES (?, ?, ?, ?, ?, ?)", rows)])
    yield toolbox
    source.close()


def test_quality_counts_rows_and_nulls_per_column(readings):
    """rows; NULL count and share only for the columns that have NULLs."""
    r = readings.check_quality(table="readings")
    assert r["table"] == "readings" and r["rows"] == 122
    assert r["nulls"] == {"temp": {"count": 12, "share": 0.098}}


def test_quality_null_share_per_group(readings):
    """Site A: 10 NULL of 61 rows; site B: 2 of 61."""
    r = readings.check_quality(table="readings", group_by="site")
    assert r["null_share_by_group"] == {"temp": {"A": 0.164, "B": 0.033}}


def test_quality_finds_the_ten_day_run_of_nulls(readings):
    """A run of at least 3 consecutive periods, all NULL, is reported with start and end; B's isolated NULLs are not."""
    r = readings.check_quality(table="readings", group_by="site", time="day")
    assert r["null_runs"] == [{"group": "A", "column": "temp", "start": "2026-01-11", "end": "2026-01-20",
                               "periods": 10}]
    assert r["period"] == "day"
    shares = r["null_share_by_period"]["temp"]
    assert shares["2026-01-11"] == pytest.approx(0.5) and shares["2026-01-04"] == pytest.approx(0.5)
    assert any("null_share_by_period" in n and "10 of 12" in n for n in r["notes"])      # capped at 10 periods


def test_quality_exact_duplicates_and_identifier_duplicates(readings):
    """One exact copy; reading_id R0A (the copy) and R5A (reused) are duplicated."""
    r = readings.check_quality(table="readings")
    assert r["duplicate_rows"]["extra_rows"] == 1 and r["duplicate_rows"]["groups"] == 1
    ids = {d["column"]: d for d in r["identifier_duplicates"]}
    assert set(ids) == {"reading_id"} and ids["reading_id"]["duplicated_values"] == 2
    assert ids["reading_id"]["extra_rows"] == 2 and sorted(ids["reading_id"]["examples"]) == ["R0A", "R5A"]


def test_quality_text_variants_after_trim_and_lower(readings):
    """'Rome', 'rome ' and 'ROME' are the same city."""
    r = readings.check_quality(table="readings")
    assert len(r["text_variants"]) == 1
    v = r["text_variants"][0]
    assert v["column"] == "city" and sorted(v["values"]) == ["ROME", "Rome", "rome "]


def test_quality_numbers_stored_as_text_with_a_decimal_comma(readings):
    """amount is mostly numbers (also '12,5'); the odd values are shown."""
    r = readings.check_quality(table="readings")
    assert [d["column"] for d in r["numbers_as_text"]] == ["amount"]
    d = r["numbers_as_text"][0]
    assert d["decimal_comma"] is True and d["share"] > 0.9 and d["not_numbers"] == ["n/a"]


def test_quality_orphans_of_the_links(stations_db):
    """start_station has one value that is not a station."""
    r = stations_db.check_quality(table="rentals")
    assert r["orphans"] == [{"column": "start_station", "to": "stations.station_id", "orphans": 1,
                             "examples": ["ST99"]}]
    assert stations_db.check_quality(table="stations").get("orphans") is None


def test_quality_lists_are_capped_and_say_so(tmp_path):
    """12 columns with a NULL each: 10 are listed and a note says '10 of 12'."""
    columns = ", ".join(f"c{i} DOUBLE" for i in range(12))
    rows = [[None if j == i else 1.0 for j in range(12)] for i in range(12)]
    toolbox, source = make_toolbox(tmp_path, [f"CREATE TABLE wide ({columns})"],
                                   [(f"INSERT INTO wide VALUES ({', '.join('?' * 12)})", rows)])
    r = toolbox.check_quality(table="wide")
    source.close()
    assert len(r["nulls"]) == 10 and any("nulls" in n and "10 of 12" in n for n in r["notes"])


def test_quality_clean_table_reports_no_problem_keys(shop):
    """No NULL, duplicate, variant, number-as-text or orphan: only the row count (and the table)."""
    r = shop.check_quality(table="customers")
    assert r == {"table": "customers", "rows": 25}


def test_quality_filters_group_by_and_errors(readings):
    """filters restrict the rows; unknown table and column are errors; a group_by of another table is refused."""
    r = readings.check_quality(table="readings", filters={"site": "B"})
    assert r["rows"] == 61 and r["nulls"]["temp"]["count"] == 2
    assert readings.check_quality(table="nope")["error_type"] == "column"
    assert readings.check_quality(table="readings", group_by="nope")["error_type"] == "column"
    assert readings.check_quality(table="readings", time="site")["error_type"] == "column"


def test_quality_is_json_and_leaves_data_alone(readings):
    """The report is JSON-safe (dates, floats) and a langchain tool exists."""
    from sim_analyst.tool_schemas import make_langchain_tools
    json.dumps(readings.check_quality(table="readings", group_by="site", time="day"))
    tool = {t.name: t for t in make_langchain_tools(readings)}["check_quality"]
    assert tool.invoke({"table": "readings"})["rows"] == 122


# ---------------------------------------------------------------- descriptions and version

def test_descriptions_tell_when_to_use_unit_period_and_check_quality():
    """The texts the model reads mention the new arguments and the new tool."""
    from sim_analyst.tool_schemas import ARGS_SCHEMAS, DESCRIPTIONS
    assert "unit" in DESCRIPTIONS["summarize_results"] and "unit" in DESCRIPTIONS["compare_groups"]
    assert "period" in DESCRIPTIONS["describe_trends"] and "date" in DESCRIPTIONS["describe_trends"].lower()
    assert "NULL" in DESCRIPTIONS["check_quality"] and "check_quality" in ARGS_SCHEMAS
    assert {"unit"} <= set(ARGS_SCHEMAS["summarize_results"].model_fields)
    assert {"unit"} <= set(ARGS_SCHEMAS["compare_groups"].model_fields)
    assert {"period"} <= set(ARGS_SCHEMAS["describe_trends"].model_fields)
    assert set(ARGS_SCHEMAS["check_quality"].model_fields) == {"table", "group_by", "time", "period", "filters"}
    assert all("replicate" not in text.lower() for text in DESCRIPTIONS.values())


def test_tools_version_is_g5():
    from sim_analyst.tools import TOOLS_VERSION
    assert TOOLS_VERSION == "g5"


def test_summarize_and_compare_accept_a_date_time_with_at(daily):
    """A date written as text works in 'at'; a missing date is reported in the notes."""
    r = daily.summarize_results(value="v", time="day", at=["2026-01-06", "2030-01-01"])
    assert "error" not in r and r["rows"][0][:3] == ["2026-01-06", 1, 1]
    assert any("2030-01-01" in n for n in r["notes"])
    c = daily.compare_groups(value="v", group_by="site", a="s1", b="s1", time="day", at="2026-01-06")
    assert "error" not in c and c["a"]["n"] == 1


def test_quality_does_not_flag_a_foreign_key_as_a_duplicated_identifier(shop):
    """rentals.customer_id repeats by nature (it links to customers): not reported; rental_id would be."""
    r = shop.check_quality(table="rentals")
    assert "identifier_duplicates" not in r


# ================================================================ g5 fixes (second round)

# ---------------------------------------------------------------- mixing: JSON paths and measure names refuse, the rest is a note

@pytest.fixture()
def visits(tmp_path):
    """grp (2 values) x day (1-30), a free-text note column with 4 values that varies inside each group."""
    rows = [[grp, day, f"note{day % 4}", float(day + (5 if grp == "b" else 0))]
            for day in range(1, 31) for grp in ("a", "b")]
    toolbox, source = make_toolbox(tmp_path, ["CREATE TABLE visits (grp VARCHAR, day INTEGER, note VARCHAR, value DOUBLE)"],
                                   [("INSERT INTO visits VALUES (?, ?, ?, ?)", rows)])
    yield toolbox
    source.close()


@pytest.fixture()
def places(tmp_path):
    """30 places: name (15 values) and region (3 values) belong to the entity, type has 2 values."""
    rows = [[i, f"place{i % 15}", f"region{i % 3}", f"type{i % 2}", 1900 + i % 20, float(100 + i)] for i in range(30)]
    toolbox, source = make_toolbox(
        tmp_path, ["CREATE TABLE places (place_id INTEGER, name VARCHAR, region VARCHAR, type VARCHAR, "
                   "founded INTEGER, population DOUBLE)"],
        [("INSERT INTO places VALUES (?, ?, ?, ?, ?, ?)", rows)])
    yield toolbox
    source.close()


@pytest.fixture()
def long_table(tmp_path):
    """Long format: entity x t x measure -> value (the measure name is a column)."""
    rows = [[e, t, m, float(e * t + (100 if m == "m2" else 0))]
            for e in range(1, 11) for t in range(1, 6) for m in ("m1", "m2")]
    toolbox, source = make_toolbox(
        tmp_path, ["CREATE TABLE long_t (entity INTEGER, t INTEGER, measure VARCHAR, value DOUBLE)"],
        [("INSERT INTO long_t VALUES (?, ?, ?, ?)", rows)])
    yield toolbox
    source.close()


def test_a_free_text_category_that_varies_gives_a_note_not_a_refusal(visits):
    r = visits.summarize_results(value="value", group_by="grp")
    assert "error" not in r and [row[:2] for row in r["rows"]] == [["a", 30], ["b", 30]]
    assert "aggregated over note (4 values)" in r["notes"]


def test_name_and_region_of_the_same_entity_give_notes_when_grouping_by_type(places):
    r = places.summarize_results(value="population", group_by="type")
    assert "error" not in r
    assert "aggregated over name (15 values), region (3 values)" in " ".join(r["notes"])


def test_the_note_names_at_most_three_columns(tmp_path):
    cols = ", ".join(f"c{i} VARCHAR" for i in range(5))
    rows = [[g, f"x{i % 2}", f"x{i % 3}", f"x{i % 5}", f"x{i % 7}", f"x{i % 11}", float(i), i // 2 * 7 + (i % 2)]
            for i, g in zip(range(60), ["a"] * 30 + ["b"] * 30)]
    toolbox, source = make_toolbox(tmp_path, [f"CREATE TABLE w (g VARCHAR, {cols}, value DOUBLE, k INTEGER)"],
                                   [("INSERT INTO w VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)])
    r = toolbox.summarize_results(value="value", group_by="g")
    source.close()
    assert "error" not in r
    note = " ".join(r["notes"])
    assert "aggregated over c0 (" in note and "c2 (" in note and "and 2 more" in note and "c3 (" not in note


def test_measure_name_detection_on_both_shapes(long_table, places, toolbox):
    assert long_table.is_measure_name("long_t", "measure", "value") is True
    assert places.is_measure_name("places", "name", "population") is False
    assert places.is_measure_name("places", "region", "population") is False
    assert toolbox.is_measure_name("series", "quantity", "value") is True      # the toy


def test_a_long_table_without_a_filter_on_the_measure_name_is_refused(long_table):
    r = long_table.summarize_results(value="value", group_by="entity")
    assert r["error_type"] == "column" and "long_t.measure (values ['m1', 'm2'])" in r["error"]
    ok = long_table.summarize_results(value="value", group_by="entity", filters={"measure": "m1"})
    assert "error" not in ok


def test_a_varying_json_path_is_still_refused(toolbox):
    r = toolbox.summarize_results(value="value", group_by="step", filters={"quantity": "x"})
    assert r["error_type"] == "column" and "runs.config (JSON" in r["error"]


def test_the_category_limit_is_its_own_constant():
    from sim_analyst import tools
    assert tools.CATEGORY_MAX_VALUES == 20


# ---------------------------------------------------------------- filters with operators

@pytest.fixture()
def grid(tmp_path):
    """n 1-10, a date per row (2026-01-01 + n - 1), g a/b, v NULL for n in (2, 4, 6), w = n * 10."""
    rows = [[n, dt.date(2026, 1, n), "a" if n <= 5 else "b", None if n in (2, 4, 6) else float(n), float(n * 10)]
            for n in range(1, 11)]
    toolbox, source = make_toolbox(tmp_path, ["CREATE TABLE grid (n INTEGER, d DATE, g VARCHAR, v DOUBLE, w DOUBLE)"],
                                   [("INSERT INTO grid VALUES (?, ?, ?, ?, ?)", rows)])
    yield toolbox
    source.close()


def total(grid, filters):
    r = grid.summarize_results(value="w", filters=filters)
    assert "error" not in r, r
    return r["rows"][0][0], r["rows"][0][1]       # n, mean (no group: the row starts with n)


def test_between_on_numbers_and_dates(grid):
    assert total(grid, {"n": {"between": [3, 5]}}) == (3, 40)
    assert total(grid, {"d": {"between": ["2026-01-02", "2026-01-04"]}}) == (3, 30)


def test_comparison_operators(grid):
    assert total(grid, {"n": {">=": 9}}) == (2, 95)
    assert total(grid, {"n": {">": 9}}) == (1, 100)
    assert total(grid, {"n": {"<": 3}}) == (2, 15)
    assert total(grid, {"n": {"<=": 3}}) == (3, 20)
    assert total(grid, {"d": {">=": "2026-01-09"}}) == (2, 95)
    assert total(grid, {"n": {">=": 2, "<": 5}}) == (3, 30)               # two operators on one column


def test_is_null_true_and_false_counts_match(grid):
    r_null = grid.summarize_results(value="w", filters={"v": {"is_null": True}})
    r_not = grid.summarize_results(value="w", filters={"v": {"is_null": False}})
    assert r_null["rows"][0][0] == 3 and r_not["rows"][0][0] == 7
    assert grid.summarize_results(value="v", filters={"v": {"is_null": True}})["rows"][0][5] == 3   # n_rows, while n = 0
    assert grid.summarize_results(value="w", filters={"g": {"is_null": False}})["rows"][0][0] == 10


def test_plain_values_and_lists_work_as_before(grid):
    assert total(grid, {"g": "a"}) == (5, 30) and total(grid, {"n": [1, 2]}) == (2, 15)


def test_none_in_a_plain_filter_points_to_is_null(grid):
    for bad in ({"g": None}, {"g": ["a", None]}):
        r = grid.summarize_results(value="w", filters=bad)
        assert r["error_type"] == "argument" and "is_null" in r["error"]


def test_bad_operators_are_argument_errors(grid):
    r = grid.summarize_results(value="w", filters={"n": {"sql": "SELECT 1"}})
    assert r["error_type"] == "argument" and "between" in r["error"] and "is_null" in r["error"]
    assert grid.summarize_results(value="w", filters={"g": {">=": "a"}})["error_type"] == "argument"    # text column
    assert grid.summarize_results(value="w", filters={"n": {"between": [1]}})["error_type"] == "argument"
    assert grid.summarize_results(value="w", filters={"v": {"is_null": "yes"}})["error_type"] == "argument"
    assert grid.summarize_results(value="w", filters={"d": {">=": "not a date"}})["error_type"] == "argument"


def test_operator_filters_work_in_the_other_tools(grid):
    assert grid.compare_groups(value="w", group_by="g", a="a", b="b", filters={"n": {"between": [3, 8]}})["a"]["n"] == 3
    assert "error" not in grid.describe_trends(value="w", time="n", filters={"n": {">=": 2}})
    assert grid.check_quality(table="grid", filters={"n": {"<=": 5}})["rows"] == 5


def test_filters_help_has_one_example_of_each_form():
    from sim_analyst.tool_schemas import FILTERS_HELP
    for form in ("between", ">=", "is_null"):
        assert form in FILTERS_HELP


# ---------------------------------------------------------------- left joins

@pytest.fixture()
def orders(tmp_path):
    """20 orders; customers 1-3 exist (tiers), orders of customer 99 (4 of them) have no customer row."""
    rows = [[f"o{i}", (i % 3) + 1 if i < 16 else 99, float(10 + i)] for i in range(20)]
    toolbox, source = make_toolbox(
        tmp_path, ["CREATE TABLE orders (order_id VARCHAR, cust INTEGER, amount DOUBLE)",
                   "CREATE TABLE customers (cust INTEGER, tier VARCHAR)"],
        [("INSERT INTO orders VALUES (?, ?, ?)", rows),
         ("INSERT INTO customers VALUES (?, ?)", [[1, "gold"], [2, "silver"], [3, "silver"]])])
    yield toolbox
    source.close()


def test_unmatched_rows_form_a_null_group_with_a_note(orders):
    r = orders.summarize_results(value="amount", group_by="tier")
    assert "error" not in r
    groups = {row[0]: row[1] for row in r["rows"]}
    assert groups == {"gold": 6, "silver": 10, None: 4} and sum(groups.values()) == 20      # totals include them
    note = " ".join(r["notes"])
    assert "4 rows of orders have no match in customers.tier" in note and "99" in note and "NULL group" in note


def test_unmatched_rows_are_included_when_the_joined_column_is_not_a_group(orders):
    r = orders.summarize_results(value="amount", filters={"tier": {"is_null": True}})
    assert r["rows"][0][0] == 4
    assert "4 rows of orders have no match in customers.tier" in " ".join(r["notes"]) and "included" in " ".join(r["notes"])


def test_no_note_when_every_row_has_its_match(toolbox):
    r = toolbox.summarize_results(value="value", group_by=AMOUNT, time="step", at=[10], filters=EXP_A)
    assert not any("no match" in n for n in r.get("notes", []))


def test_a_filter_on_the_joined_column_drops_the_unmatched_rows_without_a_note(orders):
    r = orders.summarize_results(value="amount", group_by="tier", filters={"tier": ["gold", "silver"]})
    assert {row[0] for row in r["rows"]} == {"gold", "silver"} and not any("no match" in n for n in r.get("notes", []))


# ---------------------------------------------------------------- wording and small fixes

FORBIDDEN = ("customer", "station", "rental", "simulation", "experiment", "parameter", "results database",
             "replicate", "seed", "toy")


def model_texts():
    """Every description the model reads in tool_schemas: tools, arguments, and the *_HELP constants."""
    from sim_analyst import tool_schemas
    texts = list(tool_schemas.DESCRIPTIONS.values())
    texts += [v for k, v in vars(tool_schemas).items() if k.endswith("_HELP") and isinstance(v, str)]
    for schema in tool_schemas.ARGS_SCHEMAS.values():
        texts += [f.description for f in schema.model_fields.values() if f.description]
    return texts


def test_no_domain_words_in_what_the_model_reads():
    import re
    found = [(w, t[:60]) for t in model_texts() for w in FORBIDDEN if re.search(rf"\b{w}s?\b", t.lower())]
    assert found == []
    from sim_analyst.tool_schemas import UNIT_HELP
    assert "several rows" in UNIT_HELP


def test_identifier_names_have_no_language_specific_word():
    from sim_analyst.tools import IDENTIFIER_NAME
    assert IDENTIFIER_NAME.search("user_id") and IDENTIFIER_NAME.search("code") and not IDENTIFIER_NAME.search("codice")


def test_trends_error_names_the_argument_that_is_wrong(daily):
    assert "value 'site'" in daily.describe_trends(value="site", time="day")["error"]
    assert "time 'site'" in daily.describe_trends(value="v", time="site")["error"]


def test_repetition_note_mentions_time_and_at_only_with_a_time_column():
    from sim_analyst.tools import Toolbox
    found = (["k"], 3, ["t.x"])
    assert "time + at" in Toolbox.repetition_note(found, with_time=True)
    assert "time + at" not in Toolbox.repetition_note(found, with_time=False)
    assert "unit=" in Toolbox.repetition_note(found, with_time=False)


def test_csv_timestamps_with_minutes_are_read_as_timestamps_and_a_text_date_matches_a_date(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text("ts,v\n" + "\n".join(f"2026-01-0{d} 10:30,{d}" for d in range(1, 6)) + "\n")
    source = DataSource(DataSpec(tables={"t": csv}))
    toolbox = Toolbox(source, SETTINGS)
    assert toolbox.schema["t"]["ts"].startswith("TIMESTAMP")
    r = toolbox.describe_trends(value="v", time="ts", period="day")
    assert "error" not in r and r["groups"][0]["initial"]["time"] == "2026-01-01"
    assert toolbox.summarize_results(value="v", time="ts", at="2026-01-03 10:30")["rows"][0][2] == 3
    source.close()


def test_a_varchar_date_in_at_matches_a_date_column(daily):
    r = daily.summarize_results(value="v", time="day", at="2026-01-06")
    assert r["rows"][0][:3] == ["2026-01-06", 1, 1]
