"""Smoke test of the tools on the frozen toy_shell database (skipped if the file is not there).

The expected numbers are those returned by the v1 tool summarize_results on 29/10/2026:
the generic tools must give the same values when the analyst names the columns.
"""
import pytest

from sim_analyst import stats

from sim_analyst.config import check_fingerprint, load_domain
from sim_analyst.data_access import DataSource
from sim_analyst.tools import Toolbox

LAUNCHES = "config.strategies[0].launches"


@pytest.fixture(scope="module")
def toy():
    """A Toolbox on the frozen database, after checking its fingerprint."""
    domain = load_domain("toy_shell")
    if not domain.data.database.exists():
        pytest.skip("frozen toy_shell database not found")
    check_fingerprint(domain.data)
    source = DataSource(domain.data)
    yield Toolbox(source, domain.settings)
    source.close()


def test_same_means_as_v1(toy):
    """Debris at year 60 for 40, 60 and 120 launches: same mean, min and max as the v1 tool."""
    r = toy.summarize_results(value="value", group_by=LAUNCHES, time="year", at=[60], filters={"metric": "debris"})
    rows = {row[0]: dict(zip(r["columns"], row)) for row in r["rows"]}
    assert len(rows) == 7 and "warning" not in r
    assert {k: rows[40][k] for k in ("n", "mean", "min", "max")} == {"n": 3, "mean": 569.884, "min": 419.517, "max": 722.455}
    assert (rows[60]["min"], rows[60]["max"], rows[60]["mean"]) == (616.612, 1993.459, 1211.517)
    assert rows[120]["mean"] == 2827.189
    # the counts added in g5: 3 runs, no NULL, and the sum of the values is 3 times their mean
    assert (rows[40]["n_rows"], rows[40]["n_null"]) == (3, 0)
    assert rows[40]["sum"] == pytest.approx(3 * rows[40]["mean"], abs=0.01)


def test_forgetting_the_metric_is_refused(toy):
    """Averaging satellites and debris together (a 'false computation' of 29/10) gives an error, no numbers."""
    r = toy.summarize_results(value="value", group_by=LAUNCHES, time="year", at=[60])
    assert r["error_type"] == "column" and "series.metric" in r["error"]


def debris_at_60(toy, launches: int) -> list[float]:
    """The debris values at year 60 of the runs with the given launches."""
    sql = ("SELECT s.value FROM series s JOIN runs r USING (run_id) WHERE s.metric = 'debris' AND s.year = 60 "
           f"AND CAST(json_extract_string(r.config, '$.strategies[0].launches') AS INTEGER) = {launches}")
    return [row[0] for row in toy.query_results(sql)["rows"]]


def test_welch_is_wider_than_bootstrap_with_three_runs(toy):
    """60 vs 70 launches, 3 runs each: the percentile bootstrap is too narrow, Welch's interval is wider."""
    a, b = debris_at_60(toy, 60), debris_at_60(toy, 70)
    assert len(a) == len(b) == 3
    welch = stats.welch_interval(a, b, 0.95)
    boot = stats.bootstrap_difference(a, b, 2000, 12345, 0.95)
    assert welch["interval"][1] - welch["interval"][0] > boot["interval"][1] - boot["interval"][0]
