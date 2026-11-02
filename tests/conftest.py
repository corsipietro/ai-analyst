"""A small fake results database with known answers, shared by the tests.

Names are deliberately NOT those of the toy shell (step, quantity, amount, rate), to show that
the tools do not depend on them. Content:
- runs(run_id, experiment, config, seed): experiment 'exp_a' has amount 10 and 20 (rate 0.5),
  3 repetitions each (run_id 1-6); experiment 'exp_b' has amount 10 with rate 0.9 (run_id 7-8).
- series(run_id, step, quantity, value), steps 0..10, two quantities:
  x = amount * step + r          with r = -1, 0, +1 for the three repetitions (r = 0 in exp_b)
  y = 100 * r + 0.1 * amount     (pure noise: groups differ by 1, repetitions by 100)
Known results: at step 10 the mean of x is 100 (min 99, max 101) for amount 10 and 200 for amount 20.
"""
from __future__ import annotations

import json

import duckdb
import pytest

from sim_analyst.config import DataSpec
from sim_analyst.data_access import DataSource
from sim_analyst.tools import Toolbox

SETTINGS = {"limits": {"max_rows": 200, "max_points": 40, "recursion_limit": 30},
            "bootstrap": {"resamples": 2000, "seed": 12345, "confidence": 0.95, "min_n": 30}}


def fill(con) -> None:
    """Create and fill the two tables of the fake database."""
    con.execute("CREATE TABLE runs (run_id INTEGER, experiment VARCHAR, config VARCHAR, seed BIGINT)")
    con.execute("CREATE TABLE series (run_id INTEGER, step INTEGER, quantity VARCHAR, value DOUBLE)")
    plan = [("exp_a", 0.5, 10, r) for r in (-1, 0, 1)] + [("exp_a", 0.5, 20, r) for r in (-1, 0, 1)] \
        + [("exp_b", 0.9, 10, 0), ("exp_b", 0.9, 10, 0)]
    for run_id, (experiment, rate, amount, r) in enumerate(plan, start=1):
        config = {"model": {"rate": rate}, "strategies": [{"type": "fixed", "amount": amount}]}
        con.execute("INSERT INTO runs VALUES (?, ?, ?, ?)", [run_id, experiment, json.dumps(config), 100 + run_id])
        for step in range(11):
            con.execute("INSERT INTO series VALUES (?, ?, 'x', ?)", [run_id, step, float(amount * step + r)])
            con.execute("INSERT INTO series VALUES (?, ?, 'y', ?)", [run_id, step, float(100 * r + 0.1 * amount)])


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    """Path of the fake DuckDB file (written once, then only opened read-only)."""
    path = tmp_path_factory.mktemp("data") / "fake.duckdb"
    con = duckdb.connect(str(path))
    fill(con)
    con.close()
    return path


@pytest.fixture()
def toolbox(db_path):
    """A Toolbox on the fake database, opened read-only."""
    source = DataSource(DataSpec(database=db_path))
    yield Toolbox(source, SETTINGS)
    source.close()
