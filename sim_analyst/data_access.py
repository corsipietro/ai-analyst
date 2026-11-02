"""Read-only access to the results, whatever their format.

Supported sources (config.yaml, section 'data'):
- database: one DuckDB file, opened with read_only=True (the frozen bench database is never written);
- tables:   one Parquet or CSV file per table, loaded into an in-memory database (the files are only read).

After opening, external access is switched off and the configuration is locked: a query cannot
read other files on the computer, write files, or load extensions.
"""
from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from pathlib import Path

import duckdb

from .config import DataSpec

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
READERS = {".parquet": "read_parquet", ".csv": "read_csv_auto"}


def quote_identifier(name: str) -> str:
    """Return a table or column name quoted for SQL ("name"), refusing anything unusual."""
    if not IDENTIFIER.match(name):
        raise ValueError(f"unsupported name {name!r}: use letters, digits and '_' only.")
    return f'"{name}"'


def lock_down(con: duckdb.DuckDBPyConnection) -> None:
    """Forbid access to files and extensions from now on, and lock the settings.

    Two connections to the same database file in one process share their settings: if another
    connection has already locked them, there is nothing left to do.
    """
    if con.execute("SELECT current_setting('lock_configuration')").fetchone()[0]:
        return
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")


def open_connection(spec: DataSpec) -> duckdb.DuckDBPyConnection:
    """Open a read-only DuckDB connection on the data described by spec."""
    if spec.database is not None:
        if not Path(spec.database).exists():
            raise FileNotFoundError(f"database not found: {spec.database}")
        con = duckdb.connect(str(spec.database), read_only=True)
    else:
        con = duckdb.connect(":memory:")
        for name, path in spec.tables.items():
            reader = READERS.get(Path(path).suffix.lower())
            if reader is None:
                raise ValueError(f"{path}: only {sorted(READERS)} files are supported.")
            literal = str(path).replace("'", "''")
            con.execute(f"CREATE TABLE {quote_identifier(name)} AS SELECT * FROM {reader}('{literal}')")
    lock_down(con)
    return con


class DataSource:
    """One read-only connection shared by all tools, used by one call at a time."""

    def __init__(self, spec: DataSpec):
        """Open the connection described by spec."""
        self.spec = spec
        self._con = open_connection(spec)
        self._lock = threading.Lock()   # the graph may run two tool calls in parallel

    @property
    def name(self) -> str:
        """A short name for the data (file name of the database, or the table names)."""
        if self.spec.database is not None:
            return Path(self.spec.database).name
        return ", ".join(sorted(self.spec.tables))

    @contextmanager
    def cursor(self):
        """Yield a cursor on the shared connection; only one caller at a time."""
        with self._lock:
            cur = self._con.cursor()
            try:
                yield cur
            finally:
                cur.close()

    def tables(self) -> list[str]:
        """Return the names of the tables and views, sorted."""
        with self.cursor() as cur:
            return sorted(name for (name,) in cur.execute("SHOW TABLES").fetchall())

    def columns(self, table: str) -> dict[str, str]:
        """Return {column name: DuckDB type} for one table, in table order."""
        with self.cursor() as cur:
            rows = cur.execute(f"DESCRIBE {quote_identifier(table)}").fetchall()
        return {row[0]: row[1] for row in rows}

    def close(self) -> None:
        """Close the connection."""
        self._con.close()
