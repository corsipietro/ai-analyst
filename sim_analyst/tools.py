"""Generic analysis tools: they work on any results database, with the columns given as arguments.

Nothing here knows what the simulated system is. The analyst (the language model) first reads
describe_database, then names the columns it needs:
- a column by name ('<column>') or as table.column ('<table>.<column>');
- a value inside a JSON text column as column.path ('config.model.rate', 'config.items[0].size').
Columns of another table are joined automatically through the columns the two tables share.

Every tool returns a plain dict (it travels as JSON). On a problem it returns
{'error': message, 'error_type': kind} with kind in: column, value, argument, sql.
Mixed rows (rows of different categories put in one group) are refused with a 'column' error.
The messages say what to do next, because the model reads them.
"""
from __future__ import annotations

import datetime
import functools
import json
import math
import re
from dataclasses import dataclass

import duckdb

from . import calculator, stats
from .data_access import DataSource, quote_identifier

# Version history of the tools: CHANGELOG.md.
TOOLS_VERSION = "g5"
MAX_TIMES = 20               # values accepted in 'at' by summarize_results

DATE_TYPE = re.compile(r"^(DATE|TIMESTAMP.*)$")
NUMERIC_TYPE = re.compile(r"^(U?(TINY|SMALL|BIG|HUGE)?INT(EGER)?|FLOAT|DOUBLE|REAL|DECIMAL.*|NUMERIC.*)$")
TEXT_TYPE = re.compile(r"^(VARCHAR|JSON|TEXT|STRING)$")
REFERENCE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*|\[\d+\])*$")
TOKEN = re.compile(r"\.?[A-Za-z_]\w*|\[\d+\]")
MAX_LISTED_VALUES = 20       # distinct values listed by describe_database
MAX_JSON_DOCUMENTS = 1000    # distinct JSON documents read to map their paths
MAX_GROUPS = 20              # groups returned by describe_trends
MAX_ITEMS = 10               # items of each list in check_quality
PERIODS = ("day", "week", "month")           # finest first
LINK_SHARE = 0.95            # share of a column's distinct values a unique column must contain to be a link target
NUMBER_SHARE = 0.8           # share of numeric-looking values that makes a text column "numbers stored as text"
NUMBER_TEXT = r"^\s*[-+]?\d+([.,]\d+)?\s*$"
# The one place where a column NAME is used (never to decide how to aggregate): only to report possible
# duplicates in columns that are probably identifiers. Deliberately English-only and short.
IDENTIFIER_NAME = re.compile(r"(^|_)(id|ids|code|key|uuid)(_|$)")
INTEGER_TYPE = re.compile(r"^U?(TINY|SMALL|BIG|HUGE)?INT(EGER)?$")
CATEGORY_MAX_VALUES = 20     # a text column with at most this many distinct values (and repeating) is a category
MEASURE_NAME_SHARE = 0.5     # share of rows that must become duplicates when a category is removed
OPERATORS = ("between", ">=", ">", "<", "<=", "is_null")   # the forms a filter value can take besides plain values


class ToolError(Exception):
    """A problem the model can fix: kind is column, value, argument or sql."""

    def __init__(self, kind: str, message: str):
        """Store the kind of problem and the message shown to the model."""
        super().__init__(message)
        self.kind = kind


def returns_errors(method):
    """Turn ToolError and DuckDB errors raised by a tool into {'error', 'error_type'} results."""
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except ToolError as exc:
            return {"error": str(exc), "error_type": exc.kind}
        except duckdb.Error as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "error_type": "sql"}
    return wrapper


def as_list(value) -> list:
    """None -> [], a single value -> [value], a list stays a list."""
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def deep_clean(obj):
    """Apply stats.clean to every number inside nested dicts and lists."""
    if isinstance(obj, dict):
        return {k: deep_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [deep_clean(v) for v in obj]
    return stats.clean(obj)


def flatten_json(obj, prefix: str):
    """Yield (path as written in the tools, leaf value) for every leaf of a parsed JSON document."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if re.match(r"^[A-Za-z_]\w*$", str(key)):
                yield from flatten_json(value, f"{prefix}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from flatten_json(value, f"{prefix}[{i}]")
    else:
        yield prefix, obj


@dataclass(frozen=True)
class Ref:
    """A column reference resolved against the schema: table, column and optional JSON path."""

    text: str            # as written by the analyst, used as the column name in results
    table: str
    column: str
    path: str | None     # '$.a[0].b' for a value inside a JSON column, None for a plain column


@dataclass(frozen=True)
class Selection:
    """The FROM and WHERE parts of a query over one main table and the tables joined to it."""

    from_sql: str
    where_sql: str
    params: list
    aliases: dict        # table name -> alias in the query (t0 is the main table)
    joins: tuple = ()    # (table, alias, key columns) of the tables the call needs


class Toolbox:
    """The analysis tools, bound to one read-only data source."""

    def __init__(self, source: DataSource, settings: dict):
        """Read the schema once; take row limits and bootstrap settings from the run settings."""
        self.source = source
        self.max_rows = settings["limits"]["max_rows"]
        self.max_points = settings["limits"]["max_points"]
        self.bootstrap = settings["bootstrap"]
        self.schema = {t: source.columns(t) for t in source.tables()}
        self._numeric = {}
        self._json_paths = None   # filled when first needed
        self._profiles = None     # (table, column) -> (rows, non-null values, distinct values)
        self._measure_names = {}  # (table, column, value column) -> bool
        self._links = None        # (table, column) -> [(table, column, orphans)] of the unique columns it points to

    # ------------------------------------------------------------------ what the columns hold

    def profiles(self) -> dict:
        """(rows, non-null values, distinct values) of every column of every table, read once."""
        if self._profiles is None:
            found = {}
            for table, columns in self.schema.items():
                aggregates = ", ".join(f"COUNT({quote_identifier(c)}), COUNT(DISTINCT {quote_identifier(c)})"
                                       for c in columns)
                with self.source.cursor() as cur:
                    n_rows = cur.execute(f"SELECT COUNT(*) FROM {quote_identifier(table)}").fetchone()[0]
                    counts = cur.execute(f"SELECT {aggregates} FROM {quote_identifier(table)}").fetchone()
                for i, c in enumerate(columns):
                    found[(table, c)] = (n_rows, counts[2 * i], counts[2 * i + 1])
            self._profiles = found
        return self._profiles

    def is_unique(self, table: str, column: str) -> bool:
        """True if every non-null value of the column appears in exactly one row."""
        _, non_null, distinct = self.profiles()[(table, column)]
        return non_null > 0 and distinct == non_null

    def is_json_column(self, table: str, column: str) -> bool:
        """True if the text column holds JSON documents (describe_database lists their paths)."""
        return bool(TEXT_TYPE.match(self.schema[table][column])) and any(
            p.startswith(f"{column}.") or p.startswith(f"{column}[") for p in self.json_paths())

    def is_category(self, table: str, column: str) -> bool:
        """True for a text column that repeats and has few distinct values (not an identifier, not JSON)."""
        _, _, distinct = self.profiles()[(table, column)]
        return (bool(TEXT_TYPE.match(self.schema[table][column])) and not self.is_json_column(table, column)
                and not self.is_unique(table, column) and distinct <= CATEGORY_MAX_VALUES)

    def is_measure_name(self, table: str, column: str, value_column: str | None) -> bool:
        """True if the category column names WHAT is measured in each row (the metric column of a long table).

        Over the key-like columns of the table (all but the value, the column itself, the columns unique per
        row and the non-integer numbers), removing the column must make at least MEASURE_NAME_SHARE of the rows
        duplicates of another row: the same entity and time then appear once per measure.
        """
        cache = self._measure_names.setdefault((table, column, value_column), None)
        if cache is None:
            keys = [c for c, kind in self.schema[table].items()
                    if c not in (column, value_column) and not self.is_unique(table, c)
                    and (self.kind_of(table, c) != "number" or INTEGER_TYPE.match(kind))]
            found = False
            if keys:
                key_sql = ", ".join(quote_identifier(c) for c in keys)
                with self.source.cursor() as cur:
                    repeated, rows = cur.execute(
                        f"SELECT COALESCE(SUM(n) FILTER (WHERE n > 1), 0), SUM(n) FROM "
                        f"(SELECT COUNT(*) AS n FROM {quote_identifier(table)} GROUP BY {key_sql})").fetchone()
                found = bool(rows) and repeated / rows >= MEASURE_NAME_SHARE
            self._measure_names[(table, column, value_column)] = found
            cache = found
        return cache

    def kind_of(self, table: str, column: str) -> str | None:
        """'number', 'text' or 'date' for a column that can be compared with a column of the same kind."""
        declared = self.schema[table][column]
        for kind, pattern in (("number", NUMERIC_TYPE), ("text", TEXT_TYPE), ("date", DATE_TYPE)):
            if pattern.match(declared):
                return kind
        return None

    def links(self) -> dict:
        """For each column, the unique columns of OTHER tables that contain at least LINK_SHARE of its distinct values.

        {(table, column): [(target table, target column, orphans)]}, orphans = distinct values not found in
        the target. The names of the two columns may differ (stazione_partenza -> stazione_id).
        """
        if self._links is None:
            profiles = self.profiles()
            targets = [key for key in profiles if self.is_unique(*key)]
            found = {}
            for (table, column), (_, non_null, distinct) in profiles.items():
                kind = self.kind_of(table, column)
                if non_null == 0 or kind is None or self.is_json_column(table, column):
                    continue
                for other, other_column in targets:
                    if other == table or self.kind_of(other, other_column) != kind:
                        continue
                    if profiles[(other, other_column)][1] < LINK_SHARE * distinct:
                        continue
                    q, q2 = quote_identifier(column), quote_identifier(other_column)
                    with self.source.cursor() as cur:
                        orphans = cur.execute(
                            f"SELECT COUNT(DISTINCT {q}) FROM {quote_identifier(table)} WHERE {q} IS NOT NULL AND {q} "
                            f"NOT IN (SELECT {q2} FROM {quote_identifier(other)} WHERE {q2} IS NOT NULL)").fetchone()[0]
                    if distinct - orphans >= LINK_SHARE * distinct:
                        found.setdefault((table, column), []).append((other, other_column, orphans))
            self._links = {key: sorted(value, key=lambda x: (x[2], x[0], x[1])) for key, value in found.items()}
        return self._links

    # ------------------------------------------------------------------ column references

    def _examples(self) -> str:
        """A short reminder of how to write references, with names taken from the data."""
        table = next(iter(self.schema))
        text = f"e.g. '{next(iter(self.schema[table]))}' or '{table}.{next(iter(self.schema[table]))}'"
        paths = self.json_paths()
        if paths:
            text += f"; a value inside a JSON column: '{next(iter(paths))}'"
        return text

    def _all_columns(self) -> str:
        """The columns of every table, as a compact text for error messages."""
        return "; ".join(f"{t}: {', '.join(cols)}" for t, cols in self.schema.items())

    def parse(self, text, main: str | None) -> Ref:
        """Resolve one column reference written by the analyst (see the module docstring)."""
        if not isinstance(text, str) or not REFERENCE.match(text.strip()):
            raise ToolError("column", f"{text!r} is not a column reference ({self._examples()}).")
        tokens = TOKEN.findall(text.strip())
        head = tokens[0]
        if head in self.schema and len(tokens) > 1 and tokens[1].startswith("."):
            table, column, rest = head, tokens[1][1:], tokens[2:]
            if column not in self.schema[table]:
                raise ToolError("column", f"table {table} has no column {column!r}. Columns: {self._all_columns()}.")
        elif main is not None and head in self.schema[main]:
            table, column, rest = main, head, tokens[1:]
        else:
            owners = [t for t, cols in self.schema.items() if head in cols]
            if not owners:
                raise ToolError("column", f"unknown column {head!r}. Columns: {self._all_columns()}. "
                                          f"References: {self._examples()}.")
            if len(owners) > 1:
                raise ToolError("column", f"column {head!r} is in several tables {owners}: write table.{head}.")
            table, column, rest = owners[0], head, tokens[1:]
        path = "$" + "".join(rest) if rest else None
        if path and not TEXT_TYPE.match(self.schema[table][column]):
            raise ToolError("column", f"{table}.{column} is {self.schema[table][column]}, not a JSON text "
                                      f"column, so {text!r} cannot have an inner path.")
        return Ref(text.strip(), table, column, path)

    def main_table(self, value: str, table: str | None) -> str:
        """The table the value column belongs to (given, or deduced from the value reference)."""
        if table is not None:
            if table not in self.schema:
                raise ToolError("column", f"unknown table {table!r}. Tables: {list(self.schema)}.")
            return table
        return self.parse(value, None).table

    def is_numeric(self, ref: Ref) -> bool:
        """True if the reference holds numbers (plain numeric column, or JSON values that all parse as numbers)."""
        if ref.path is None:
            return bool(NUMERIC_TYPE.match(self.schema[ref.table][ref.column]))
        key = (ref.table, ref.column, ref.path)
        if key not in self._numeric:
            raw = f"json_extract_string({quote_identifier(ref.column)}, '{ref.path}')"
            with self.source.cursor() as cur:
                present, numbers = cur.execute(
                    f"SELECT COUNT({raw}), COUNT(TRY_CAST({raw} AS DOUBLE)) FROM {quote_identifier(ref.table)}"
                ).fetchone()
            if present == 0:
                known = [p for p in self.json_paths() if p.startswith(f"{ref.column}.") or p.startswith(f"{ref.column}[")]
                raise ToolError("column", f"{ref.text!r} is NULL in every row: this JSON path does not exist. "
                                          f"Existing paths: {known[:30]}.")
            self._numeric[key] = present == numbers
        return self._numeric[key]

    def expression(self, ref: Ref, sel: Selection) -> str:
        """The SQL expression of a reference inside a selection (JSON numbers are cast to DOUBLE)."""
        base = f"{sel.aliases[ref.table]}.{quote_identifier(ref.column)}"
        if ref.path is None:
            return base
        extracted = f"json_extract_string({base}, '{ref.path}')"
        return f"TRY_CAST({extracted} AS DOUBLE)" if self.is_numeric(ref) else extracted

    def join_keys(self, main: str, table: str) -> list[str]:
        """The columns shared by the main table and another table, if the other table has one row per key."""
        keys = [c for c in self.schema[main] if c in self.schema[table]]
        if not keys:
            raise ToolError("column", f"tables {main} and {table} share no column, so they cannot be "
                                      "combined here: use query_results with an explicit JOIN.")
        key_list = ", ".join(quote_identifier(k) for k in keys)
        with self.source.cursor() as cur:
            duplicated = cur.execute(f"SELECT COUNT(*) FROM (SELECT {key_list} FROM {quote_identifier(table)} "
                                     "GROUP BY ALL HAVING COUNT(*) > 1)").fetchone()[0]
        if duplicated:
            raise ToolError("column", f"table {table} has several rows per ({', '.join(keys)}): joining it to "
                                      f"{main} would repeat rows. Use query_results with an explicit JOIN.")
        return keys

    def join_condition(self, main: str, table: str, alias: str) -> str:
        """The ON clause joining a table to the main table (t0) through their shared columns."""
        keys = self.join_keys(main, table)
        return " AND ".join(f"t0.{quote_identifier(k)} = {alias}.{quote_identifier(k)}" for k in keys)

    def select(self, main: str, refs: list[Ref], filters: dict, join_related: bool = False) -> Selection:
        """Build FROM (main table + LEFT JOINed tables) and WHERE (filters) for the given references.

        Rows of the main table without a match keep NULL in the joined columns (they are not dropped). With
        join_related, every other table that can be joined safely is joined too (after the needed ones, so
        the aliases of the needed tables do not change): used by require_not_mixed.
        """
        others = []
        for ref in refs + list(filters):
            if ref.table != main and ref.table not in others:
                others.append(ref.table)
        aliases = {main: "t0", **{t: f"t{i + 1}" for i, t in enumerate(others)}}
        parts = [f"{quote_identifier(main)} AS t0"]
        joins = []
        for table in others:
            parts.append(f"LEFT JOIN {quote_identifier(table)} AS {aliases[table]} "
                         f"ON {self.join_condition(main, table, aliases[table])}")
            joins.append((table, aliases[table], tuple(self.join_keys(main, table))))
        if join_related:
            for table in self.schema:
                if table in aliases:
                    continue
                alias = f"t{len(aliases)}"
                try:
                    on = self.join_condition(main, table, alias)
                except ToolError:
                    continue
                aliases[table] = alias
                parts.append(f"LEFT JOIN {quote_identifier(table)} AS {alias} ON {on}")
        sel = Selection(" ".join(parts), "TRUE", [], aliases)
        conditions, params = [], []
        for ref, values in filters.items():
            expr = self.expression(ref, sel)
            if isinstance(values, dict):
                self.operator_conditions(ref, expr, values, conditions, params)
                continue
            numeric = self.is_numeric(ref)
            try:
                typed = [float(v) if numeric else str(v) for v in values]
            except (TypeError, ValueError):
                raise ToolError("value", f"filter {ref.text!r} holds numbers, got {values}.")
            conditions.append(f"{expr} IN ({', '.join('?' for _ in typed)})")
            params.extend(typed)
        return Selection(sel.from_sql, " AND ".join(conditions) or "TRUE", params, aliases, tuple(joins))

    def operator_conditions(self, ref: Ref, expr: str, spec: dict, conditions: list, params: list) -> None:
        """Add the SQL conditions of a filter written with operators (already checked by parse_operators)."""
        dated = self.is_date(ref)
        left = f"CAST({expr} AS DATE)" if dated else expr
        marker = "CAST(? AS DATE)" if dated else "?"
        for op, operand in spec.items():
            if op == "is_null":
                conditions.append(f"{expr} IS {'' if operand else 'NOT '}NULL")
            elif op == "between":
                conditions.append(f"{left} BETWEEN {marker} AND {marker}")
                params.extend(operand)
            else:
                conditions.append(f"{left} {op} {marker}")
                params.append(operand)

    def parse_operators(self, ref: Ref, spec: dict) -> dict:
        """Check a filter written with operators and convert its operands (numbers, or dates as YYYY-MM-DD)."""
        known = f"Operators: {{\"between\": [a, b]}}, {{\">=\": x}} (also >, <, <=) and {{\"is_null\": true or false}}."
        unknown = [op for op in spec if op not in OPERATORS]
        if unknown or not spec:
            raise ToolError("argument", f"filter {ref.text!r}: unknown operator(s) {unknown or 'none given'}. {known}")
        parsed = {}
        for op, operand in spec.items():
            if op == "is_null":
                if not isinstance(operand, bool):
                    raise ToolError("argument", f"filter {ref.text!r}: is_null must be true or false, got {operand!r}.")
                parsed[op] = operand
                continue
            if not (self.is_numeric(ref) or self.is_date(ref)):
                raise ToolError("argument", f"filter {ref.text!r}: {op} needs a numeric or DATE/TIMESTAMP column; "
                                            "for a text column give the values (or is_null).")
            operands = operand if op == "between" else [operand]
            if op == "between" and not (isinstance(operand, (list, tuple)) and len(operand) == 2):
                raise ToolError("argument", f"filter {ref.text!r}: between needs two values [a, b].")
            try:
                converted = [self.day_of(x, op) if self.is_date(ref) else float(x) for x in operands]
            except (TypeError, ValueError):
                raise ToolError("argument", f"filter {ref.text!r}: {op} needs numbers (dates as YYYY-MM-DD), "
                                            f"got {operand!r}.")
            parsed[op] = converted if op == "between" else converted[0]
        return parsed

    def parse_filters(self, filters, main: str) -> dict:
        """Turn {'column': value, [values] or {operator: operand}} into {Ref: list of values or dict of operators}."""
        if filters is None:
            return {}
        if not isinstance(filters, dict):
            raise ToolError("argument", 'filters must be an object like {"column": value} or {"column": [v1, v2]}.')
        parsed = {}
        for key, values in filters.items():
            ref = self.parse(key, main)
            if isinstance(values, dict):
                parsed[ref] = self.parse_operators(ref, values)
                continue
            listed = as_list(values)
            if values is None or any(v is None for v in listed):
                raise ToolError("argument", f"filter {key!r} holds None: to select NULL (or not NULL) values write "
                                            f'{{"{key}": {{"is_null": true}}}} (or false).')
            if not listed:
                raise ToolError("argument", f"filter {key!r} has no values.")
            parsed[ref] = listed
        return parsed

    # ------------------------------------------------------------------ checks shared by the tools

    def available_values(self, refs: list[Ref], main: str) -> dict:
        """Distinct values (at most 20) of some references over the whole data, for error messages."""
        found = {}
        for ref in refs:
            sel = self.select(main, [ref], {})
            with self.source.cursor() as cur:
                rows = cur.execute(f"SELECT DISTINCT {self.expression(ref, sel)} AS x FROM {sel.from_sql} "
                                   f"ORDER BY x LIMIT {MAX_LISTED_VALUES + 1}").fetchall()
            found[ref.text] = [stats.clean(r[0]) for r in rows[:MAX_LISTED_VALUES]]
        return found

    def require_rows(self, sel: Selection, filters: dict, main: str) -> None:
        """Raise a 'value' error, listing the values that exist, if the filters select no row."""
        with self.source.cursor() as cur:
            n = cur.execute(f"SELECT COUNT(*) FROM {sel.from_sql} WHERE {sel.where_sql}", sel.params).fetchone()[0]
        if n == 0:
            raise ToolError("value", f"no rows match the filters. Values that exist: "
                                     f"{self.available_values(list(filters), main)}.")

    def require_not_mixed(self, main: str, refs: list[Ref], filters: dict, cells: list[str], value: Ref) -> list[str]:
        """Refuse (a 'column' error, no numbers) if rows put together in one cell differ in a JSON path or a measure name.

        Returns the notes for every other category that varies inside a cell ("aggregated over ...").

        Every table that can be joined is looked at, not only the ones the call needed. Two things refuse:
        - a JSON column whose paths vary (for example the parameters of a run kept in a JSON document);
        - a "measure-name" category of the value's table (the metric column of a long table, see
          is_measure_name): averaging two different quantities would give a meaningless number.
        Any other category (a note, a name, a region...) is an attribute: the rows are aggregated over it and a
        note says so. Identifiers (unique per row, or with more than CATEGORY_MAX_VALUES values) are ignored.
        No numbers are returned when it refuses, so mixed values never reach the conversation (bench of 29/10:
        a warning alone was ignored in 2 answers out of 6).
        cells: the SQL expressions that define a cell (groups and time), valid in the main selection.
        """
        sel = self.select(main, refs, filters, join_related=True)
        plain_used = {(r.table, r.column) for r in list(refs) + list(filters) if r.path is None}
        candidates = [(table, col) for table in sel.aliases
                      for col, kind in self.schema[table].items()
                      if TEXT_TYPE.match(kind) and (table, col) not in plain_used
                      and (self.is_json_column(table, col) or self.is_category(table, col))]
        if not candidates:
            return []
        counts = ", ".join(f"COUNT(DISTINCT {sel.aliases[t]}.{quote_identifier(c)}) AS c{i}"
                           for i, (t, c) in enumerate(candidates))
        group = f" GROUP BY {', '.join(cells)}" if cells else ""
        maxima = ", ".join(f"MAX(c{i})" for i in range(len(candidates)))
        with self.source.cursor() as cur:
            result = cur.execute(f"SELECT {maxima} FROM (SELECT {counts} FROM {sel.from_sql} "
                                 f"WHERE {sel.where_sql}{group})", sel.params).fetchone()
        varying = [(t, c, m) for (t, c), m in zip(candidates, result) if m and m > 1]
        value_column = value.column if value.table == main else None
        refused = [(t, c) for t, c, _ in varying
                   if self.is_json_column(t, c) or (t == main and self.is_measure_name(main, c, value_column))]
        if refused:
            details = []
            for table, column in refused:
                if self.is_json_column(table, column):
                    details.append(f"{table}.{column} (JSON: some of its paths vary, see json_paths)")
                else:
                    with self.source.cursor() as cur:
                        values = cur.execute(f"SELECT DISTINCT {sel.aliases[table]}.{quote_identifier(column)} AS x "
                                             f"FROM {sel.from_sql} WHERE {sel.where_sql} ORDER BY x LIMIT 11",
                                             sel.params).fetchall()
                    details.append(f"{table}.{column} (values {[v[0] for v in values[:10]]})")
            raise ToolError("column", "NO RESULT: the rows that would be put together in the same group differ in "
                                      + "; ".join(details) + ". Their values must not be averaged together. "
                                      "Add each of these columns (or the JSON paths that vary) to filters, to keep one "
                                      "value, or to group_by, to keep them apart, and call again.")
        shown = [f"{c} ({m} values)" for t, c, m in varying[:3]]
        if not varying:
            return []
        more = f" and {len(varying) - 3} more" if len(varying) > 3 else ""
        return ["aggregated over " + ", ".join(shown) + more]

    def join_notes(self, main: str, sel: Selection, refs: list[Ref], filters: dict, groups: list[Ref],
                   where: str, params: list) -> list[str]:
        """One note per joined table when some selected rows of the main table have no match in it."""
        notes = []
        for table, alias, keys in sel.joins:
            used = [r for r in [*refs, *filters] if r.table == table]
            unmatched = f"{where} AND {alias}.{quote_identifier(keys[0])} IS NULL"
            with self.source.cursor() as cur:
                n = cur.execute(f"SELECT COUNT(*) FROM {sel.from_sql} WHERE {unmatched}", params).fetchone()[0]
                if not n:
                    continue
                examples = cur.execute(f"SELECT DISTINCT {', '.join(f't0.{quote_identifier(k)}' for k in keys)} "
                                       f"FROM {sel.from_sql} WHERE {unmatched} ORDER BY 1 LIMIT 3", params).fetchall()
            shown = ", ".join("/".join(str(stats.clean(x)) for x in row) for row in examples)
            fate = "they are in the NULL group" if any(r in groups for r in used) else "they are included"
            notes.append(f"{n} rows of {main} have no match in {table}.{used[0].column} (e.g. {shown}): {fate}.")
        return notes

    def link_keys(self, main: str) -> list[list[str]]:
        """The sets of columns that link the main table to each other table (their shared columns)."""
        keys = []
        for table in self.schema:
            shared = [c for c in self.schema[main] if table != main and c in self.schema[table]]
            if shared and shared not in keys:
                keys.append(shared)
        return keys

    def unit_candidates(self, main: str, skip: set) -> list[list[str]]:
        """The columns of the main table that may identify a unit: shared columns and columns that link to another table.

        skip: (table, column) pairs to leave out (the columns the groups are made of).
        """
        keys = [k for k in self.link_keys(main)]
        for (table, column) in self.links():
            if table == main and [column] not in keys:
                keys.append([column])
        return [k for k in keys if not any((main, c) in skip for c in k)]

    def repeated_key(self, main: str, sel: Selection, v: Ref, where: str, params: list,
                     cells: list[str] | None = None, skip: set = frozenset()):
        """Find rows that repeat inside a group: (key columns, most rows for one key value, varying columns) or None.

        The groups are the cells (SQL expressions of the group columns and time); without cells, all the
        selected rows are one group. A key is a shared column (a column that links the main table to the
        others) or a column that links to another table: several rows with the same key value in a group
        (for example all the times of one run) are not independent observations. The varying columns are
        the other columns of the main table that take several values within those rows.
        """
        for key in self.unit_candidates(main, skip):
            key_sql = ", ".join([*(cells or []), *(f"t0.{quote_identifier(c)}" for c in key)])
            others = [c for c in self.schema[main] if c not in key and not (v.table == main and v.column == c
                                                                            and v.path is None)]
            counts = "".join(f", COUNT(DISTINCT t0.{quote_identifier(c)}) AS c{i}" for i, c in enumerate(others))
            query = (f"SELECT COUNT(*) AS n{counts} FROM {sel.from_sql} WHERE {where} "
                     f"AND {self.expression(v, sel)} IS NOT NULL GROUP BY {key_sql}")
            with self.source.cursor() as cur:
                rows = cur.execute(f"SELECT MAX(n){''.join(f', MAX(c{i})' for i in range(len(others)))} "
                                   f"FROM ({query})", params).fetchone()
            if not rows or not rows[0] or rows[0] < 2:
                continue
            return key, rows[0], [f"{main}.{c}" for c, m in zip(others, rows[1:]) if m and m > 1]
        return None

    @staticmethod
    def repetition_note(found, with_time: bool = False) -> str:
        """The note that tells the model rows repeat, and what to do (pass unit, or fix the varying columns)."""
        key, most, varying = found
        columns = ", ".join(key)
        return (f"rows repeat: a group holds up to {most} rows for the same {columns}, so its rows are not "
                f"independent observations (columns that vary within those rows: {varying or 'none found'}). "
                f"Pass unit={json.dumps(key)} to average per {columns} first, or fix the varying columns with "
                "filters" + (", or with time + at." if with_time else "."))

    def last_time(self, sel: Selection, time_expr: str):
        """The largest time value among the selected rows."""
        with self.source.cursor() as cur:
            return cur.execute(f"SELECT MAX({time_expr}) FROM {sel.from_sql} WHERE {sel.where_sql}",
                               sel.params).fetchone()[0]

    # ------------------------------------------------------------------ the tools

    def json_paths(self) -> dict:
        """Map every JSON text column to its paths: {'col.a[0].b': sorted distinct leaf values}."""
        if self._json_paths is None:
            paths = {}
            for table, columns in self.schema.items():
                for column, kind in columns.items():
                    if not TEXT_TYPE.match(kind):
                        continue
                    with self.source.cursor() as cur:
                        docs = cur.execute(f"SELECT DISTINCT {quote_identifier(column)} FROM {quote_identifier(table)} "
                                           f"WHERE {quote_identifier(column)} IS NOT NULL LIMIT {MAX_JSON_DOCUMENTS}").fetchall()
                    for (doc,) in docs:
                        try:
                            parsed = json.loads(doc)
                        except (TypeError, ValueError):
                            break   # not a JSON column
                        if not isinstance(parsed, (dict, list)):
                            break
                        for path, leaf in flatten_json(parsed, column):
                            paths.setdefault(path, set()).add(json.dumps(leaf) if isinstance(leaf, (dict, list)) else leaf)
            self._json_paths = paths
        return self._json_paths

    @returns_errors
    def describe_database(self) -> dict:
        """Tables, columns, types, row counts, distinct values, numeric ranges, JSON paths, sample rows."""
        tables = {}
        for table, columns in self.schema.items():
            q_table = quote_identifier(table)
            names = list(columns)
            ranges = [c for c in names if NUMERIC_TYPE.match(columns[c]) or DATE_TYPE.match(columns[c])]
            minmax = ", ".join(f"MIN({quote_identifier(c)}), MAX({quote_identifier(c)})" for c in ranges)
            with self.source.cursor() as cur:
                n_rows = cur.execute(f"SELECT COUNT(*) FROM {q_table}").fetchone()[0]
                limits = list(cur.execute(f"SELECT {minmax} FROM {q_table}").fetchone()) if ranges else []
                sample = cur.execute(f"SELECT * FROM {q_table} LIMIT 3").fetchall()
            info = {}
            for c in names:
                _, non_null, distinct = self.profiles()[(table, c)]
                entry = {"type": columns[c], "distinct": distinct, "nulls": n_rows - non_null,
                         "role": "unique per row" if self.is_unique(table, c) else "repeats"}
                if c in ranges:
                    entry["min"], entry["max"] = limits.pop(0), limits.pop(0)
                if self.links().get((table, c)):
                    entry["links"] = [{"to": f"{t}.{col}", "orphans": orphans}
                                      for t, col, orphans in self.links()[(table, c)]]
                info[c] = entry
            for c in names:
                is_json = any(p.startswith(f"{c}.") or p.startswith(f"{c}[") for p in self.json_paths()) \
                    and TEXT_TYPE.match(columns[c])
                if is_json:
                    info[c]["json"] = True
                elif info[c]["distinct"] <= MAX_LISTED_VALUES:
                    with self.source.cursor() as cur:
                        rows = cur.execute(f"SELECT DISTINCT {quote_identifier(c)} AS x FROM {q_table} ORDER BY x").fetchall()
                    info[c]["values"] = [r[0] for r in rows]
            tables[table] = {
                "rows": n_rows,
                "columns": info,
                "sample_rows": [[v[:200] + "..." if isinstance(v, str) and len(v) > 200 else v for v in row]
                                for row in sample],
            }
        json_paths = {}
        for path, leaves in self.json_paths().items():
            entry = {"distinct": len(leaves)}
            numbers = [v for v in leaves if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if len(leaves) <= MAX_LISTED_VALUES:
                entry["values"] = sorted(leaves, key=lambda v: (0, v, "") if isinstance(v, (int, float)) and not isinstance(v, bool) else (1, 0, str(v)))
            elif len(numbers) == len(leaves):
                entry["min"], entry["max"] = min(numbers), max(numbers)
            json_paths[path] = entry
        names = list(self.schema)
        shared = {f"{a} & {b}": [c for c in self.schema[a] if c in self.schema[b]]
                  for i, a in enumerate(names) for b in names[i + 1:]}
        return deep_clean({
            "data": self.source.name,
            "tables": tables,
            "json_paths": json_paths,
            "shared_columns": {k: v for k, v in shared.items() if v},
            "how_to_refer_to_columns": (
                "In summarize_results, describe_trends and compare_groups write a column by name or as "
                "table.column, and a value inside a JSON column as column.path, exactly as listed in "
                f"json_paths ({self._examples()}). Other tables are joined automatically through "
                "shared_columns. In query_results use json_extract_string(column, '$.path') and CAST it "
                "to a number before sorting or comparing."),
        })

    @returns_errors
    def calculate(self, expression: str) -> dict:
        """Compute an arithmetic expression exactly (see calculator.py); the result has 12 significant digits."""
        try:
            result = calculator.calculate(expression)
        except ValueError as exc:
            raise ToolError("argument", str(exc)) from None
        return {"expression": expression, "result": stats.round_significant(result)}

    @returns_errors
    def query_results(self, sql: str) -> dict:
        """Run ONE read-only SELECT (or WITH ... SELECT); return at most max_rows rows, with warnings."""
        text = (sql or "").strip().rstrip(";").strip()
        if not text.lower().startswith(("select", "with")):
            raise ToolError("sql", "only one SELECT (or WITH ... SELECT) query is allowed.")
        if ";" in text:
            raise ToolError("sql", "one query at a time: remove the ';' inside the text.")
        with self.source.cursor() as cur:
            cursor = cur.execute(text)
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchmany(self.max_rows + 1)
        truncated = len(rows) > self.max_rows
        result = {"columns": columns, "rows": [stats.clean_row(r) for r in rows[:self.max_rows]],
                  "truncated": truncated}
        if truncated:
            result["warning"] = (f"result CUT at {self.max_rows} rows: the rows after these are missing. "
                                 "Do not analyze it. Aggregate, filter, or use summarize_results / describe_trends.")
        elif not rows:
            result["warning"] = ("0 rows. Before concluding that data are missing, check your filters and "
                                 "values with SELECT DISTINCT and describe_database.")
        else:
            null_cols = [c for i, c in enumerate(columns) if all(r[i] is None for r in rows)]
            if null_cols:
                result["warning"] = (f"columns {null_cols} are all NULL: possibly a wrong column or JSON path; "
                                     "check json_paths in describe_database before concluding that "
                                     "values are missing.")
        return result

    def time_value(self, t: Ref, x):
        """A time given by the model as the database compares it: a float for a numeric time, as written otherwise."""
        if self.is_numeric(t):
            try:
                return float(x)
            except (TypeError, ValueError):
                raise ToolError("value", f"a time for {t.text!r} must be a number, got {x!r}.")
        return x

    @returns_errors
    def summarize_results(self, value: str, group_by=None, time: str | None = None, at=None,
                          filters: dict | None = None, table: str | None = None, unit=None) -> dict:
        """Count, mean, standard deviation, min, max, rows, NULLs and sum of one numeric column, per group and time.

        With unit, the values of each unit are averaged first and the statistics are over the units.
        """
        main = self.main_table(value, table)
        v = self.parse(value, main)
        groups = [self.parse(g, main) for g in as_list(group_by)]
        units = [self.parse(u, main) for u in as_list(unit)]
        t = self.parse(time, main) if time else None
        filt = self.parse_filters(filters, main)
        if at is not None and t is None:
            raise ToolError("argument", "'at' needs 'time': the column that holds the time.")
        if len(as_list(at)) > MAX_TIMES:
            raise ToolError("argument", f"'at' holds {len(as_list(at))} values: at most {MAX_TIMES} are allowed. "
                                        "Give fewer times, or use describe_trends for a whole series.")
        sel = self.select(main, [v, *groups, *units, *([t] if t else [])], filt)
        if not self.is_numeric(v):
            raise ToolError("column", f"{value!r} is not numeric: choose a numeric column as value.")
        self.require_rows(sel, filt, main)
        notes = []
        exprs = [self.expression(g, sel) for g in groups]
        where, params = sel.where_sql, list(sel.params)
        if t is not None:
            t_expr = self.expression(t, sel)
            exprs.append(t_expr)
            times = as_list(at)
            if not times:
                times = [self.last_time(sel, t_expr)]
                notes.append(f"'at' not given: summarized at the last time, {stats.clean(times[0])}.")
            times = [self.time_value(t, x) for x in times]
            where += f" AND {t_expr} IN ({', '.join('?' for _ in times)})"
            params += times
        inner = ", ".join(f"{e} AS k{i}" for i, e in enumerate(exprs))
        inner += (", " if inner else "") + "".join(f"{self.expression(u, sel)} AS u{i}, " for i, u in enumerate(units))
        inner += f"{self.expression(v, sel)} AS v"
        keys = ", ".join(f"k{i}" for i in range(len(exprs)))
        head = keys + ", " if keys else ""
        order = f" GROUP BY {keys} ORDER BY {keys}" if keys else ""
        rows_sql = f"SELECT {inner} FROM {sel.from_sql} WHERE {where}"
        if units:
            unit_keys = ", ".join(f"u{i}" for i in range(len(units)))
            per_unit = (f"SELECT {head}AVG(v) AS m, COUNT(*) AS n_rows, COUNT(*) - COUNT(v) AS n_null, SUM(v) AS s, "
                        f"COUNT(v) AS n_values FROM ({rows_sql}) GROUP BY {head}{unit_keys}")
            query = (f"SELECT {head}COUNT(m), AVG(m), STDDEV_SAMP(m), MIN(m), MAX(m), SUM(n_rows), SUM(n_null), SUM(s), "
                     f"MIN(n_values) FILTER (WHERE n_values > 0), MAX(n_values) FROM ({per_unit}){order}")
        else:
            query = (f"SELECT {head}COUNT(v), AVG(v), STDDEV_SAMP(v), MIN(v), MAX(v), COUNT(*), COUNT(*) - COUNT(v), "
                     f"SUM(v) FROM ({rows_sql}){order}")
        with self.source.cursor() as cur:
            rows = cur.execute(query, params).fetchmany(self.max_rows + 1)
        if t is not None and at is not None:
            found = {str(stats.clean(r[len(exprs) - 1])) for r in rows}
            missing = [x for x in as_list(at) if str(stats.clean(self.time_value(t, x))) not in found]
            if missing:
                notes.append(f"no rows at times {missing}.")
        used = [v, *groups, *units, *([t] if t else [])]
        notes += self.require_not_mixed(main, used, filt, exprs, v)
        notes += self.join_notes(main, sel, used, filt, groups, where, params)
        if not units:
            skip = {(g.table, g.column) for g in groups if g.path is None}
            repeated = self.repeated_key(main, sel, v, where, params, cells=exprs, skip=skip)
            if repeated:
                notes.append(self.repetition_note(repeated, with_time=t is not None))
        names = ["n_units" if units else "n", "mean", "std", "min", "max", "n_rows", "n_null", "sum"]
        result = {"value": v.text,
                  "columns": [g.text for g in groups] + ([t.text] if t else []) + names
                  + (["rows_per_unit_min", "rows_per_unit_max"] if units else []),
                  "rows": [stats.clean_row(r) for r in rows[:self.max_rows]],
                  "truncated": len(rows) > self.max_rows}
        if units:
            result["unit"] = [u.text for u in units]
            result["explanation"] = ("the values of each unit are averaged first; n_units, mean, std, min and max are "
                                     "over the units; n_rows and n_null count rows; sum is over all the values.")
        if notes:
            result["notes"] = notes
        return result

    def is_date(self, ref: Ref) -> bool:
        """True if the reference is a plain DATE or TIMESTAMP column."""
        return ref.path is None and bool(DATE_TYPE.match(self.schema[ref.table][ref.column]))

    @staticmethod
    def bucket(t_expr: str, period: str) -> str:
        """SQL cutting a date or timestamp to the first day of its day, week (Monday) or month."""
        return f"CAST(date_trunc('{period}', {t_expr}) AS DATE)"

    @staticmethod
    def day_of(text, name: str) -> str:
        """A date written as YYYY-MM-DD (any later part is ignored), or an 'argument' error."""
        try:
            return datetime.date.fromisoformat(str(text)[:10]).isoformat()
        except ValueError:
            raise ToolError("argument", f"{name} must be a date like 2026-01-31 for a date time column, got {text!r}.")

    def count_periods(self, sel: Selection, t_expr: str, where: str, params: list, period: str) -> int:
        """How many distinct periods the selected rows fall in."""
        with self.source.cursor() as cur:
            return cur.execute(f"SELECT COUNT(DISTINCT {self.bucket(t_expr, period)}) FROM {sel.from_sql} "
                               f"WHERE {where} AND {t_expr} IS NOT NULL", params).fetchone()[0]

    @staticmethod
    def dates_in_summary(summary: dict, origin: datetime.date) -> dict:
        """Turn the day numbers of a trend summary (days since origin) into ISO dates; the slope is per day."""
        def iso(days) -> str:
            return (origin + datetime.timedelta(days=int(days))).isoformat()
        for key in ("initial", "final", "max_of_mean", "min_of_mean"):
            summary[key]["time"] = iso(summary[key]["time"])
        slope = summary["slope_last_part"]
        slope["from_time"], slope["to_time"] = iso(slope["from_time"]), iso(slope["to_time"])
        slope["slope_per_day"] = slope.pop("slope_per_time_unit")
        for turn in summary["turning_points_sampled"]:
            turn["time"] = iso(turn["time"])
        for row in summary["series_sampled"]["rows"]:
            row[0] = iso(row[0])
        return summary

    @returns_errors
    def describe_trends(self, value: str, time: str, group_by=None, filters: dict | None = None,
                        step: float | None = None, start=None, end=None, table: str | None = None,
                        period: str | None = None) -> dict:
        """Per group, how the mean over the observations of a numeric column evolves in time (see stats.trend_summary).

        The time is a number (series sampled every `step`) or a DATE / TIMESTAMP (one point per `period`).
        """
        main = self.main_table(value, table)
        v = self.parse(value, main)
        t = self.parse(time, main)
        groups = [self.parse(g, main) for g in as_list(group_by)]
        filt = self.parse_filters(filters, main)
        sel = self.select(main, [v, t, *groups], filt)
        dated = self.is_date(t)
        if not self.is_numeric(v):
            raise ToolError("column", f"value {value!r} is not numeric: choose a numeric column as value.")
        if not (dated or self.is_numeric(t)):
            raise ToolError("column", f"time {time!r} is not numeric, DATE or TIMESTAMP: choose a column that holds "
                                      "numbers or dates as time.")
        if period is not None and period not in PERIODS:
            raise ToolError("argument", f"period must be one of {list(PERIODS)}, got {period!r}.")
        self.require_rows(sel, filt, main)
        exprs = [self.expression(g, sel) for g in groups]
        t_expr = self.expression(t, sel)
        where, params = sel.where_sql, list(sel.params)
        notes = []
        for bound, name, sign in ((start, "start", ">="), (end, "end", "<=")):
            if bound is None:
                continue
            if dated:
                where += f" AND CAST({t_expr} AS DATE) {sign} CAST(? AS DATE)"
                params.append(self.day_of(bound, name))
            else:
                where += f" AND {t_expr} {sign} ?"
                params.append(float(bound))
        if dated:
            return self.date_trends(main, sel, v, t, groups, filt, exprs, t_expr, where, params, period, step)
        if period is not None:
            notes.append("period ignored: the time column is numeric (use step).")
        inner = "".join(f"{e} AS k{i}, " for i, e in enumerate(exprs)) + f"{t_expr} AS t, {self.expression(v, sel)} AS v"
        keys = "".join(f"k{i}, " for i in range(len(exprs)))
        with self.source.cursor() as cur:
            rows = cur.execute(f"SELECT {keys}t, COUNT(v), AVG(v), MIN(v), MAX(v) FROM "
                               f"(SELECT {inner} FROM {sel.from_sql} WHERE {where} AND v IS NOT NULL AND t IS NOT NULL) "
                               f"GROUP BY {keys}t ORDER BY {keys}t", params).fetchall()
        if not rows:
            raise ToolError("value", "no rows in the requested time range.")
        series = {}
        for r in rows:
            series.setdefault(tuple(r[:len(exprs)]), []).append(tuple(r[len(exprs):]))
        if len(series) > MAX_GROUPS:
            raise ToolError("argument", f"{len(series)} groups: too many. Use filters to keep at most {MAX_GROUPS}.")
        span = max(r[len(exprs)] for r in rows) - min(r[len(exprs)] for r in rows)
        if step is None:
            step = stats.nice_step(span)
        elif step <= 0:
            raise ToolError("argument", "step must be positive.")
        if span / step + 1 > self.max_points:
            step = stats.nice_step(span, self.max_points - 1)
            notes.append(f"step enlarged to {step} to keep at most {self.max_points} points per group.")
        out = [{"group": dict(zip([g.text for g in groups], key)), **stats.trend_summary(points, step)}
               for key, points in series.items()]
        result = {"value": v.text, "time": t.text, "step": step,
                  "explanation": ("mean, min and max are over the observations (rows) of each group at each time; "
                                  "slope_last_part is a least-squares slope of the mean over the last 20% of the "
                                  "time range, per unit of time; shape and turning points use the sampled series."),
                  "groups": out}
        notes += self.require_not_mixed(main, [v, t, *groups], filt, exprs + [t_expr], v)
        notes += self.join_notes(main, sel, [v, t, *groups], filt, groups, where, params)
        if notes:
            result["notes"] = notes
        return deep_clean(result)

    def date_trends(self, main, sel, v, t, groups, filt, exprs, t_expr, where, params, period, step) -> dict:
        """describe_trends for a DATE or TIMESTAMP time: one point per period (day, week or month)."""
        notes = []
        if step is not None:
            notes.append("step ignored: the time column is a date (use period).")
        if period is None:   # the finest period that keeps at most max_points points; else the coarsest
            for period in PERIODS:
                n_periods = self.count_periods(sel, t_expr, where, params, period)
                if n_periods <= self.max_points:
                    break
        else:
            n_periods = self.count_periods(sel, t_expr, where, params, period)
        if n_periods == 0:
            raise ToolError("value", "no rows in the requested time range.")
        bucket = self.bucket(t_expr, period)
        inner = "".join(f"{e} AS k{i}, " for i, e in enumerate(exprs)) + f"{bucket} AS t, {self.expression(v, sel)} AS v"
        keys = "".join(f"k{i}, " for i in range(len(exprs)))
        with self.source.cursor() as cur:
            rows = cur.execute(f"SELECT {keys}t, COUNT(v), AVG(v), MIN(v), MAX(v) FROM "
                               f"(SELECT {inner} FROM {sel.from_sql} WHERE {where} AND v IS NOT NULL AND {t_expr} IS NOT NULL) "
                               f"GROUP BY {keys}t ORDER BY {keys}t", params).fetchall()
        if not rows:
            raise ToolError("value", "no rows in the requested time range.")
        origin = min(r[len(exprs)] for r in rows)
        series = {}
        for r in rows:   # the time becomes the number of days since the first period
            series.setdefault(tuple(r[:len(exprs)]), []).append(((r[len(exprs)] - origin).days, *r[len(exprs) + 1:]))
        if len(series) > MAX_GROUPS:
            raise ToolError("argument", f"{len(series)} groups: too many. Use filters to keep at most {MAX_GROUPS}.")
        every = math.ceil(n_periods / self.max_points)
        if every > 1:
            notes.append(f"{n_periods} {period}s are more than {self.max_points} points: series_sampled keeps one "
                         f"{period} in {every} and the last; the other values use every {period}.")
        out = []
        for key, points in series.items():
            sampled = points[::every] + ([points[-1]] if (len(points) - 1) % every else [])
            summary = stats.trend_summary(points, 1, sampled=sampled)
            out.append({"group": dict(zip([g.text for g in groups], key)), **self.dates_in_summary(summary, origin)})
        result = {"value": v.text, "time": t.text, "period": period,
                  "explanation": (f"each point is one {period} (the date of its first day): mean, min and max are over "
                                  "the observations (rows) in it; slope_last_part is a least-squares slope of the mean "
                                  "over the last 20% of the time range, per day; shape and turning points use the "
                                  "sampled series."),
                  "groups": out}
        notes += self.require_not_mixed(main, [v, t, *groups], filt, exprs + [bucket], v)
        notes += self.join_notes(main, sel, [v, t, *groups], filt, groups, where, params)
        if notes:
            result["notes"] = notes
        return deep_clean(result)

    def verdict(self, a_s: list[float], b_s: list[float]) -> dict:
        """The statistical part of compare_groups: Welch's interval decides, the bootstrap only adds.

        With fewer than 2 values in a group there is no verdict. The percentile bootstrap is
        reported (never deciding) only if both groups have at least bootstrap.min_n values,
        because with fewer it is too narrow.
        """
        if min(len(a_s), len(b_s)) < 2:
            return {"verdict": "cannot be assessed: at least 2 observations per group are needed"}
        confidence = self.bootstrap["confidence"]
        welch = stats.welch_interval(a_s, b_s, confidence)
        out = {"method": "welch", "difference_b_minus_a": welch["difference"], "confidence": confidence,
               "assumption": "Welch's t interval assumes values roughly normal within each group."}
        if welch["interval"] is None:
            out["verdict"] = "cannot be assessed: no variability in either group, so no interval can be computed"
            return out
        out["interval"] = welch["interval"]
        out["degrees_of_freedom"] = welch["degrees_of_freedom"]
        out["verdict"] = "established" if welch["interval"][0] > 0 or welch["interval"][1] < 0 else "not established"
        if min(len(a_s), len(b_s)) >= self.bootstrap["min_n"]:
            boot = stats.bootstrap_difference(a_s, b_s, self.bootstrap["resamples"], self.bootstrap["seed"],
                                              confidence)
            out["bootstrap_interval"] = boot["interval"]
        return out

    @returns_errors
    def compare_groups(self, value: str, group_by: str, a, b, at=None, time: str | None = None,
                       filters: dict | None = None, table: str | None = None, unit=None) -> dict:
        """Difference of means between two groups (b minus a), with Welch's interval over the observations.

        With unit, the values of each unit are averaged first and Welch's interval is over the units. Without
        unit, if the rows of a group repeat for a key (several rows per customer, say), the means are given
        but the verdict is "none": the rows are not independent.
        """
        if isinstance(group_by, (list, tuple)):
            if len(group_by) != 1:
                raise ToolError("argument", "compare_groups compares two values of ONE column: give one "
                                            "group_by column and fix the other columns with filters.")
            group_by = group_by[0]
        main = self.main_table(value, table)
        v = self.parse(value, main)
        g = self.parse(group_by, main)
        units = [self.parse(u, main) for u in as_list(unit)]
        t = self.parse(time, main) if time else None
        filt = self.parse_filters(filters, main)
        if at is not None and t is None:
            raise ToolError("argument", "'at' needs 'time': the column that holds the time.")
        sel = self.select(main, [v, g, *units, *([t] if t else [])], filt)
        if not self.is_numeric(v):
            raise ToolError("column", f"{value!r} is not numeric: choose a numeric column as value.")
        self.require_rows(sel, filt, main)
        numeric_group = self.is_numeric(g)
        try:
            a_val, b_val = (float(a), float(b)) if numeric_group else (str(a), str(b))
        except (TypeError, ValueError):
            raise ToolError("value", f"{group_by!r} holds numbers: a and b must be numbers, got {a!r}, {b!r}.")
        g_expr = self.expression(g, sel)
        where, params = sel.where_sql + f" AND {g_expr} = ?", list(sel.params)
        notes = []
        if isinstance(at, (list, tuple)):
            if len(at) != 1:
                raise ToolError("argument", "compare_groups compares at ONE time: give a single value for 'at'.")
            at = at[0]
        if t is not None:
            t_expr = self.expression(t, sel)
            if at is None:
                at = self.last_time(sel, t_expr)
                notes.append(f"'at' not given: compared at the last time, {stats.clean(at)}.")
            where += f" AND {t_expr} = ?"
        samples, rows_per_unit = {}, {}
        for label, group_value in (("a", a_val), ("b", b_val)):
            extra = [group_value] + ([self.time_value(t, at)] if t is not None else [])
            value_sql = self.expression(v, sel)
            if units:   # one mean per unit
                unit_sql = ", ".join(f"{self.expression(u, sel)} AS u{i}" for i, u in enumerate(units))
                query = (f"SELECT AVG(v), COUNT(v) FROM (SELECT {value_sql} AS v, {unit_sql} FROM {sel.from_sql} "
                         f"WHERE {where}) GROUP BY {', '.join(f'u{i}' for i in range(len(units)))}")
            else:
                query = f"SELECT {value_sql}, 1 FROM {sel.from_sql} WHERE {where}"
            with self.source.cursor() as cur:
                rows = cur.execute(query, params + extra).fetchall()
            samples[label] = [r[0] for r in rows if r[0] is not None]
            rows_per_unit[label] = [r[1] for r in rows if r[0] is not None]
            if not samples[label]:
                raise ToolError("value", f"no rows for {group_by} = {group_value!r}"
                                         + (f" at {time} = {at}" if t is not None else "")
                                         + f". Values that exist: {self.available_values([g], main)}.")
        a_s, b_s = samples["a"], samples["b"]
        result = {"value": v.text, "group_by": g.text, "at": at if t is not None else None}
        if units:
            result["unit"] = [u.text for u in units]
        for label, group_value in (("a", a_val), ("b", b_val)):
            s = samples[label]
            result[label] = {"group": group_value, "n_units" if units else "n": len(s), "mean": sum(s) / len(s),
                             "min": min(s), "max": max(s)}
            if units:
                result[label]["rows_per_unit"] = [min(rows_per_unit[label]), max(rows_per_unit[label])]
        result["ranges_overlap"] = max(min(a_s), min(b_s)) <= min(max(a_s), max(b_s))
        repeated = None
        if not units:
            skip = {(g.table, g.column)} if g.path is None else set()
            for group_value in (a_val, b_val):
                extra = [group_value] + ([self.time_value(t, at)] if t is not None else [])
                repeated = repeated or self.repeated_key(main, sel, v, where, params + extra, skip=skip)
        if repeated:
            notes.append(self.repetition_note(repeated, with_time=t is not None))
            result["verdict"] = "none"
            result["verdict_reason"] = (f"the rows of a group repeat for the same {', '.join(repeated[0])}, so they are "
                                        "not independent and Welch's interval would be too narrow. Pass "
                                        f"unit={json.dumps(repeated[0])} to compare per unit, or fix the varying columns.")
        else:
            result.update(self.verdict(a_s, b_s))
        notes += self.require_not_mixed(main, [v, g, *units, *([t] if t else [])], filt,
                                        [g_expr] + ([self.expression(t, sel)] if t else []), v)
        if notes:
            result["notes"] = notes
        return deep_clean(result)

    # ------------------------------------------------------------------ data quality

    @staticmethod
    def cap(items: list, label: str, notes: list) -> list:
        """The first MAX_ITEMS items; if there are more, a note says how many are shown."""
        if len(items) > MAX_ITEMS:
            notes.append(f"{label}: showing {MAX_ITEMS} of {len(items)}.")
        return items[:MAX_ITEMS]

    @staticmethod
    def label_of(key) -> str:
        """A group (the values of the group_by columns) as a short text."""
        return " | ".join("NULL" if x is None else str(stats.clean(x)) for x in key)

    def null_report(self, sel: Selection, table: str, groups: list[Ref], t: Ref | None, period: str | None,
                    columns: list[str], out: dict, notes: list) -> None:
        """NULL share per group, per period, and the runs of at least 3 periods in which a column is all NULL."""
        group_sql = [self.expression(g, sel) for g in groups]
        counts = "".join(f", COUNT(*) - COUNT(t0.{quote_identifier(c)})" for c in columns)
        if groups and not t:
            with self.source.cursor() as cur:
                rows = cur.execute(f"SELECT {', '.join(group_sql)}, COUNT(*){counts} FROM {sel.from_sql} "
                                   f"WHERE {sel.where_sql} GROUP BY {', '.join(group_sql)}", sel.params).fetchall()
            n_keys = len(groups)
            shares = {}
            for c_index, c in enumerate(columns):
                by_group = sorted(((self.label_of(r[:n_keys]), r[n_keys + 1 + c_index] / r[n_keys]) for r in rows),
                                  key=lambda x: (-x[1], x[0]))
                shares[c] = dict(self.cap(by_group, f"null_share_by_group.{c}", notes))
            out["null_share_by_group"] = shares
        if not t:
            return
        t_expr = self.expression(t, sel)
        bucket = self.bucket(t_expr, period) if period else t_expr
        key_sql = ", ".join([*group_sql, bucket])
        with self.source.cursor() as cur:
            rows = cur.execute(f"SELECT {key_sql}, COUNT(*){counts} FROM {sel.from_sql} "
                               f"WHERE {sel.where_sql} AND {t_expr} IS NOT NULL GROUP BY {key_sql}",
                               sel.params).fetchall()
        n_keys = len(groups)
        periods = sorted({r[n_keys] for r in rows})
        position = {p: i for i, p in enumerate(periods)}
        cells = {}                                    # (group label, period position) -> (rows, nulls per column)
        for r in rows:
            cells[(self.label_of(r[:n_keys]), position[r[n_keys]])] = (r[n_keys + 1], r[n_keys + 2:])
        group_labels = sorted({label for label, _ in cells})
        if groups:
            shares = {}
            for c_index, c in enumerate(columns):
                by_group = {}
                for label in group_labels:
                    rows_n = sum(cells[(label, i)][0] for i in range(len(periods)) if (label, i) in cells)
                    nulls = sum(cells[(label, i)][1][c_index] for i in range(len(periods)) if (label, i) in cells)
                    by_group[label] = nulls / rows_n if rows_n else 0
                ordered = sorted(by_group.items(), key=lambda x: (-x[1], x[0]))
                shares[c] = dict(self.cap(ordered, f"null_share_by_group.{c}", notes))
            out["null_share_by_group"] = shares
        per_period, runs = {}, []
        for c_index, c in enumerate(columns):
            totals = []
            for i, p in enumerate(periods):
                present = [cells[(label, i)] for label in group_labels if (label, i) in cells]
                rows_n, nulls = sum(x[0] for x in present), sum(x[1][c_index] for x in present)
                if nulls:
                    totals.append((p, nulls / rows_n))
            top = sorted(totals, key=lambda x: (-x[1], x[0]))
            kept = sorted(self.cap(top, f"null_share_by_period.{c}", notes))
            per_period[c] = {str(stats.clean(p)): share for p, share in kept}
            for label in group_labels:
                run = []
                for i in range(len(periods) + 1):
                    cell = cells.get((label, i))
                    if cell and cell[0] and cell[1][c_index] == cell[0]:
                        run.append(i)
                        continue
                    if len(run) >= 3:
                        runs.append({"group": label, "column": c, "start": str(stats.clean(periods[run[0]])),
                                     "end": str(stats.clean(periods[run[-1]])), "periods": len(run)})
                    run = []
        out["null_share_by_period"] = per_period
        if not groups:
            for run in runs:
                del run["group"]
        runs.sort(key=lambda r: (-r["periods"], r.get("group", ""), r["column"], r["start"]))
        if runs:
            out["null_runs"] = self.cap(runs, "null_runs", notes)

    def text_columns(self, table: str) -> list[str]:
        """The plain text columns of a table (JSON documents excluded)."""
        return [c for c, kind in self.schema[table].items()
                if TEXT_TYPE.match(kind) and not self.is_json_column(table, c)]

    @returns_errors
    def check_quality(self, table: str, group_by=None, time: str | None = None, period: str | None = None,
                      filters: dict | None = None) -> dict:
        """A compact data-quality report on one table: NULLs, runs of NULLs, duplicates, text variants, orphans.

        The model decides what a NULL or a duplicate means; this only reports the facts.
        """
        if table not in self.schema:
            raise ToolError("column", f"unknown table {table!r}. Tables: {list(self.schema)}.")
        groups = [self.parse(g, table) for g in as_list(group_by)]
        t = self.parse(time, table) if time else None
        filt = self.parse_filters(filters, table)
        for ref in [*groups, *([t] if t else []), *filt]:
            if ref.table != table:
                raise ToolError("column", f"{ref.text!r} is a column of table {ref.table}: check_quality looks at "
                                          f"one table at a time, here {table}.")
        if period is not None and period not in PERIODS:
            raise ToolError("argument", f"period must be one of {list(PERIODS)}, got {period!r}.")
        if t is not None and not (self.is_date(t) or self.is_numeric(t)):
            raise ToolError("column", f"{time!r} must be a numeric, DATE or TIMESTAMP column to be used as time.")
        sel = self.select(table, [*groups, *([t] if t else [])], filt)
        self.require_rows(sel, filt, table)
        notes = []
        columns = list(self.schema[table])
        q = {c: f"t0.{quote_identifier(c)}" for c in columns}
        with self.source.cursor() as cur:
            counts = cur.execute(f"SELECT COUNT(*){''.join(f', COUNT({q[c]})' for c in columns)} FROM {sel.from_sql} "
                                 f"WHERE {sel.where_sql}", sel.params).fetchone()
        n_rows = counts[0]
        out = {"table": table, "rows": n_rows}
        if n_rows == 0:
            return out
        null_counts = [(c, n_rows - n) for c, n in zip(columns, counts[1:]) if n < n_rows]
        null_counts.sort(key=lambda x: -x[1])
        listed = self.cap(null_counts, "nulls", notes)
        if listed:
            out["nulls"] = {c: {"count": n, "share": n / n_rows} for c, n in listed}
        dated = t is not None and self.is_date(t)
        if t is not None and not dated and period is not None:
            notes.append("period ignored: the time column is numeric.")
        if dated:
            period = period or "day"
            out["period"] = period
        if listed and (groups or t is not None):
            self.null_report(sel, table, groups, t, period if dated else None, [c for c, _ in listed], out, notes)
        self.duplicate_report(sel, table, columns, out, notes)
        self.text_report(sel, table, out, notes)
        orphans = []
        for (link_table, column), targets in self.links().items():
            for other, other_column, n_orphans in targets:
                if link_table == table and n_orphans:
                    qc, qo = quote_identifier(column), quote_identifier(other_column)
                    with self.source.cursor() as cur:
                        examples = cur.execute(
                            f"SELECT DISTINCT {qc} AS x FROM {quote_identifier(table)} WHERE {qc} IS NOT NULL AND "
                            f"{qc} NOT IN (SELECT {qo} FROM {quote_identifier(other)} WHERE {qo} IS NOT NULL) "
                            "ORDER BY x LIMIT 3").fetchall()
                    orphans.append({"column": column, "to": f"{other}.{other_column}", "orphans": n_orphans,
                                    "examples": [e[0] for e in examples]})
        if orphans:
            out["orphans"] = self.cap(orphans, "orphans", notes)
        if notes:
            out["notes"] = notes
        return deep_clean(out)

    def duplicate_report(self, sel: Selection, table: str, columns: list[str], out: dict, notes: list) -> None:
        """Exact duplicate rows, and repeated values in the columns whose name says they are identifiers.

        Columns that link to another table are left out: a customer id repeats in the rentals by nature.
        """
        rows_sql = f"SELECT t0.* FROM {sel.from_sql} WHERE {sel.where_sql}"
        repeated_rows = f"SELECT *, COUNT(*) AS copies__ FROM ({rows_sql}) GROUP BY ALL HAVING COUNT(*) > 1"
        with self.source.cursor() as cur:
            extra, groups = cur.execute(f"SELECT COALESCE(SUM(copies__ - 1), 0), COUNT(*) FROM ({repeated_rows})",
                                        sel.params).fetchone()
            if extra:
                examples = cur.execute(f"SELECT * EXCLUDE (copies__) FROM ({repeated_rows}) LIMIT 3", sel.params)
                names = [d[0] for d in examples.description]
                out["duplicate_rows"] = {"extra_rows": extra, "groups": groups,
                                         "examples": [dict(zip(names, stats.clean_row(r))) for r in examples.fetchall()]}
        found = []
        for c in columns:
            if not IDENTIFIER_NAME.search(c.lower()) or self.is_json_column(table, c):
                continue
            if (table, c) in self.links():   # a column that points to another table repeats by nature
                continue
            q = f"t0.{quote_identifier(c)}"
            with self.source.cursor() as cur:
                repeated = cur.execute(f"SELECT {q} AS x, COUNT(*) AS c FROM {sel.from_sql} WHERE {sel.where_sql} AND "
                                       f"{q} IS NOT NULL GROUP BY x HAVING COUNT(*) > 1 ORDER BY c DESC, x",
                                       sel.params).fetchall()
            if repeated:
                found.append({"column": c, "duplicated_values": len(repeated),
                              "extra_rows": sum(n - 1 for _, n in repeated),
                              "examples": [x for x, _ in repeated[:3]]})
        if found:
            out["identifier_duplicates"] = self.cap(found, "identifier_duplicates", notes)

    def text_report(self, sel: Selection, table: str, out: dict, notes: list) -> None:
        """Text values equal after trim + lower, and text columns whose values are mostly numbers."""
        variants, as_numbers = [], []
        for c in self.text_columns(table):
            q = f"t0.{quote_identifier(c)}"
            base = f"SELECT {q} AS x FROM {sel.from_sql} WHERE {sel.where_sql} AND {q} IS NOT NULL"
            with self.source.cursor() as cur:
                groups = cur.execute(f"SELECT list(DISTINCT x ORDER BY x) FROM ({base}) GROUP BY lower(trim(x)) "
                                     "HAVING COUNT(DISTINCT x) > 1 ORDER BY lower(trim(x))", sel.params).fetchall()
                total, numbers, commas = cur.execute(
                    f"SELECT COUNT(*), COUNT(*) FILTER (WHERE regexp_matches(x, '{NUMBER_TEXT}')), "
                    f"COUNT(*) FILTER (WHERE regexp_matches(x, '{NUMBER_TEXT}') AND strpos(x, ',') > 0) FROM ({base})",
                    sel.params).fetchone()
                others = []
                if total and numbers / total >= NUMBER_SHARE:
                    others = cur.execute(f"SELECT DISTINCT x FROM ({base}) WHERE NOT regexp_matches(x, '{NUMBER_TEXT}') "
                                         "ORDER BY x LIMIT 3", sel.params).fetchall()
            variants += [{"column": c, "values": g[0][:MAX_ITEMS]} for g in groups]
            if total and numbers / total >= NUMBER_SHARE:
                as_numbers.append({"column": c, "share": numbers / total, "decimal_comma": commas > 0,
                                   "not_numbers": [o[0] for o in others]})
        if variants:
            out["text_variants"] = self.cap(variants, "text_variants", notes)
        if as_numbers:
            out["numbers_as_text"] = self.cap(as_numbers, "numbers_as_text", notes)
