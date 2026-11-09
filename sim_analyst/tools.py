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

# g5.1-c/d/e (1-2/11): metric label and mixed-category warning (c); x instead of time (d); any x: irregular, decimal,
# date or timestamp (hour and year periods), thresholds in settings.yaml (analysis), numbers equal within a tolerance,
# numeric parameters of other tables noted, identifiers found from the values (e).
# g5 (31/10): the model declares, the code verifies. describe_database adds nulls, role (unique per row / repeats)
# and links (columns of other tables that contain the column's values, with the orphans); the mixing check only
# looks at text columns that are categories (not identifiers); the g4 refusal of several rows per unit is gone:
# summarize_results and compare_groups take 'unit' (average per unit first) and, without it, add a note on repeated
# rows (compare_groups then gives no verdict); describe_trends accepts a DATE/TIMESTAMP x with 'period';
# new tool check_quality (NULLs, runs of NULLs, duplicates, text variants, numbers as text, orphans);
# summarize_results adds n_rows, n_null and sum.
# g5 fixes (31/10, before the first g5 bench): mixing refuses only a varying JSON path or a "measure-name" category
# (removing it makes at least half of the rows duplicates: the metric column of a long table); any other category
# that varies gives a note "aggregated over ..."; filters accept {"between": [a, b]}, ">=", ">", "<", "<=" and
# {"is_null": true/false}, and refuse None in plain values; joins are LEFT JOINs from the value's table (unmatched rows
# keep NULL in the joined columns, with a note); wording without domain words; the error of describe_trends says which
# argument is wrong; the repetition note mentions x + at only when there is an x column.
# g4.1 (31/10): serialization only (dates, datetimes, times, intervals, decimals, uuids and bytes become JSON-safe
# in stats.clean; date and timestamp columns get min and max in describe_database); no change on data without dates.
# g4 (30/10): compare_groups refuses (error 'column', no numbers) a group that holds several rows for the same
# value of the key linking the value's table to the other tables (e.g. all the steps of one run pooled).
# g3 (30/10): compare_groups decides from Welch's t interval (bootstrap only added from bootstrap.min_n
# values per group, never deciding); summarize_results applies the 20-values limit on 'at' and no longer
# says that each row is a replicate; query_results' all-NULL warning mentions json_paths; new tool calculate
# (safe arithmetic, calculator.py); 'observations' instead of 'replicates' in every text the model reads.
# g2 (29/10): mixed rows give an error, not a warning; g1 = first version.
TOOLS_VERSION = "g5"

# Thresholds of the analysis: the values of settings.yaml (section analysis). The copy here is only the fallback for
# settings that do not have the section; test_generic_x checks that the two are the same.
ANALYSIS_DEFAULTS = {
    "category_max_values": 20, "measure_name_share": 0.5, "link_share": 0.95, "number_share": 0.8, "max_groups": 20,
    "max_at_values": 20, "max_items": 10, "max_listed_values": 20, "trend_last_fraction": 0.2,
    "min_points_per_group": 6, "default_points": 12, "x_tolerance": 1.0e-9,
    "grid_max_share": 0.5, "identifier_min_share": 0.95,
}

DATE_TYPE = re.compile(r"^(DATE|TIMESTAMP.*)$")
NUMERIC_TYPE = re.compile(r"^(U?(TINY|SMALL|BIG|HUGE)?INT(EGER)?|FLOAT|DOUBLE|REAL|DECIMAL.*|NUMERIC.*)$")
TEXT_TYPE = re.compile(r"^(VARCHAR|JSON|TEXT|STRING)$")
REFERENCE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*|\[\d+\])*$")
TOKEN = re.compile(r"\.?[A-Za-z_]\w*|\[\d+\]")
MAX_JSON_DOCUMENTS = 1000    # distinct JSON documents read to map their paths
PERIODS = ("hour", "day", "week", "month", "year")   # finest first; hour only for a TIMESTAMP x
NUMBER_TEXT = r"^\s*[-+]?\d+([.,]\d+)?\s*$"
INTEGER_TYPE = re.compile(r"^U?(TINY|SMALL|BIG|HUGE)?INT(EGER)?$")
TIMESTAMP_TYPE = re.compile(r"^TIMESTAMP.*$")
FILTER_ARGUMENTS = {"summarize_results": ("x", "at"), "compare_groups": ("x", "at"), "describe_trends": ("x",)}
FEW_VALUES = 5   # a group that holds up to this many values of a filtered column lists them; with more, a range
OPERATORS = ("between", ">=", ">", "<", "<=", "is_null")   # the forms a filter value can take besides plain values


class ToolError(Exception):
    """A problem the model can fix: kind is column, value, argument or sql."""

    def __init__(self, kind: str, message: str):
        """Store the kind of problem and the message shown to the model."""
        super().__init__(message)
        self.kind = kind


def returns_errors(method):
    """Turn ToolError and DuckDB errors raised by a tool into {'error', 'error_type'} results.

    The sentences the tool collected in toolbox._read_as (a variant it corrected, see `Toolbox.parse`) are cleared
    at the start of every call and, if there are any, come back as the first field `read_as` of the result, also
    of an error result.
    """
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        toolbox = args[0]
        toolbox._read_as = []
        try:
            result = method(*args, **kwargs)
        except ToolError as exc:
            result = {"error": str(exc), "error_type": exc.kind}
        except duckdb.Error as exc:
            result = {"error": f"{type(exc).__name__}: {exc}", "error_type": "sql"}
        if toolbox._read_as and isinstance(result, dict):
            result = {"read_as": " ".join(toolbox._read_as), **result}
        return result
    return wrapper


def as_list(value) -> list:
    """None -> [], a single value -> [value], a list stays a list."""
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


KEY_FIELDS = ("group", "labels", "filters", "at", "x", "from_x", "to_x", "examples")   # fields that hold data values, not measures


def deep_clean(obj, digits: int = 3):
    """Apply stats.clean to every number inside nested dicts and lists.

    Numbers under the fields in KEY_FIELDS (data values) and in the leading `x` columns of a
    {"columns", "rows"} table keep stats.KEY_DIGITS significant digits; the others keep `digits`.
    """
    if isinstance(obj, dict):
        if isinstance(obj.get("columns"), list) and isinstance(obj.get("rows"), list):
            n_keys = 0
            while n_keys < len(obj["columns"]) and obj["columns"][n_keys] == "x":
                n_keys += 1
            rows = [stats.clean_row(r, n_keys) if isinstance(r, (list, tuple)) else deep_clean(r, digits)
                    for r in obj["rows"]]
            return {k: rows if k == "rows" else deep_clean(v, digits) for k, v in obj.items()}
        return {k: deep_clean(v, stats.KEY_DIGITS if k in KEY_FIELDS else digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [deep_clean(v, digits) for v in obj]
    return stats.clean(obj, digits)


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

    @property
    def qualified(self) -> str:
        """The reference with its table in front ('quantity' -> 'series.quantity'), whichever way it was written."""
        return self.text if self.text.startswith(f"{self.table}.") else f"{self.table}.{self.text}"


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
        """Read the schema once; take row limits, thresholds (analysis) and bootstrap settings from the run settings."""
        self.source = source
        self.analysis = {**ANALYSIS_DEFAULTS, **settings.get("analysis", {})}
        self.max_rows = settings["limits"]["max_rows"]
        self.max_points = settings["limits"]["max_points"]
        self.max_total_points = settings["limits"].get("max_total_points", 80)
        self.bootstrap = settings["bootstrap"]
        self.schema = {t: source.columns(t) for t in source.tables()}
        self._numeric = {}
        self._json_paths = None   # filled when first needed
        self._profiles = None     # (table, column) -> (rows, non-null values, distinct values)
        self._measure_names = {}  # (table, column, value column) -> bool
        self._links = None        # (table, column) -> [(table, column, orphans)] of the unique columns it points to
        self._read_as = []        # what the current call corrected in its arguments (see returns_errors)

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
                and not self.is_unique(table, column) and distinct <= self.analysis["category_max_values"])

    def is_measure_name(self, table: str, column: str, value_column: str | None) -> bool:
        """True if the category column names WHAT is measured in each row (the metric column of a long table).

        Over the key-like columns of the table (all but the value, the column itself and the columns unique per
        row; a non-integer number column only if it repeats a lot, at most measure_name_share distinct values per
        non-null value: a grid of x, not a continuous measure), removing the column must make at least
        measure_name_share of the rows duplicates of another row: the same entity and x then appear once per measure.
        """
        cache = self._measure_names.setdefault((table, column, value_column), None)
        if cache is None:
            keys = [c for c, kind in self.schema[table].items()
                    if c not in (column, value_column) and not self.is_unique(table, c)
                    and (self.kind_of(table, c) != "number" or INTEGER_TYPE.match(kind) or self.repeats_a_lot(table, c))]
            found = False
            if keys:
                key_sql = ", ".join(quote_identifier(c) for c in keys)
                with self.source.cursor() as cur:
                    repeated, rows = cur.execute(
                        f"SELECT COALESCE(SUM(n) FILTER (WHERE n > 1), 0), SUM(n) FROM "
                        f"(SELECT COUNT(*) AS n FROM {quote_identifier(table)} GROUP BY {key_sql})").fetchone()
                found = bool(rows) and repeated / rows >= self.analysis["measure_name_share"]
            self._measure_names[(table, column, value_column)] = found
            cache = found
        return cache

    def repeats_a_lot(self, table: str, column: str) -> bool:
        """True if the column has at most grid_max_share distinct values per non-null value (a grid, not a measure)."""
        _, non_null, distinct = self.profiles()[(table, column)]
        return non_null > 0 and distinct <= self.analysis["grid_max_share"] * non_null

    def kind_of(self, table: str, column: str) -> str | None:
        """'number', 'text' or 'date' for a column that can be compared with a column of the same kind."""
        declared = self.schema[table][column]
        for kind, pattern in (("number", NUMERIC_TYPE), ("text", TEXT_TYPE), ("date", DATE_TYPE)):
            if pattern.match(declared):
                return kind
        return None

    def links(self) -> dict:
        """For each column, the unique columns of OTHER tables that contain at least link_share of its distinct values.

        {(table, column): [(target table, target column, orphans)]}, orphans = distinct values not found in
        the target. The names of the two columns may differ (stazione_partenza -> stazione_id).
        """
        if self._links is None:
            profiles = self.profiles()
            share = self.analysis["link_share"]
            targets = [key for key in profiles if self.is_unique(*key)]
            found = {}
            for (table, column), (_, non_null, distinct) in profiles.items():
                kind = self.kind_of(table, column)
                if non_null == 0 or kind is None or self.is_json_column(table, column):
                    continue
                for other, other_column in targets:
                    if other == table or self.kind_of(other, other_column) != kind:
                        continue
                    if profiles[(other, other_column)][1] < share * distinct:
                        continue
                    q, q2 = quote_identifier(column), quote_identifier(other_column)
                    with self.source.cursor() as cur:
                        orphans = cur.execute(
                            f"SELECT COUNT(DISTINCT {q}) FROM {quote_identifier(table)} WHERE {q} IS NOT NULL AND {q} "
                            f"NOT IN (SELECT {q2} FROM {quote_identifier(other)} WHERE {q2} IS NOT NULL)").fetchone()[0]
                    if distinct - orphans >= share * distinct:
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
        """Resolve one column reference written by the analyst (see the module docstring).

        A segment made only of digits after a dot ('a.0.b') can only be a list position, so it is read as '[0]';
        the Ref then carries the corrected text and read_as says so.
        """
        if isinstance(text, str):
            written = text.strip()
            corrected = re.sub(r"\.(\d+)(?=\.|\[|$)", r"[\1]", written)
            if corrected != written and REFERENCE.match(corrected):
                text = corrected
                sentence = f"{written!r} read as {corrected!r}: list positions go in brackets."
                if sentence not in self._read_as:
                    self._read_as.append(sentence)
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
            if numeric:
                condition, found = self.numeric_equal(expr, typed)
                conditions.append(condition)
                params.extend(found)
            else:
                conditions.append(f"{expr} IN ({', '.join('?' for _ in typed)})")
                params.extend(typed)
        return Selection(sel.from_sql, " AND ".join(conditions) or "TRUE", params, aliases, tuple(joins))

    def numeric_equal(self, expr: str, values: list[float]) -> tuple[str, list]:
        """SQL (and its parameters) true where a number equals one of the values, within x_tolerance.

        abs(expr - v) <= x_tolerance * greatest(1, abs(v)): 0.30000000000000004 in the database is 0.3 for the
        model, which reads the values with 12 digits.
        """
        tolerance = repr(float(self.analysis["x_tolerance"]))
        one = (f"abs(CAST({expr} AS DOUBLE) - CAST(? AS DOUBLE)) <= {tolerance} * greatest(1.0, abs(CAST(? AS DOUBLE)))")
        return "(" + " OR ".join(one for _ in values) + ")", [v for value in values for v in (value, value)]

    def equal_condition(self, ref: Ref, expr: str, values: list) -> tuple[str, list]:
        """SQL and parameters selecting the rows where a column equals one of the values.

        A number is compared within x_tolerance; a DATE or TIMESTAMP by day (values already as YYYY-MM-DD);
        anything else exactly.
        """
        if self.is_numeric(ref):
            return self.numeric_equal(expr, values)
        if self.is_date(ref):
            return f"CAST({expr} AS DATE) IN ({', '.join('CAST(? AS DATE)' for _ in values)})", list(values)
        return f"{expr} IN ({', '.join('?' for _ in values)})", list(values)

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

    def read_filters(self, tool: str, filters, x, at, main: str):
        """Correct the variants of `filters` that have one clear reading; return (filters, x, at).

        - a key x or at, if no table has a column of that name, is the argument of the tool: it leaves the filters and
          becomes the argument (a different value for an argument already given, or an argument the tool does not
          have, is an error). In describe_trends only x exists.
        - {"equals": v} is the plain value v.
        Every correction adds a sentence to read_as.
        """
        if not isinstance(filters, dict):
            return filters, x, at
        kept, moved = {}, {}
        for key, value in filters.items():
            if tool in FILTER_ARGUMENTS and key in ("x", "at") and not any(key in cols for cols in self.schema.values()):
                if key not in FILTER_ARGUMENTS[tool]:
                    raise ToolError("argument", f"{tool} has no {key!r}: it describes the whole series along x. "
                                                "For the values at one x use summarize_results with x and at.")
                given = x if key == "x" else at
                if given is not None:
                    same = (self.parse(given, main).qualified == self.parse(value, main).qualified if key == "x"
                            else as_list(given) == as_list(value))
                    if not same:
                        raise ToolError("argument", f"{key} is given twice: as an argument ({given!r}) and inside "
                                                    f"filters ({value!r}). Give it once, as an argument.")
                moved[key] = value
            elif isinstance(value, dict) and "equals" in value:
                if len(value) > 1:
                    raise ToolError("argument", f"filter {key!r}: equals cannot be combined with other operators.")
                kept[key] = value["equals"]
                shown = json.dumps(value["equals"])
                self._read_as.append('{"equals": ' + shown + '} read as the value ' + shown + '.')
            else:
                kept[key] = value
        if moved:
            names = [k for k in ("x", "at") if k in moved]
            many = len(names) > 1
            self._read_as.append(f"{' and '.join(repr(k) for k in names)} {'are arguments' if many else 'is an argument'}, "
                                 f"not {'filters' if many else 'a filter'}: read as "
                                 f"{', '.join(f'{k}={moved[k]!r}' for k in names)}.")
            x, at = moved.get("x", x), moved.get("at", at)
        return kept, x, at

    @staticmethod
    def mixed_warning(parsed: dict, groups: list[Ref], units: list[Ref] = (), holders: set | None = None) -> dict:
        """{"warning": text} if a filter keeps 2 or more values of a column that is neither grouped nor the unit, else {}.

        holders: with groups, the filtered columns (qualified names) of which some group holds more than one value
        (see group_labels); a filter not in it mixes nothing and does not warn. None (no groups: the single group is
        the whole selection): every such filter warns.
        """
        apart = {r.qualified for r in [*groups, *units]}
        sentences = []
        for ref, values in parsed.items():
            if not isinstance(values, list) or ref.qualified in apart:
                continue
            distinct = list(dict.fromkeys(values))
            if len(distinct) < 2 or (holders is not None and ref.qualified not in holders):
                continue
            q = ref.qualified
            sentences.append(f"filter {q} keeps {len(distinct)} values ({', '.join(map(str, distinct))}) and some groups "
                             f"hold more than one: their numbers mix them (see {q} in each group). To see them apart, "
                             f"add {q} to group_by.")
        return {"warning": " ".join(sentences)} if sentences else {}

    @staticmethod
    def varied_filters(filt: dict, groups: list[Ref], units: list[Ref], t: Ref | None) -> list[Ref]:
        """The filtered columns that keep several values (a list of 2 or more, or operators other than is_null alone).

        Not the columns of x, group_by and unit: they are told apart already.
        """
        apart = {r.qualified for r in [*groups, *units, *([t] if t else [])]}
        found = []
        for ref, values in filt.items():
            if ref.qualified in apart:
                continue
            if isinstance(values, dict):
                if any(op != "is_null" for op in values):
                    found.append(ref)
            elif len(set(values)) >= 2:
                found.append(ref)
        return found

    def few_values(self, sel: Selection, where: str, params: list, group_exprs: list[str], expr: str) -> dict:
        """The first FEW_VALUES distinct non-NULL values of an expression in each group, sorted: {key: [values]}."""
        keys = "".join(f"{e} AS k{i}, " for i, e in enumerate(group_exprs))
        names = ", ".join(f"k{i}" for i in range(len(group_exprs)))
        partition = f"PARTITION BY {names} " if names else ""
        query = (f"SELECT {names + ', ' if names else ''}c FROM (SELECT DISTINCT {keys}{expr} AS c FROM {sel.from_sql} "
                 f"WHERE {where}) WHERE c IS NOT NULL QUALIFY ROW_NUMBER() OVER ({partition}ORDER BY c) <= {FEW_VALUES}")
        with self.source.cursor() as cur:
            rows = cur.execute(query, params).fetchall()
        found = {}
        for row in rows:
            found.setdefault(tuple(row[:len(group_exprs)]), []).append(row[-1])
        return found

    def filter_labels(self, sel: Selection, where: str, params: list, group_exprs: list[str], filt: dict,
                      varied: list[Ref]) -> tuple[dict, set]:
        """What the rows of each group hold of the filtered columns that keep several values.

        Returns ({group key: {qualified column: value}}, qualified names of the list filters of which some group
        holds more than one value). The value is the one the rows have, the sorted list of them (2 to FEW_VALUES),
        or {"between": [min, max]}. The groups are made by group_exprs (none: the whole selection is one group, key ()).
        """
        if not varied:
            return {}, set()
        exprs = [self.expression(r, sel) for r in varied]
        keys = ", ".join(group_exprs)
        aggregates = ", ".join(f"COUNT(DISTINCT {e}), MIN({e}), MAX({e})" for e in exprs)
        with self.source.cursor() as cur:
            rows = cur.execute(f"SELECT {keys + ', ' if keys else ''}{aggregates} FROM {sel.from_sql} WHERE {where}"
                               + (f" GROUP BY {keys}" if keys else ""), params).fetchall()
        n = len(group_exprs)
        found = {tuple(r[:n]): r[n:] for r in rows}
        labels = {key: {} for key in found}
        holders = set()
        for j, ref in enumerate(varied):
            listed = any(2 <= row[3 * j] <= FEW_VALUES for row in found.values())
            few = self.few_values(sel, where, params, group_exprs, exprs[j]) if listed else {}
            for key, row in found.items():
                count, low, high = row[3 * j: 3 * j + 3]
                if count == 0:
                    value = None
                elif count == 1:
                    value = low
                elif count <= FEW_VALUES:
                    value = few.get(key, [])
                else:
                    value = {"between": [low, high]}
                labels[key][ref.qualified] = deep_clean(value, stats.KEY_DIGITS)
                if isinstance(filt[ref], list) and count >= 2:
                    holders.add(ref.qualified)
        return labels, holders

    def parameters_of(self, table: str, key: Ref, index: int, keys: list[tuple], taken: set) -> list[tuple]:
        """The columns of a table, one row per value of `key`, that take different values in the groups.

        Returns [(qualified name, {group key: value})] in the order of the columns of the table (the paths of a JSON
        column, in the order of json_paths, take the place of the column). index: the position of the key
        column in the group keys.
        """
        refs = []
        for column in self.schema[table]:
            if column == key.column:
                continue
            if self.is_json_column(table, column):
                refs += [self.parse(f"{table}.{path}", table) for path in self.json_paths()
                         if path.startswith(f"{column}.") or path.startswith(f"{column}[")]
            else:
                refs.append(self.parse(f"{table}.{column}", table))
        alone = Selection(f"{quote_identifier(table)} AS t0", "TRUE", [], {table: "t0"})
        usable, exprs = [], []
        for ref in refs:
            if ref.qualified in taken:
                continue
            try:
                exprs.append(self.expression(ref, alone))
            except ToolError:
                continue
            usable.append(ref)
        wanted = list({k[index] for k in keys if k[index] is not None})
        if not usable or not wanted:
            return []
        key_sql = f"t0.{quote_identifier(key.column)}"
        with self.source.cursor() as cur:
            rows = cur.execute(f"SELECT {key_sql}, {', '.join(exprs)} FROM {alone.from_sql} "
                               f"WHERE {key_sql} IN ({', '.join('?' for _ in wanted)})", wanted).fetchall()
        by_value = {row[0]: row[1:] for row in rows}
        found = []
        for j, ref in enumerate(usable):
            per_key = {k: (by_value[k[index]][j] if k[index] in by_value else None) for k in keys}
            if len(set(per_key.values())) > 1:
                found.append((ref.qualified, per_key))
        return found

    def parameter_labels(self, main: str, groups: list[Ref], keys: list[tuple], taken: set) -> tuple[dict, list[str]]:
        """The parameters of the units the groups are made of: ({group key: {qualified column: value}}, notes).

        A plain column of group_by that is the key of another table (the same column name, unique there, and the table
        can be joined to the main one through that column alone) makes the other columns of that table
        (JSON paths included) parameters of the group. Only those that take different values in the groups shown are
        kept, at most max_items; the ones cut are named in a note.
        """
        found = []
        for index, g in enumerate(groups):
            if g.path is not None:
                continue
            for table in self.schema:
                if table in (main, g.table) or g.column not in self.schema[table] or not self.is_unique(table, g.column):
                    continue
                try:
                    if self.join_keys(main, table) != [g.column]:
                        continue
                except ToolError:
                    continue
                found += self.parameters_of(table, g, index, keys, taken)
        most = self.analysis["max_items"]
        labels = {k: {} for k in keys}
        for name, per_key in found[:most]:
            for k in keys:
                labels[k][name] = deep_clean(per_key[k], stats.KEY_DIGITS)
        cut = [name for name, _ in found[most:]]
        return labels, ([f"unit parameters cut to {most}: {', '.join(cut)}."] if cut else [])

    def group_labels(self, main: str, sel: Selection, where: str, params: list, filt: dict, groups: list[Ref],
                     units: list[Ref], t: Ref | None, group_exprs: list[str], keys: list[tuple],
                     known: dict) -> tuple[dict, set | None, list[str]]:
        """What each group says about itself, besides its group_by values and the single-value filters (known).

        Returns ({group key: {qualified column: value}}, the list filters of which some group holds more than one
        value, notes). Two kinds of label: the filtered columns that keep several values (filter_labels) and the
        parameters of the units (parameter_labels). Without group_by there is one group, the whole selection: no
        label, and None instead of the set (mixed_warning then warns as for a single group).
        """
        if not groups:
            return {}, None, []
        varied = self.varied_filters(filt, groups, units, t)
        from_filters, holders = self.filter_labels(sel, where, params, group_exprs, filt, varied)
        taken = {*known, *(g.qualified for g in groups), *(r.qualified for r in varied)}
        from_units, notes = self.parameter_labels(main, groups, keys, taken)
        return {k: {**from_filters.get(k, {}), **from_units.get(k, {})} for k in keys}, holders, notes

    @staticmethod
    def echo_filters(filters: dict | None, parsed: dict) -> dict:
        """The filters as the model gave them, under the table-qualified column names (parse_filters keeps the order)."""
        return {ref.qualified: raw for ref, raw in zip(parsed, (filters or {}).values())}

    @staticmethod
    def single_value_labels(filters: dict | None, parsed: dict) -> dict:
        """The filters that fix ONE value (a scalar or a one-element list), by qualified name, as the model gave them."""
        labels = {}
        for ref, raw in zip(parsed, (filters or {}).values()):
            values = parsed[ref]
            if isinstance(values, list) and len(values) == 1:
                labels[ref.qualified] = values[0]
        return labels

    def points_per_group(self, n_groups: int) -> int:
        """Points of series_sampled per group: max_points, lowered (not below min_points_per_group) to share max_total_points."""
        return min(self.max_points, max(self.analysis["min_points_per_group"], self.max_total_points // max(1, n_groups)))

    # ------------------------------------------------------------------ checks shared by the tools

    def available_values(self, refs: list[Ref], main: str) -> dict:
        """Distinct values (at most 20) of some references over the whole data, for error messages."""
        found = {}
        limit = self.analysis["max_listed_values"]
        for ref in refs:
            sel = self.select(main, [ref], {})
            with self.source.cursor() as cur:
                rows = cur.execute(f"SELECT DISTINCT {self.expression(ref, sel)} AS x FROM {sel.from_sql} "
                                   f"ORDER BY x LIMIT {limit + 1}").fetchall()
            found[ref.text] = [stats.clean(r[0], stats.KEY_DIGITS) for r in rows[:limit]]
        return found

    def require_rows(self, sel: Selection, filters: dict, main: str) -> None:
        """Raise a 'value' error, listing the values that exist, if the filters select no row."""
        with self.source.cursor() as cur:
            n = cur.execute(f"SELECT COUNT(*) FROM {sel.from_sql} WHERE {sel.where_sql}", sel.params).fetchone()[0]
        if n == 0:
            raise ToolError("value", f"no rows match the filters. Values that exist: "
                                     f"{self.available_values(list(filters), main)}.")

    def distinct_expression(self, table: str, column: str, alias: str, used: list[Ref]) -> str | None:
        """The SQL whose distinct values tell if a candidate column varies inside a cell (None: it cannot vary).

        A JSON column is counted on the paths the call does not use: a path in group_by is told apart already, and
        one in filters is the model's choice to pool (the warning says so), as for a plain column in filters.
        """
        base = f"{alias}.{quote_identifier(column)}"
        if not self.is_json_column(table, column):
            return base
        skipped = {r.path for r in used if r.table == table and r.column == column and r.path}
        if not skipped:
            return base
        rest = [p for p in self.json_paths() if (p.startswith(f"{column}.") or p.startswith(f"{column}["))
                and "$" + p[len(column):] not in skipped]
        if not rest:
            return None
        return "concat_ws('|', " + ", ".join(f"COALESCE(json_extract_string({base}, '${p[len(column):]}'), '~')"
                                            for p in rest) + ")"

    def require_not_mixed(self, main: str, refs: list[Ref], filters: dict, cells: list[str], value: Ref) -> list[str]:
        """Refuse (a 'column' error, no numbers) if rows put together in one cell differ in a JSON path or a measure name.

        Returns the notes for every other category that varies inside a cell ("aggregated over ...").

        Every table that can be joined is looked at, not only the ones the call needed. Two things refuse:
        - a JSON column whose paths vary (for example the parameters of a run kept in a JSON document);
        - a "measure-name" category of the value's table (the metric column of a long table, see
          is_measure_name): averaging two different quantities would give a meaningless number.
        Any other category (a note, a name, a region...) is an attribute: the rows are aggregated over it and a
        note says so. So is a numeric column of a joined table (not the main one: its numbers are measures) that
        repeats, with at most category_max_values values, for example a parameter kept in the table of the runs.
        Identifiers (unique per row, or with more than category_max_values values) are ignored.
        No numbers are returned when it refuses, so mixed values never reach the conversation (bench of 29/10:
        a warning alone was ignored in 2 answers out of 6).
        cells: the SQL expressions that define a cell (groups and x), valid in the main selection.
        """
        sel = self.select(main, refs, filters, join_related=True)
        plain_used = {(r.table, r.column) for r in list(refs) + list(filters) if r.path is None}
        candidates = [(table, col) for table in sel.aliases
                      for col, kind in self.schema[table].items()
                      if TEXT_TYPE.match(kind) and (table, col) not in plain_used
                      and (self.is_json_column(table, col) or self.is_category(table, col))]
        candidates += [(table, col) for table in sel.aliases if table != main
                       for col, kind in self.schema[table].items()
                       if NUMERIC_TYPE.match(kind) and col not in self.schema[main] and (table, col) not in plain_used
                       and not self.is_unique(table, col)
                       and self.profiles()[(table, col)][2] <= self.analysis["category_max_values"]]
        used = [*refs, *filters]
        distinct = {(t, c): self.distinct_expression(t, c, sel.aliases[t], used) for t, c in candidates}
        candidates = [key for key in candidates if distinct[key] is not None]
        if not candidates:
            return []
        counts = ", ".join(f"COUNT(DISTINCT {distinct[key]}) AS c{i}" for i, key in enumerate(candidates))
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
            shown = ", ".join("/".join(str(stats.clean(x, stats.KEY_DIGITS)) for x in row) for row in examples)
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

        The groups are the cells (SQL expressions of the group columns and x); without cells, all the
        selected rows are one group. A key is a shared column (a column that links the main table to the
        others) or a column that links to another table: several rows with the same key value in a group
        (for example all the steps of one run) are not independent observations. The varying columns are
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
    def repetition_note(found, with_x: bool = False) -> str:
        """The note that tells the model rows repeat, and what to do (pass unit, or fix the varying columns)."""
        key, most, varying = found
        columns = ", ".join(key)
        return (f"rows repeat: a group holds up to {most} rows for the same {columns}, so its rows are not "
                f"independent observations (columns that vary within those rows: {varying or 'none found'}). "
                f"Pass unit={json.dumps(key)} to average per {columns} first, or fix the varying columns with "
                "filters" + (", or with x + at." if with_x else "."))

    def last_time(self, sel: Selection, time_expr: str):   # time_expr: the SQL of x
        """The largest value of x among the selected rows."""
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
                elif info[c]["distinct"] <= self.analysis["max_listed_values"]:
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
            if len(leaves) <= self.analysis["max_listed_values"]:
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
        }, stats.KEY_DIGITS)

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
        result = {"columns": columns, "rows": [stats.clean_row(r, len(r)) for r in rows[:self.max_rows]],
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
        """A value of x given by the model as the database compares it: a float for a numeric x, a day (YYYY-MM-DD)
        for a DATE or TIMESTAMP x, as written otherwise."""
        if self.is_numeric(t):
            try:
                return float(x)
            except (TypeError, ValueError):
                raise ToolError("value", f"a value of x for {t.text!r} must be a number, got {x!r}.")
        if self.is_date(t):
            return self.day_of(x, f"a value of x for {t.text!r}")
        return x

    @returns_errors
    def summarize_results(self, value: str, group_by=None, x: str | None = None, at=None,
                          filters: dict | None = None, table: str | None = None, unit=None) -> dict:
        """Count, mean, standard deviation, min, max, rows, NULLs and sum of one numeric column, per group and value of x.

        With unit, the values of each unit are averaged first and the statistics are over the units.
        """
        main = self.main_table(value, table)
        filters, x, at = self.read_filters("summarize_results", filters, x, at, main)
        v = self.parse(value, main)
        groups = [self.parse(g, main) for g in as_list(group_by)]
        units = [self.parse(u, main) for u in as_list(unit)]
        t = self.parse(x, main) if x else None
        filt = self.parse_filters(filters, main)
        if at is not None and t is None:
            raise ToolError("argument", "'at' needs 'x': the column along which the data are ordered.")
        most = self.analysis["max_at_values"]
        if len(as_list(at)) > most:
            raise ToolError("argument", f"'at' holds {len(as_list(at))} values: at most {most} are allowed. "
                                        "Give fewer values, or use describe_trends for a whole series.")
        sel = self.select(main, [v, *groups, *units, *([t] if t else [])], filt)
        if not self.is_numeric(v):
            raise ToolError("column", f"{value!r} is not numeric: choose a numeric column as value.")
        self.require_rows(sel, filt, main)
        notes = []
        exprs = [self.expression(g, sel) for g in groups]
        where, params = sel.where_sql, list(sel.params)
        group_exprs = list(exprs)
        if t is not None:
            t_expr = self.expression(t, sel)
            exprs.append(f"CAST({t_expr} AS DATE)" if self.is_timestamp(t) else t_expr)   # a timestamp is grouped by day
            times = as_list(at)
            if not times:
                times = [self.last_time(sel, t_expr)]
                notes.append(f"'at' not given: summarized at the last value of x, {stats.clean(times[0], stats.KEY_DIGITS)}.")
            times = [self.time_value(t, x) for x in times]
            at_sql, at_params = self.equal_condition(t, t_expr, times)
            where += f" AND {at_sql}"
            params += at_params
            if self.is_timestamp(t):
                notes.append("at compares whole days of x.")
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
            found = [r[len(exprs) - 1] for r in rows]
            missing = [x for x in as_list(at) if not self.x_is_among(t, self.time_value(t, x), found)]
            if missing:
                notes.append(f"no rows at values of x {missing}.")
        used = [v, *groups, *units, *([t] if t else [])]
        notes += self.require_not_mixed(main, used, filt, exprs, v)
        notes += self.join_notes(main, sel, used, filt, groups, where, params)
        if not units:
            skip = {(g.table, g.column) for g in groups if g.path is None}
            repeated = self.repeated_key(main, sel, v, where, params, cells=exprs, skip=skip)
            if repeated:
                notes.append(self.repetition_note(repeated, with_x=t is not None))
        shown = rows[:self.max_rows]
        group_keys = list(dict.fromkeys(tuple(r[:len(groups)]) for r in shown))
        labels, holders, label_notes = self.group_labels(main, sel, where, params, filt, groups, units, t, group_exprs,
                                                         group_keys, {})
        notes += label_notes
        label_names = list(next(iter(labels.values()), {}))
        n_keys = len(groups) + len(label_names) + (1 if t else 0)
        table_rows = [stats.clean_row([*r[:len(groups)], *(labels[tuple(r[:len(groups)])][n] for n in label_names),
                                       *r[len(groups):]], n_keys) for r in shown]
        names = ["n_units" if units else "n", "mean", "std", "min", "max", "n_rows", "n_null", "sum"]
        result = {"value": v.text, "filters": self.echo_filters(filters, filt), "group_by": [g.qualified for g in groups],
                  **self.mixed_warning(filt, groups, units, holders),
                  "columns": [g.text for g in groups] + label_names + ([t.text] if t else []) + names
                  + (["rows_per_unit_min", "rows_per_unit_max"] if units else []),
                  "rows": table_rows,
                  "truncated": len(rows) > self.max_rows}
        if units:
            result["unit"] = [u.text for u in units]
            result["explanation"] = ("the values of each unit are averaged first; n_units, mean, std, min and max are "
                                     "over the units; n_rows and n_null count rows; sum is over all the values.")
        if notes:
            result["notes"] = notes
        return result

    def x_is_among(self, t: Ref, wanted, found: list) -> bool:
        """True if a value of x (as time_value gives it) equals one of the values read from the database."""
        tolerance = self.analysis["x_tolerance"]
        for value in found:
            if value is None:
                continue
            if self.is_numeric(t):
                if abs(float(value) - wanted) <= tolerance * max(1.0, abs(wanted)):
                    return True
            elif self.is_date(t):
                if str(value)[:10] == wanted:
                    return True
            elif value == wanted:
                return True
        return False

    def is_date(self, ref: Ref) -> bool:
        """True if the reference is a plain DATE or TIMESTAMP column."""
        return ref.path is None and bool(DATE_TYPE.match(self.schema[ref.table][ref.column]))

    def is_timestamp(self, ref: Ref) -> bool:
        """True if the reference is a plain TIMESTAMP column."""
        return ref.path is None and bool(TIMESTAMP_TYPE.match(self.schema[ref.table][ref.column]))

    def check_period(self, t: Ref, period: str | None) -> None:
        """Raise an 'argument' error for an unknown period, or for hour when x is a DATE."""
        if period is not None and period not in PERIODS:
            raise ToolError("argument", f"period must be one of {list(PERIODS)}, got {period!r}.")
        if period == "hour" and self.is_date(t) and not self.is_timestamp(t):
            raise ToolError("argument", "period hour needs a TIMESTAMP x; x is a DATE.")

    @staticmethod
    def bucket(t_expr: str, period: str) -> str:
        """SQL cutting a date or timestamp to its hour (a timestamp), or to the first day of its day, week (Monday), month or year."""
        if period == "hour":
            return f"date_trunc('hour', {t_expr})"
        return f"CAST(date_trunc('{period}', {t_expr}) AS DATE)"

    @staticmethod
    def day_of(text, name: str) -> str:
        """A date written as YYYY-MM-DD (any later part is ignored), or an 'argument' error."""
        try:
            return datetime.date.fromisoformat(str(text)[:10]).isoformat()
        except ValueError:
            raise ToolError("argument", f"{name} must be a date like 2026-01-31 for a date column, got {text!r}.")

    def count_periods(self, sel: Selection, t_expr: str, where: str, params: list, period: str) -> int:
        """How many distinct periods the selected rows fall in."""
        with self.source.cursor() as cur:
            return cur.execute(f"SELECT COUNT(DISTINCT {self.bucket(t_expr, period)}) FROM {sel.from_sql} "
                               f"WHERE {where} AND {t_expr} IS NOT NULL", params).fetchone()[0]

    @staticmethod
    def dates_in_summary(summary: dict, origin: datetime.date | datetime.datetime, period: str = "day") -> dict:
        """Turn the numbers of a trend summary (days since origin, hours for period hour) into ISO dates.

        The slope is per day, or per hour for period hour, whose x are written YYYY-MM-DDTHH:00.
        """
        hourly = period == "hour"

        def iso(n) -> str:
            if hourly:
                return (origin + datetime.timedelta(hours=int(n))).strftime("%Y-%m-%dT%H:00")
            return (origin + datetime.timedelta(days=int(n))).isoformat()
        for key in ("initial", "final", "max_of_mean", "min_of_mean"):
            summary[key]["x"] = iso(summary[key]["x"])
        slope = summary["slope_last_part"]
        slope["from_x"], slope["to_x"] = iso(slope["from_x"]), iso(slope["to_x"])
        slope["slope_per_hour" if hourly else "slope_per_day"] = slope.pop("slope_per_unit_of_x")
        for turn in summary["turning_points_sampled"]:
            turn["x"] = iso(turn["x"])
        for row in summary["series_sampled"]["rows"]:
            row[0] = iso(row[0])
        return summary

    @returns_errors
    def describe_trends(self, value: str, x: str, group_by=None, filters: dict | None = None,
                        step: float | None = None, start=None, end=None, table: str | None = None,
                        period: str | None = None) -> dict:
        """Per group, how the mean over the observations of a numeric column changes along x (see stats.trend_summary).

        x is a number (series sampled every `step`) or a DATE / TIMESTAMP (one point per `period`).
        """
        main = self.main_table(value, table)
        filters, x, _ = self.read_filters("describe_trends", filters, x, None, main)
        v = self.parse(value, main)
        t = self.parse(x, main)
        groups = [self.parse(g, main) for g in as_list(group_by)]
        filt = self.parse_filters(filters, main)
        sel = self.select(main, [v, t, *groups], filt)
        dated = self.is_date(t)
        if not self.is_numeric(v):
            raise ToolError("column", f"value {value!r} is not numeric: choose a numeric column as value.")
        if not (dated or self.is_numeric(t)):
            raise ToolError("column", f"x {x!r} is not numeric, DATE or TIMESTAMP: choose a column that holds "
                                      "numbers or dates.")
        self.check_period(t, period)
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
            return self.date_trends(main, sel, v, t, groups, filt, exprs, t_expr, where, params, period, step, filters)
        if period is not None:
            notes.append("period ignored: x is numeric (use step).")
        inner = "".join(f"{e} AS k{i}, " for i, e in enumerate(exprs)) + f"{t_expr} AS t, {self.expression(v, sel)} AS v"
        keys = "".join(f"k{i}, " for i in range(len(exprs)))
        with self.source.cursor() as cur:
            rows = cur.execute(f"SELECT {keys}t, COUNT(v), AVG(v), MIN(v), MAX(v) FROM "
                               f"(SELECT {inner} FROM {sel.from_sql} WHERE {where} AND v IS NOT NULL AND t IS NOT NULL) "
                               f"GROUP BY {keys}t ORDER BY {keys}t", params).fetchall()
        if not rows:
            raise ToolError("value", "no rows in the requested range of x.")
        series = {}
        for r in rows:
            series.setdefault(tuple(r[:len(exprs)]), []).append(tuple(r[len(exprs):]))
        self.check_group_count(len(series))
        span = max(r[len(exprs)] for r in rows) - min(r[len(exprs)] for r in rows)
        if step is not None and step <= 0:
            raise ToolError("argument", "step must be positive.")
        per_group = self.points_per_group(len(series))
        asked = step is not None   # a step given by the model is always applied
        if step is None:
            step = stats.nice_step(span, self.analysis["default_points"])
        # Without a step from the model, a group keeps every x if it has no more points than the normal sampling
        # would give (at most per_group, and at most one per default step): 6 irregular values stay 6, 31 regular
        # values 0..30 are still sampled every 5.
        keep_up_to = min(per_group, math.floor(span / step + 1e-9) + 1)
        every_x = not asked and all(len(points) <= keep_up_to for points in series.values())
        if every_x:   # few values of x (possibly irregular): none is dropped, so there is no step to report
            reported = None
        else:
            if span / step + 1 > per_group:
                step = stats.nice_step(span, per_group - 1)
                notes.append(f"step enlarged to {step} to keep at most {per_group} points per group.")
            reported = step
        fraction = self.analysis["trend_last_fraction"]
        labels = self.single_value_labels(filters, filt)
        extra, holders, label_notes = self.group_labels(main, sel, where, params, filt, groups, [], t, exprs,
                                                        list(series), labels)
        notes += label_notes
        out = [{"group": {**labels, **dict(zip([g.text for g in groups], key)), **extra.get(key, {})},
                **stats.trend_summary(points, step, fraction, per_group=None if asked else keep_up_to)}
               for key, points in series.items()]
        result = {"value": v.text, "filters": self.echo_filters(filters, filt), "group_by": [g.qualified for g in groups],
                  **self.mixed_warning(filt, groups, (), holders),
                  "x": t.text, "step": reported,
                  "explanation": (f"mean is over the observations (rows) of each group at each value of {t.text}; "
                                  "min and max, only in final, are over the observations at the last value; "
                                  f"slope_last_part is a least-squares slope of the mean over the last "
                                  f"{round(100 * fraction)}% of the range, per unit of {t.text}; shape and turning "
                                  "points use the sampled series."),
                  "groups": out}
        notes += self.require_not_mixed(main, [v, t, *groups], filt, exprs + [t_expr], v)
        notes += self.join_notes(main, sel, [v, t, *groups], filt, groups, where, params)
        notes += self.repeated_rows_note(main, sel, v, groups, where, params, exprs + [t_expr])
        if notes:
            result["notes"] = notes
        return deep_clean(result)

    def check_group_count(self, n_groups: int) -> None:
        """Raise an 'argument' error if describe_trends would return more than max_groups groups."""
        most = self.analysis["max_groups"]
        if n_groups > most:
            raise ToolError("argument", f"{n_groups} groups: too many. Use filters to keep at most {most}.")

    def repeated_rows_note(self, main: str, sel: Selection, v: Ref, groups: list[Ref], where: str, params: list,
                           cells: list[str]) -> list[str]:
        """The repetition note of summarize_results for describe_trends: [] or one note (no 'x + at' advice)."""
        skip = {(g.table, g.column) for g in groups if g.path is None}
        repeated = self.repeated_key(main, sel, v, where, params, cells=cells, skip=skip)
        return [self.repetition_note(repeated, with_x=False)] if repeated else []

    def date_trends(self, main, sel, v, t, groups, filt, exprs, t_expr, where, params, period, step,
                    filters: dict | None = None) -> dict:
        """describe_trends for a DATE or TIMESTAMP x: one point per period (hour, day, week, month or year)."""
        notes = []
        if step is not None:
            notes.append("step ignored: x is a date (use period).")
        if period is None:   # the finest period allowed for the type of x that keeps at most max_points points; else the coarsest
            allowed = [p for p in PERIODS if p != "hour" or self.is_timestamp(t)]
            for period in allowed:
                n_periods = self.count_periods(sel, t_expr, where, params, period)
                if n_periods <= self.max_points:
                    break
        else:
            n_periods = self.count_periods(sel, t_expr, where, params, period)
        if n_periods == 0:
            raise ToolError("value", "no rows in the requested range of x.")
        bucket = self.bucket(t_expr, period)
        inner = "".join(f"{e} AS k{i}, " for i, e in enumerate(exprs)) + f"{bucket} AS t, {self.expression(v, sel)} AS v"
        keys = "".join(f"k{i}, " for i in range(len(exprs)))
        with self.source.cursor() as cur:
            rows = cur.execute(f"SELECT {keys}t, COUNT(v), AVG(v), MIN(v), MAX(v) FROM "
                               f"(SELECT {inner} FROM {sel.from_sql} WHERE {where} AND v IS NOT NULL AND {t_expr} IS NOT NULL) "
                               f"GROUP BY {keys}t ORDER BY {keys}t", params).fetchall()
        if not rows:
            raise ToolError("value", "no rows in the requested range of x.")
        origin = min(r[len(exprs)] for r in rows)
        hourly = period == "hour"
        series = {}
        for r in rows:   # x becomes the number of days (hours for period hour) since the first period
            elapsed = r[len(exprs)] - origin
            number = int(elapsed.total_seconds() // 3600) if hourly else elapsed.days
            series.setdefault(tuple(r[:len(exprs)]), []).append((number, *r[len(exprs) + 1:]))
        self.check_group_count(len(series))
        per_group = self.points_per_group(len(series))
        every = math.ceil(n_periods / per_group)
        labels = self.single_value_labels(filters, filt)
        extra, holders, label_notes = self.group_labels(main, sel, where, params, filt, groups, [], t, exprs,
                                                        list(series), labels)
        notes += label_notes
        if every > 1:
            notes.append(f"{n_periods} {period}s are more than {per_group} points: series_sampled keeps one "
                         f"{period} in {every} and the last; the other values use every {period}.")
        out = []
        fraction = self.analysis["trend_last_fraction"]
        for key, points in series.items():
            sampled = points[::every] + ([points[-1]] if (len(points) - 1) % every else [])
            summary = stats.trend_summary(points, 1, fraction, sampled=sampled)
            out.append({"group": {**labels, **dict(zip([g.text for g in groups], key)), **extra.get(key, {})},
                        **self.dates_in_summary(summary, origin, period)})
        result = {"value": v.text, "filters": self.echo_filters(filters, filt), "group_by": [g.qualified for g in groups],
                  **self.mixed_warning(filt, groups, (), holders),
                  "x": t.text, "period": period,
                  "explanation": (f"each point is one {period} ({'the date and hour of its start' if hourly else 'the date of its first day'}): "
                                  "mean is over the observations (rows) in it; min and max, only in final, are over the "
                                  f"observations of the last {period}; slope_last_part is a least-squares slope of the "
                                  f"mean over the last {round(100 * fraction)}% of the range, per {'hour' if hourly else 'day'}; "
                                  "shape and turning points use the sampled series."),
                  "groups": out}
        notes += self.require_not_mixed(main, [v, t, *groups], filt, exprs + [bucket], v)
        notes += self.join_notes(main, sel, [v, t, *groups], filt, groups, where, params)
        notes += self.repeated_rows_note(main, sel, v, groups, where, params, exprs + [bucket])
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
    def compare_groups(self, value: str, group_by: str, a, b, at=None, x: str | None = None,
                       filters: dict | None = None, table: str | None = None, unit=None) -> dict:
        """Difference of means between two groups (b minus a), with Welch's interval over the observations.

        With unit, the values of each unit are averaged first and Welch's interval is over the units. Without
        unit, if the rows of a group repeat for a key (several rows per unit, say), the means are given
        but the verdict is "none": the rows are not independent.
        """
        if isinstance(group_by, (list, tuple)):
            if len(group_by) != 1:
                raise ToolError("argument", "compare_groups compares two values of ONE column: give one "
                                            "group_by column and fix the other columns with filters.")
            group_by = group_by[0]
        main = self.main_table(value, table)
        filters, x, at = self.read_filters("compare_groups", filters, x, at, main)
        v = self.parse(value, main)
        g = self.parse(group_by, main)
        units = [self.parse(u, main) for u in as_list(unit)]
        t = self.parse(x, main) if x else None
        filt = self.parse_filters(filters, main)
        if at is not None and t is None:
            raise ToolError("argument", "'at' needs 'x': the column along which the data are ordered.")
        sel = self.select(main, [v, g, *units, *([t] if t else [])], filt)
        if not self.is_numeric(v):
            raise ToolError("column", f"{value!r} is not numeric: choose a numeric column as value.")
        self.require_rows(sel, filt, main)
        numeric_group = self.is_numeric(g)
        dated_group = self.is_date(g)
        try:
            if dated_group:   # a and b are days
                a_val, b_val = self.day_of(a, "a"), self.day_of(b, "b")
            else:
                a_val, b_val = (float(a), float(b)) if numeric_group else (str(a), str(b))
        except (TypeError, ValueError):
            raise ToolError("value", f"{group_by!r} holds numbers: a and b must be numbers, got {a!r}, {b!r}.")
        g_expr = self.expression(g, sel)
        notes = []
        if isinstance(at, (list, tuple)):
            if len(at) != 1:
                raise ToolError("argument", "compare_groups compares at ONE value of x: give a single value for 'at'.")
            at = at[0]
        if t is not None:
            t_expr = self.expression(t, sel)
            if at is None:
                at = self.last_time(sel, t_expr)
                notes.append(f"'at' not given: compared at the last value of x, {stats.clean(at, stats.KEY_DIGITS)}.")
            at_sql, at_params = self.equal_condition(t, t_expr, [self.time_value(t, at)])
            if self.is_timestamp(t):
                notes.append("at compares whole days of x.")
        samples, rows_per_unit, selected = {}, {}, {}
        for label, group_value in (("a", a_val), ("b", b_val)):
            g_sql, g_params = self.equal_condition(g, g_expr, [group_value]) if numeric_group or dated_group else (f"{g_expr} = ?", [group_value])
            where = f"{sel.where_sql} AND {g_sql}" + (f" AND {at_sql}" if t is not None else "")
            params = list(sel.params) + g_params + (at_params if t is not None else [])
            selected[label] = (where, params)
            value_sql = self.expression(v, sel)
            if units:   # one mean per unit
                unit_sql = ", ".join(f"{self.expression(u, sel)} AS u{i}" for i, u in enumerate(units))
                query = (f"SELECT AVG(v), COUNT(v) FROM (SELECT {value_sql} AS v, {unit_sql} FROM {sel.from_sql} "
                         f"WHERE {where}) GROUP BY {', '.join(f'u{i}' for i in range(len(units)))}")
            else:
                query = f"SELECT {value_sql}, 1 FROM {sel.from_sql} WHERE {where}"
            with self.source.cursor() as cur:
                rows = cur.execute(query, params).fetchall()
            samples[label] = [r[0] for r in rows if r[0] is not None]
            rows_per_unit[label] = [r[1] for r in rows if r[0] is not None]
            if not samples[label]:
                raise ToolError("value", f"no rows for {group_by} = {group_value!r}"
                                         + (f" at {x} = {at}" if t is not None else "")
                                         + f". Values that exist: {self.available_values([g], main)}.")
        a_s, b_s = samples["a"], samples["b"]
        varied = self.varied_filters(filt, [g], units, t)
        labels, holders, keys = {}, set(), []
        for label in ("a", "b"):   # what each of the two groups holds of the filtered columns that keep several values
            found, held = self.filter_labels(sel, *selected[label], [], filt, varied)
            labels[label] = found.get((), {})
            holders |= held
            with self.source.cursor() as cur:   # the value of the group as the database has it
                keys.append((cur.execute(f"SELECT {g_expr} FROM {sel.from_sql} WHERE {selected[label][0]} LIMIT 1",
                                         selected[label][1]).fetchone()[0],))
        from_units, label_notes = self.parameter_labels(main, [g], keys, {*(r.qualified for r in varied), g.qualified})
        notes += label_notes
        result = {"value": v.text, "filters": self.echo_filters(filters, filt), "group_by": g.qualified,
                  **self.mixed_warning(filt, [g], units, holders), "at": at if t is not None else None}
        if units:
            result["unit"] = [u.text for u in units]
        for label, group_value, key in (("a", a_val, keys[0]), ("b", b_val, keys[1])):
            s = samples[label]
            result[label] = {"group": group_value, **({"labels": {**labels[label], **from_units[key]}}
                                                      if labels[label] or from_units[key] else {}),
                             "n_units" if units else "n": len(s), "mean": sum(s) / len(s), "min": min(s), "max": max(s)}
            if units:
                result[label]["rows_per_unit"] = [min(rows_per_unit[label]), max(rows_per_unit[label])]
        result["ranges_overlap"] = max(min(a_s), min(b_s)) <= min(max(a_s), max(b_s))
        repeated = None
        if not units:
            skip = {(g.table, g.column)} if g.path is None else set()
            for label in ("a", "b"):
                repeated = repeated or self.repeated_key(main, sel, v, *selected[label], skip=skip)
        if repeated:
            notes.append(self.repetition_note(repeated, with_x=t is not None))
            result["verdict"] = "none"
            result["verdict_reason"] = (f"the rows of a group repeat for the same {', '.join(repeated[0])}, so they are "
                                        "not independent and Welch's interval would be too narrow. Pass "
                                        f"unit={json.dumps(repeated[0])} to compare per unit, or fix the varying columns.")
        else:
            result.update(self.verdict(a_s, b_s))
        notes += self.require_not_mixed(main, [v, g, *units, *([t] if t else [])], filt,
                                        [f"CAST({g_expr} AS DATE)" if dated_group else g_expr]
                                        + ([f"CAST({self.expression(t, sel)} AS DATE)" if self.is_timestamp(t) else self.expression(t, sel)]
                                           if t else []), v)
        if notes:
            result["notes"] = notes
        return deep_clean(result)

    # ------------------------------------------------------------------ data quality

    def cap(self, items: list, label: str, notes: list) -> list:
        """The first max_items items; if there are more, a note says how many are shown."""
        most = self.analysis["max_items"]
        if len(items) > most:
            notes.append(f"{label}: showing {most} of {len(items)}.")
        return items[:most]

    @staticmethod
    def label_of(key) -> str:
        """A group (the values of the group_by columns) as a short text."""
        return " | ".join("NULL" if x is None else str(stats.clean(x, stats.KEY_DIGITS)) for x in key)

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
            per_period[c] = {str(stats.clean(p, stats.KEY_DIGITS)): share for p, share in kept}
            for label in group_labels:
                run = []
                for i in range(len(periods) + 1):
                    cell = cells.get((label, i))
                    if cell and cell[0] and cell[1][c_index] == cell[0]:
                        run.append(i)
                        continue
                    if len(run) >= 3:
                        runs.append({"group": label, "column": c, "start": str(stats.clean(periods[run[0]], stats.KEY_DIGITS)),
                                     "end": str(stats.clean(periods[run[-1]], stats.KEY_DIGITS)), "periods": len(run)})
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
    def check_quality(self, table: str, group_by=None, x: str | None = None, period: str | None = None,
                      filters: dict | None = None) -> dict:
        """A compact data-quality report on one table: NULLs, runs of NULLs, duplicates, text variants, orphans.

        The model decides what a NULL or a duplicate means; this only reports the facts.
        """
        if table not in self.schema:
            raise ToolError("column", f"unknown table {table!r}. Tables: {list(self.schema)}.")
        filters, _, _ = self.read_filters("check_quality", filters, None, None, table)
        groups = [self.parse(g, table) for g in as_list(group_by)]
        t = self.parse(x, table) if x else None
        filt = self.parse_filters(filters, table)
        for ref in [*groups, *([t] if t else []), *filt]:
            if ref.table != table:
                raise ToolError("column", f"{ref.text!r} is a column of table {ref.table}: check_quality looks at "
                                          f"one table at a time, here {table}.")
        if t is not None and not (self.is_date(t) or self.is_numeric(t)):
            raise ToolError("column", f"x must be a numeric, DATE or TIMESTAMP column, got {x!r}.")
        if t is not None:
            self.check_period(t, period)
        elif period is not None and period not in PERIODS:
            raise ToolError("argument", f"period must be one of {list(PERIODS)}, got {period!r}.")
        sel = self.select(table, [*groups, *([t] if t else [])], filt)
        self.require_rows(sel, filt, table)
        notes = []
        columns = list(self.schema[table])
        q = {c: f"t0.{quote_identifier(c)}" for c in columns}
        with self.source.cursor() as cur:
            counts = cur.execute(f"SELECT COUNT(*){''.join(f', COUNT({q[c]})' for c in columns)} FROM {sel.from_sql} "
                                 f"WHERE {sel.where_sql}", sel.params).fetchone()
        n_rows = counts[0]
        out = {"table": table, "filters": self.echo_filters(filters, filt), "group_by": [g.qualified for g in groups],
               "rows": n_rows}
        if n_rows == 0:
            return out
        null_counts = [(c, n_rows - n) for c, n in zip(columns, counts[1:]) if n < n_rows]
        null_counts.sort(key=lambda x: -x[1])
        listed = self.cap(null_counts, "nulls", notes)
        if listed:
            out["nulls"] = {c: {"count": n, "share": n / n_rows} for c, n in listed}
        dated = t is not None and self.is_date(t)
        if t is not None and not dated and period is not None:
            notes.append("period ignored: x is numeric.")
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
        """Exact duplicate rows, and repeated values in the columns that are unique in almost every row.

        A text or integer column that is not JSON, does not link to another table and has distinct values in at
        least identifier_min_share (but not all) of its non-null values is an identifier with some repeats. Columns that link
        to another table are left out: an id repeats in the table that refers to it by nature.
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
                                         "examples": [dict(zip(names, stats.clean_row(r, len(r)))) for r in examples.fetchall()]}
        found = []
        for c in columns:
            kind = self.schema[table][c]
            if not (TEXT_TYPE.match(kind) or INTEGER_TYPE.match(kind)) or self.is_json_column(table, c):
                continue
            if (table, c) in self.links():   # a column that points to another table repeats by nature
                continue
            _, non_null, distinct = self.profiles()[(table, c)]
            if not (non_null and distinct < non_null and distinct >= self.analysis["identifier_min_share"] * non_null):
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
                if total and numbers / total >= self.analysis["number_share"]:
                    others = cur.execute(f"SELECT DISTINCT x FROM ({base}) WHERE NOT regexp_matches(x, '{NUMBER_TEXT}') "
                                         "ORDER BY x LIMIT 3", sel.params).fetchall()
            variants += [{"column": c, "values": g[0][:self.analysis["max_items"]]} for g in groups]
            if total and numbers / total >= self.analysis["number_share"]:
                as_numbers.append({"column": c, "share": numbers / total, "decimal_comma": commas > 0,
                                   "not_numbers": [o[0] for o in others]})
        if variants:
            out["text_variants"] = self.cap(variants, "text_variants", notes)
        if as_numbers:
            out["numbers_as_text"] = self.cap(as_numbers, "numbers_as_text", notes)
