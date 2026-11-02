"""What the language model sees of the tools: names, descriptions and argument schemas.

The Python code of the tools is in tools.py; here only the text the model reads, kept in one place.
The model chooses the columns itself, from describe_database, and the text is improved from the
errors observed in the bench.
"""
from __future__ import annotations

from typing import Any, Literal, Union

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .tools import Toolbox

Scalar = Union[float, int, str]

COLUMN_HELP = ("a column name, table.column, or a value inside a JSON column written "
               "column.path exactly as listed in json_paths of describe_database")
UNIT_HELP = ("column(s) identifying the entity that has several rows (the unit): the values of each unit are "
             "averaged first and the statistics are over the units. Use it when one unit has several rows in "
             "a group")
PERIOD_HELP = "day, week or month: the size of the time bucket for a date or timestamp column"
FILTERS_HELP = ('keep only the rows where each column matches, e.g. {"<text column>": "<one value>", '
                '"<column.path>": [<v1>, <v2>], "<number or date column>": {"between": [<a>, <b>]}, '
                '"<number or date column>": {">=": <x>} (also >, <, <=; dates as YYYY-MM-DD), '
                '"<column>": {"is_null": true} (or false)}. Plain values must exist: see describe_database')


class NoArgs(BaseModel):
    """describe_database takes no argument."""


class QueryArgs(BaseModel):
    """Arguments of query_results."""

    sql: str = Field(description="ONE read-only SELECT or WITH ... SELECT query, DuckDB dialect.")


class SummarizeArgs(BaseModel):
    """Arguments of summarize_results."""

    value: str = Field(description=f"the numeric column to summarize: {COLUMN_HELP}.")
    group_by: Union[str, list[str], None] = Field(
        default=None, description="column(s) defining the groups: every column whose values you want kept apart.")
    time: Union[str, None] = Field(default=None, description="the column holding the time, if any.")
    at: Union[Scalar, list[Scalar], None] = Field(
        default=None, description="the time value(s) to summarize at (at most 20); default: the last time.")
    filters: Union[dict[str, Any], None] = Field(default=None, description=FILTERS_HELP + ".")
    table: Union[str, None] = Field(default=None, description="only if the value column name is in several tables.")
    unit: Union[str, list[str], None] = Field(default=None, description=UNIT_HELP + ".")


class TrendsArgs(BaseModel):
    """Arguments of describe_trends."""

    value: str = Field(description=f"the numeric column whose evolution you want: {COLUMN_HELP}.")
    time: str = Field(description="the column holding the time.")
    group_by: Union[str, list[str], None] = Field(
        default=None, description="column(s) defining the groups: every column whose values you want kept apart.")
    filters: Union[dict[str, Any], None] = Field(default=None, description=FILTERS_HELP + ".")
    step: Union[float, None] = Field(
        default=None, description="numeric time only: time step of the sampled series (default: about 12 points).")
    period: Union[Literal["day", "week", "month"], None] = Field(
        default=None, description=PERIOD_HELP + " (default: the finest that keeps the series short).")
    start: Union[float, str, None] = Field(
        default=None, description="first time to include (a date like 2026-01-31 for a date column; default: the first).")
    end: Union[float, str, None] = Field(default=None, description="last time to include (default: the last).")
    table: Union[str, None] = Field(default=None, description="only if the value column name is in several tables.")


class CompareArgs(BaseModel):
    """Arguments of compare_groups."""

    value: str = Field(description=f"the numeric column to compare: {COLUMN_HELP}.")
    group_by: str = Field(description="ONE column whose two values you compare.")
    a: Scalar = Field(description="first value of group_by.")
    b: Scalar = Field(description="second value of group_by; the difference is mean(b) - mean(a).")
    time: Union[str, None] = Field(default=None, description="the column holding the time, if any.")
    at: Union[Scalar, None] = Field(default=None, description="the time to compare at (default: the last time).")
    filters: Union[dict[str, Any], None] = Field(
        default=None, description=FILTERS_HELP + "; use it to fix every other column that varies.")
    table: Union[str, None] = Field(default=None, description="only if the value column name is in several tables.")
    unit: Union[str, list[str], None] = Field(default=None, description=UNIT_HELP + ".")


class CheckQualityArgs(BaseModel):
    """Arguments of check_quality."""

    table: str = Field(description="the table to check.")
    group_by: Union[str, list[str], None] = Field(
        default=None, description="column(s) of that table: NULL share per group, runs of NULLs per group.")
    time: Union[str, None] = Field(
        default=None, description="the time column of that table: NULLs per period and runs of NULLs.")
    period: Union[Literal["day", "week", "month"], None] = Field(
        default=None, description=PERIOD_HELP + " (default: day).")
    filters: Union[dict[str, Any], None] = Field(default=None, description=FILTERS_HELP + ".")


class CalculateArgs(BaseModel):
    """Arguments of calculate."""

    expression: str = Field(
        description="an arithmetic expression with numbers, + - * / ** and parentheses, e.g. '(3.5 - 1.2) / 4'.")


DESCRIPTIONS = {
    "describe_database": (
        "Describe the database: tables, columns, types, row counts, distinct values, numeric "
        "ranges, per column the NULL count, whether it is unique per row or repeats and which columns of other "
        "tables it links to (with the values not found there), the paths inside JSON columns with their values, "
        "shared columns between tables, sample rows."),
    "query_results": (
        "Run ONE read-only SQL query (DuckDB) on the data. Returns at most a limited number of rows, "
        "with a warning if the result was cut, empty or all NULL. Aggregate in SQL instead of reading raw rows. "
        "Prefer summarize_results, describe_trends and compare_groups for means, trends and differences."),
    "summarize_results": (
        "Count (n), mean, std, min, max, n_rows (rows including NULL values), n_null and sum of one numeric column, "
        "per group and at the requested times. Statistics are computed over all rows that share the same group_by "
        "values and time. They describe a sample only if those rows are comparable observations of one condition: "
        "fix every other varying column with filters or group_by. If one unit (the entity that has several rows) has several "
        "rows in a group, pass unit: its values are averaged first and the statistics are over the units. "
        "Use this for means and spreads: never compute them yourself. "
        "If rows of different categories would be put together, it returns an error instead: fix filters or group_by."),
    "describe_trends": (
        "How a numeric column evolves in time, per group: initial and final value, maximum and minimum of the "
        "mean with their times, slope over the last part of the time range, shape and turning points, and the "
        "series sampled every 'step' with the mean, min and max over the observations. Use it for questions about time. "
        "The time column may be a number (use step) or a date/timestamp (use period: day, week or month; times are "
        "returned as dates and the slope is per day)."),
    "compare_groups": (
        "Compare two values (a, b) of one column: difference of the means (b minus a) with Welch's t interval "
        "over the observations, its degrees of freedom, n, mean, min and max of each group, whether the min-max "
        "ranges overlap, and a verdict 'established' or 'not established' (no verdict with fewer than 2 "
        "observations in a group). Welch's interval assumes values roughly normal within each group. With many "
        "observations a bootstrap interval is added as a check. Use it before claiming that two cases differ. "
        "If a unit (the entity that has several rows) has several rows in a group, pass unit: the values of each unit are "
        "averaged first and the interval is over the units; without unit, repeated rows give the means but "
        "verdict 'none'."),
    "check_quality": (
        "Data-quality report on ONE table: rows, NULL count and share per column, NULL share per group (group_by), "
        "NULLs per period and stretches of at least 3 consecutive periods all NULL (time, period), exact duplicate rows, "
        "repeated values in identifier columns, text values that differ only by case or spaces, text columns that "
        "hold numbers (also with a decimal comma), and values of a column that are missing in the table it links to. "
        "It only reports facts: what a NULL or a duplicate means is for you to decide. Use it before relying on a "
        "column, or when a question is about missing, duplicated or dirty data."),
    "calculate": (
        "Compute an arithmetic expression exactly. Use it for every arithmetic step instead of computing in "
        "your head. Only numbers, + - * / ** and parentheses are accepted."),
}


# Argument schema of each tool: used to build the LangChain tools and, in rewoo.py, to check a plan's steps.
ARGS_SCHEMAS = {
    "describe_database": NoArgs,
    "query_results": QueryArgs,
    "summarize_results": SummarizeArgs,
    "describe_trends": TrendsArgs,
    "compare_groups": CompareArgs,
    "check_quality": CheckQualityArgs,
    "calculate": CalculateArgs,
}


def make_langchain_tools(toolbox: Toolbox) -> list[StructuredTool]:
    """Wrap the toolbox methods as LangChain tools with the descriptions and schemas above."""
    return [StructuredTool.from_function(func=getattr(toolbox, name), name=name, description=DESCRIPTIONS[name],
                                         args_schema=schema)
            for name, schema in ARGS_SCHEMAS.items()]
