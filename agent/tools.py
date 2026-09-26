"""
Agent tools: the ONLY code that touches the DataFrame.

Why a fixed set of functions?
-----------------------------
The LLM never writes Pandas code. It chooses one of these functions and fills
its parameters. That gives us three properties the brief asks for:
  * Safe        - nothing outside this list can ever run (no eval / exec).
  * Verifiable  - every function is unit-tested against known answers.
  * Explainable - every result records the filters, rows used and rows excluded.

Every tool is read-only: it never modifies the DataFrame it receives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

Op = Literal["==", "!=", ">", ">=", "<", "<=", "in", "contains"]
Agg = Literal["sum", "mean", "min", "max", "count"]

AGG_LABEL = {"sum": "total", "mean": "average", "min": "minimum",
             "max": "maximum", "count": "number of rows"}


class ToolError(Exception):
    """Raised when parameters are impossible (e.g. mean of a text column)."""


@dataclass
class Filter:
    column: str
    op: Op
    value: Any

    def describe(self) -> str:
        return f"{self.column} {self.op} {self.value!r}"


@dataclass
class ToolResult:
    action: str
    value: float | int | None = None          # single-number answers
    table: pd.DataFrame | None = None         # tabular answers
    rows_matched: int = 0                     # rows left after filters
    rows_excluded_missing: int = 0            # dropped because the metric was empty
    filters: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.rows_matched == 0


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _require(df: pd.DataFrame, *columns: str | None) -> None:
    missing = [c for c in columns if c and c not in df.columns]
    if missing:
        raise ToolError(f"Column(s) not found: {missing}")


def _require_numeric(df: pd.DataFrame, column: str, agg: str) -> None:
    if agg in ("sum", "mean") and not pd.api.types.is_numeric_dtype(df[column]):
        raise ToolError(f"Cannot compute the {AGG_LABEL[agg]} of '{column}' because it is not numeric.")


def _coerce(series: pd.Series, value: Any) -> Any:
    """Convert the filter value to the column's type ('600' -> 600.0 for numbers)."""
    if isinstance(value, (list, tuple, set)):
        return [_coerce(series, v) for v in value]
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.Timestamp(value)
    if pd.api.types.is_numeric_dtype(series):
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ToolError(f"'{series.name}' is numeric but the filter value {value!r} is not a number.")
    return str(value)


def _mask(df: pd.DataFrame, f: Filter) -> pd.Series:
    s = df[f.column]
    v = _coerce(s, f.value)
    is_text = not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s))

    if is_text and f.op in ("==", "!=", "in"):
        # case-insensitive text matching: "rafah" == "Rafah"
        left = s.astype("string").str.casefold()
        vals = [str(x).casefold() for x in (v if isinstance(v, list) else [v])]
        m = left.isin(vals)
        return (~m if f.op == "!=" else m).fillna(False)
    if f.op == "contains":
        return s.astype("string").str.contains(str(v), case=False, regex=False).fillna(False)
    if f.op == "in":
        return s.isin(v if isinstance(v, list) else [v])
    if f.op in (">", ">=", "<", "<=") and is_text:
        raise ToolError(f"'{f.column}' is text; '{f.op}' only works on numbers or dates.")
    ops = {"==": s.eq, "!=": s.ne, ">": s.gt, ">=": s.ge, "<": s.lt, "<=": s.le}
    return ops[f.op](v).fillna(False)


def apply_filters(df: pd.DataFrame, filters: list[Filter] | None) -> tuple[pd.DataFrame, list[str]]:
    filters = filters or []
    _require(df, *[f.column for f in filters])
    mask = pd.Series(True, index=df.index)
    for f in filters:
        mask &= _mask(df, f)
    return df[mask], [f.describe() for f in filters]


def _group_key(df: pd.DataFrame, group_by: str, time_grain: str | None) -> pd.Series:
    """Group dates by month/week/year instead of by exact day."""
    s = df[group_by]
    if time_grain and pd.api.types.is_datetime64_any_dtype(s):
        freq = {"day": "D", "week": "W", "month": "M", "year": "Y"}[time_grain]
        return s.dt.to_period(freq).astype(str).rename(group_by)
    return s


# ----------------------------------------------------------------------------
# The five approved actions
# ----------------------------------------------------------------------------
def count_rows(df: pd.DataFrame, filters: list[Filter] | None = None) -> ToolResult:
    sub, desc = apply_filters(df, filters)
    return ToolResult("count_rows", value=len(sub), rows_matched=len(sub), filters=desc)


def filter_rows(df: pd.DataFrame, filters: list[Filter] | None = None,
                sort_by: str | None = None, ascending: bool = True,
                limit: int = 20) -> ToolResult:
    _require(df, sort_by)
    sub, desc = apply_filters(df, filters)
    if sort_by:
        sub = sub.sort_values(sort_by, ascending=ascending)
    return ToolResult("filter_rows", table=sub.head(limit), rows_matched=len(sub),
                      filters=desc, details={"shown": min(limit, len(sub)), "limit": limit})


def aggregate(df: pd.DataFrame, column: str, agg: Agg,
              filters: list[Filter] | None = None) -> ToolResult:
    _require(df, column)
    _require_numeric(df, column, agg)
    sub, desc = apply_filters(df, filters)
    values = sub[column].dropna()
    excluded = len(sub) - len(values)
    result = None if values.empty else values.agg(agg)
    if hasattr(result, "item"):
        result = result.item()                 # numpy scalar -> plain Python
    return ToolResult("aggregate", value=result, rows_matched=len(sub),
                      rows_excluded_missing=excluded, filters=desc,
                      details={"column": column, "agg": agg})


def group_aggregate(df: pd.DataFrame, group_by: str, agg: Agg, column: str | None = None,
                    filters: list[Filter] | None = None, time_grain: str | None = None,
                    ascending: bool = False) -> ToolResult:
    """Metric per category, e.g. total quantity per governorate.
    agg='count' counts rows per group and needs no column."""
    if agg != "count" and not column:
        raise ToolError(f"'{agg}' needs a numeric column to calculate.")
    _require(df, group_by, column)
    if column:
        _require_numeric(df, column, agg)
    sub, desc = apply_filters(df, filters)

    excluded = 0
    if agg == "count":
        metric = "rows"
        table = sub.groupby(_group_key(sub, group_by, time_grain)).size()
    else:
        metric = column
        excluded = int(sub[column].isna().sum())
        table = sub.groupby(_group_key(sub, group_by, time_grain))[column].agg(agg)

    table = table.sort_index() if time_grain else table.sort_values(ascending=ascending)
    if table.dtype.kind == "f" and (table.dropna() % 1 == 0).all() and agg != "mean":
        table = table.astype("Int64")      # 3713.0 -> 3713 (floats only because of NaN)
    table = table.rename(metric).reset_index()
    return ToolResult("group_aggregate", table=table, rows_matched=len(sub),
                      rows_excluded_missing=excluded, filters=desc,
                      details={"group_by": group_by, "column": metric, "agg": agg,
                               "time_grain": time_grain})


def top_n(df: pd.DataFrame, column: str | None, n: int = 5, agg: Agg = "sum",
          group_by: str | None = None, ascending: bool = False,
          filters: list[Filter] | None = None) -> ToolResult:
    """Two shapes:
       * group_by set  -> top groups by aggregated metric ("partner with most households")
       * group_by None -> top individual rows by a column ("5 largest distributions")"""
    n = max(1, min(int(n), 50))
    if group_by:
        res = group_aggregate(df, group_by, agg, column, filters, ascending=ascending)
        res.table = res.table.head(n)
        res.action = "top_n"
        res.details["n"] = n
        res.details["direction"] = "lowest" if ascending else "highest"
        return res

    if not column:
        raise ToolError("top_n without group_by needs a column to rank by.")
    _require(df, column)
    sub, desc = apply_filters(df, filters)
    ranked = sub.dropna(subset=[column])
    ranked = ranked.nsmallest(n, column) if ascending else ranked.nlargest(n, column)
    return ToolResult("top_n", table=ranked, rows_matched=len(sub),
                      rows_excluded_missing=int(sub[column].isna().sum()), filters=desc,
                      details={"column": column, "n": n,
                               "direction": "lowest" if ascending else "highest"})


# Registry: the agent can only call names in this dict.
TOOLS = {
    "count_rows": count_rows,
    "filter_rows": filter_rows,
    "aggregate": aggregate,
    "group_aggregate": group_aggregate,
    "top_n": top_n,
}
