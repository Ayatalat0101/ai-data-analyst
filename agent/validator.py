"""
Validator: checks a Plan against the REAL dataset before anything runs.

Why?
----
Planners make mistakes - the LLM can invent a column called "beneficiaries",
and a rule can misread a word. The validator is the checkpoint between
"deciding" and "doing". It either returns a corrected, safe plan, or turns
the plan into a clarification / rejection the user can act on.
It never guesses silently.
"""
from __future__ import annotations

import re
from difflib import get_close_matches

import pandas as pd

from .data_loader import DatasetProfile
from .plan import FilterSpec, Plan, clarify, reject

NUMERIC_AGGS = {"sum", "mean"}


def _norm(text: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(text).casefold())


def resolve_column(name: str | None, profile: DatasetProfile) -> str | None:
    """Exact name, or same name ignoring case/spaces/underscores ('Households Reached')."""
    if name is None:
        return None
    if name in profile.column_names:
        return name
    matches = [c for c in profile.column_names if _norm(c) == _norm(name)]
    return matches[0] if matches else None


def _missing_column(name: str, profile: DatasetProfile) -> Plan:
    close = get_close_matches(name, profile.column_names, n=1, cutoff=0.6)
    hint = f" Did you mean '{close[0]}'?" if close else ""
    return reject(f"There is no column called '{name}' in this file.{hint} "
                  f"Available columns: {', '.join(profile.column_names)}.",
                  "missing_column")


def validate(plan: Plan, profile: DatasetProfile, question: str = "") -> Plan:
    if plan.action in ("ask_clarification", "reject"):
        return plan

    p = plan.model_copy(deep=True)

    # 1) every referenced column must exist ---------------------------------
    for attr in ("column", "group_by", "sort_by"):
        raw = getattr(p, attr)
        if raw is None:
            continue
        real = resolve_column(raw, profile)
        if real is None:
            return _missing_column(raw, profile)
        setattr(p, attr, real)

    # 2) the action has the parameters it needs -------------------------------
    if p.action == "aggregate" and not (p.column and p.agg):
        return clarify("Which column should I calculate, and how (total, average, min, max)?")
    if p.action == "group_aggregate":
        if not p.group_by:
            return clarify("What should I group the results by?",
                           [f"{question} by {c}" for c in profile.columns_of_kind("category")[:3]])
        if not p.agg:
            p.agg = "sum" if p.column else "count"
        if p.agg != "count" and not p.column:
            return clarify(f"Which number should I calculate per {p.group_by}?",
                           [f"Total {c} by {p.group_by}" for c in profile.columns_of_kind("number")[:3]])
    if p.action == "top_n":
        if not p.column and p.agg != "count":
            return clarify("Rank by which number?",
                           [f"{question} by {c}" for c in profile.columns_of_kind("number")[:3]])
        p.agg = p.agg or "sum"
        p.n = max(1, min(p.n or 5, 50))

    # 3) numeric operations only on numeric columns ---------------------------
    if p.agg in NUMERIC_AGGS and p.column:
        col = profile.column(p.column)
        if col.kind != "number":
            return reject(f"'{p.column}' contains {col.kind} values, so I can't compute its "
                          f"{'total' if p.agg == 'sum' else 'average'}. Numeric columns: "
                          f"{', '.join(profile.columns_of_kind('number'))}.", "unclear")
    if p.group_by and profile.column(p.group_by).kind == "date" and not p.time_grain:
        p.time_grain = "month"

    # 4) filters: real column, right type, real category value ---------------
    clean: list[FilterSpec] = []
    for f in p.filters:
        real = resolve_column(f.column, profile)
        if real is None:
            return _missing_column(f.column, profile)
        col = profile.column(real)
        f = f.model_copy(update={"column": real})

        if col.kind == "number":
            if f.op in ("contains", "in"):
                return clarify(f"'{real}' is numeric. Use a comparison like '{real} greater than 100'.")
            try:
                f.value = float(str(f.value).replace(",", ""))
            except ValueError:
                return clarify(f"'{real}' is numeric, but '{f.value}' is not a number. Which value did you mean?")

        elif col.kind == "date":
            try:
                f.value = pd.Timestamp(str(f.value)).date().isoformat()
            except ValueError:
                return clarify(f"I couldn't read '{f.value}' as a date. Please use a format like 2026-03-01.")

        elif col.kind == "category" and f.op in ("==", "!=", "in"):
            wanted = f.value if isinstance(f.value, list) else [str(f.value)]
            fixed = []
            for w in wanted:
                exact = [v for v in col.values if v.casefold() == str(w).casefold()]
                if exact:
                    fixed.append(exact[0])
                    continue
                close = get_close_matches(str(w), col.values, n=1, cutoff=0.7)
                if close and question:
                    return clarify(f"I couldn't find '{w}' in '{real}'. Did you mean '{close[0]}'?",
                                   [re.sub(re.escape(str(w)), close[0], question, flags=re.I)])
                return clarify(f"'{w}' is not a value of '{real}'. "
                               f"Available values: {', '.join(col.values)}.")
            f.value = fixed if f.op == "in" else fixed[0]
        clean.append(f)
    p.filters = clean
    return p
