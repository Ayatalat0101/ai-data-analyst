"""
The Plan: the ONLY thing a planner (rules or LLM) is allowed to produce.

Why a typed model?
------------------
The planner does not answer the question and does not write code. It fills
this form. Pydantic guarantees the form has the right shape (e.g. `action`
must be one of 7 words), and the same class is given to Gemini as its
`response_schema`, so the LLM physically cannot return anything else.
"""
from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

Action = Literal["count_rows", "filter_rows", "aggregate", "group_aggregate", "top_n",
                 "ask_clarification", "reject"]


class FilterSpec(BaseModel):
    column: str = Field(description="Exact column name from the schema")
    op: Literal["==", "!=", ">", ">=", "<", "<=", "in", "contains"]
    value: Union[float, str, list[str]] = Field(
        description="Number for numeric columns, ISO date for dates, exact category value for categories")


class Plan(BaseModel):
    action: Action
    reason: str = Field(description="One short sentence: why this action fits the question")

    # analysis parameters (only the ones the action needs)
    column: Optional[str] = Field(None, description="Numeric column to measure or rank by")
    agg: Optional[Literal["sum", "mean", "min", "max", "count"]] = None
    group_by: Optional[str] = Field(None, description="Category or date column to group by")
    time_grain: Optional[Literal["day", "week", "month", "year"]] = None
    filters: list[FilterSpec] = Field(default_factory=list)
    n: Optional[int] = Field(None, description="How many results for top_n")
    ascending: bool = Field(False, description="True for lowest / least / smallest")
    sort_by: Optional[str] = None

    # conversation paths
    question_to_user: Optional[str] = Field(
        None, description="For ask_clarification or reject: the message shown to the user")
    options: list[str] = Field(
        default_factory=list,
        description="For ask_clarification: 2-4 complete, unambiguous rewrites of the user's question")
    reject_reason: Optional[Literal["missing_column", "out_of_scope", "unsafe", "unclear"]] = None


def clarify(message: str, options: list[str] | None = None, reason: str = "") -> Plan:
    return Plan(action="ask_clarification", reason=reason or "The question is ambiguous.",
                question_to_user=message, options=options or [])


def reject(message: str, why: str, reason: str = "") -> Plan:
    return Plan(action="reject", reason=reason or message, question_to_user=message,
                reject_reason=why)
