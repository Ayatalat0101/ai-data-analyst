"""
The Agent: one loop that connects every part.

    question ─► planner (LLM or rules) ─► Plan
                                          │
                              validator ◄─┘  (real columns? real values? right types?)
                                  │
            ┌─────────────────────┼──────────────────────┐
     ask_clarification         reject              approved action
            │                     │                      │
      question to user     explain limit         TOOLS[action](df, ...)
                                                          │
                                           answer + table + "how I got this"

Answers are written from the tool result by a template, NOT by the LLM.
The LLM decides WHAT to compute; it never states a number itself, so it
cannot hallucinate one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import pandas as pd

from .data_loader import DatasetProfile
from .plan import Plan, reject
from .rule_planner import UNSAFE
from .rule_planner import plan_question as rule_plan
from .tools import AGG_LABEL, TOOLS, Filter, ToolError, ToolResult
from .validator import validate

Planner = Callable[[str, DatasetProfile], Plan]


@dataclass
class AgentResponse:
    kind: Literal["answer", "clarification", "rejected", "no_results", "error"]
    message: str
    plan: Plan | None = None
    result: ToolResult | None = None
    options: list[str] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)
    planner_used: str = "rules"


def _fmt(x) -> str:
    """1234 -> '1,234' · 289.333 -> '289.33' · works for numpy numbers too."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    if hasattr(x, "item"):
        x = x.item()                      # numpy int64 / float64 -> Python
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{x:,.0f}" if x.is_integer() else f"{x:,.2f}"
    if isinstance(x, int):
        return f"{x:,}"
    return str(x)


def _where(res: ToolResult) -> str:
    return f" (where {' and '.join(res.filters)})" if res.filters else ""


def _answer_text(plan: Plan, res: ToolResult) -> str:
    agg_word = AGG_LABEL.get(plan.agg or "", plan.agg or "")
    if plan.action == "count_rows":
        return f"**{_fmt(res.value)}** rows match{_where(res)}."
    if plan.action == "aggregate":
        return f"The {agg_word} of **{plan.column}** is **{_fmt(res.value)}**{_where(res)}."
    if plan.action == "filter_rows":
        shown = res.details.get("shown", 0)
        extra = f" Showing the first {shown}." if shown < res.rows_matched else ""
        return f"Found **{res.rows_matched}** matching rows{_where(res)}.{extra}"
    if plan.action == "group_aggregate":
        metric = res.details["column"]
        top = res.table.iloc[0] if plan.time_grain is None else res.table.loc[res.table[metric].idxmax()]
        return (f"{agg_word.capitalize()} of **{metric}** by **{plan.group_by}**{_where(res)}. "
                f"Highest: **{top[plan.group_by]}** ({_fmt(top[metric])}).")
    if plan.action == "top_n":
        direction = res.details["direction"]
        if plan.group_by:
            metric = res.details["column"]
            first = res.table.iloc[0]
            label = "row count" if metric == "rows" else f"{agg_word} {metric}"
            lead = f"**{first[plan.group_by]}** has the {direction} {label}: **{_fmt(first[metric])}**"
            if plan.n == 1:
                return lead + _where(res) + "." + (" Full ranking below." if len(res.table) > 1 else "")
            return lead + f"{_where(res)}. Top {len(res.table)} shown below."
        return f"Top {len(res.table)} rows by **{plan.column}** ({direction} first){_where(res)}."
    return "Done."


def _explain(plan: Plan, res: ToolResult, planner_name: str) -> list[str]:
    lines = [f"**Planner:** {planner_name}",
             f"**Action:** `{plan.action}` — {plan.reason}"]
    cols = [c for c in (plan.column, plan.group_by) if c]
    if cols:
        lines.append(f"**Columns:** {', '.join(f'`{c}`' for c in cols)}"
                     + (f" · grouped by {plan.time_grain}" if plan.time_grain else ""))
    if plan.agg:
        lines.append(f"**Calculation:** {AGG_LABEL[plan.agg]}")
    lines.append(f"**Filters:** {' AND '.join(f'`{f}`' for f in res.filters) if res.filters else 'none (all rows)'}")
    lines.append(f"**Rows used:** {res.rows_matched}")
    if res.rows_excluded_missing:
        lines.append(f"⚠️ **{res.rows_excluded_missing} row(s) excluded** because "
                     f"`{plan.column}` was empty.")
    return lines


def _run_tool(plan: Plan, df: pd.DataFrame) -> ToolResult:
    filters = [Filter(f.column, f.op, f.value) for f in plan.filters]
    a = plan.action
    if a == "count_rows":
        return TOOLS[a](df, filters)
    if a == "filter_rows":
        return TOOLS[a](df, filters, sort_by=plan.sort_by, ascending=plan.ascending)
    if a == "aggregate":
        return TOOLS[a](df, plan.column, plan.agg, filters)
    if a == "group_aggregate":
        return TOOLS[a](df, plan.group_by, plan.agg, plan.column, filters, plan.time_grain)
    if a == "top_n":
        # "Which partner is highest?" names ONE winner, but we return the full
        # ranking so the user can see how close the others are.
        n = 10 if (plan.group_by and plan.n == 1) else (plan.n or 5)
        return TOOLS[a](df, plan.column, n, plan.agg or "sum", plan.group_by,
                        plan.ascending, filters)
    raise ToolError(f"Unknown action '{a}'")


def _no_results_hint(plan: Plan, profile: DatasetProfile) -> str:
    hints = []
    for f in plan.filters:
        col = profile.column(f.column)
        if col and col.kind in ("number", "date"):
            hints.append(f"`{f.column}` ranges from {col.min} to {col.max}")
    return (" Note: " + "; ".join(hints) + ".") if hints else " Try removing a filter."


FIXED_REJECTIONS = {
    # Found in testing: the LLM told a user "Uploading additional files is not
    # allowed", which is false. For refusals, the LLM chooses the CATEGORY and
    # our code writes the words, so the app never makes claims about itself
    # that aren't true.
    "unsafe": "I can only read and analyse this file. I can't change data, run code, "
              "or act outside this analysis.",
    "out_of_scope": "That's outside what I can do: I summarise what is already in the file "
                    "(counts, totals, averages, rankings, monthly trends). I don't forecast "
                    "or explain causes. Try, for example, a total per month so far.",
}


def _repair_llm_messages(plan: Plan, question: str, profile: DatasetProfile) -> Plan:
    if plan.action == "reject" and plan.reject_reason in FIXED_REJECTIONS:
        return plan.model_copy(update={"question_to_user": FIXED_REJECTIONS[plan.reject_reason]})

    if plan.action == "ask_clarification" and not plan.options:
        # Found in testing: the LLM asked "How should I evaluate partners?" but gave
        # no options, leaving the user to guess the answer format. Borrow the rule
        # planner's options when it also sees ambiguity; otherwise offer examples.
        ruled = rule_plan(question, profile)
        options = ruled.options if ruled.action == "ask_clarification" else []
        if not options:
            num = profile.columns_of_kind("number")
            cat = profile.columns_of_kind("category")
            options = [o for o in (
                "How many rows are there?",
                f"Total {num[0].replace('_', ' ')} by {cat[0]}" if num and cat else None,
                f"Which {cat[0]} has the most rows?" if cat else None,
            ) if o]
        return plan.model_copy(update={"options": options[:4]})
    return plan


def answer(question: str, df: pd.DataFrame, profile: DatasetProfile,
           planner: Planner | None = None, planner_name: str = "rules") -> AgentResponse:
    """The agent loop for ONE question."""
    q = (question or "").strip()
    if len(q) < 3:
        return AgentResponse("error", "Please type a question about your data.")
    if len(q) > 300:
        return AgentResponse("error", "Please shorten the question to under 300 characters.")

    # 0. hard guardrail: deterministic, runs BEFORE any LLM ------------------
    #    An unsafe request is refused by code, so no prompt trick can talk
    #    its way past it, and it never costs an API call.
    if UNSAFE.search(" " + q.casefold() + " "):
        plan = reject("I can only read and analyse this file. I can't change data or run code.",
                      "unsafe", reason="Blocked by the safety guardrail before planning.")
        return AgentResponse("rejected", plan.question_to_user, plan=plan, planner_used="guardrail")

    # 1. decide ---------------------------------------------------------------
    try:
        out = (planner or rule_plan)(q, profile)
        plan, planner_name = out if isinstance(out, tuple) else (out, planner_name)
    except Exception as err:                          # LLM down, quota, bad JSON...
        plan = rule_plan(q, profile)
        planner_name = f"rules (fallback: {type(err).__name__})"

    # 2. check the decision against the real data ----------------------------
    plan = validate(plan, profile, q)
    plan = _repair_llm_messages(plan, q, profile)

    if plan.action == "ask_clarification":
        return AgentResponse("clarification", plan.question_to_user or "Could you clarify?",
                             plan=plan, options=plan.options, planner_used=planner_name)
    if plan.action == "reject":
        return AgentResponse("rejected", plan.question_to_user or "I can't answer that.",
                             plan=plan, planner_used=planner_name)

    # 3. act with an approved tool -------------------------------------------
    try:
        res = _run_tool(plan, df)
    except ToolError as err:
        return AgentResponse("error", f"I couldn't run this analysis: {err}", plan=plan,
                             planner_used=planner_name)

    explanation = _explain(plan, res, planner_name)
    if res.is_empty:
        return AgentResponse("no_results",
                             f"No rows match{_where(res)}.{_no_results_hint(plan, profile)}",
                             plan=plan, result=res, explanation=explanation,
                             planner_used=planner_name)

    # 4. report ---------------------------------------------------------------
    return AgentResponse("answer", _answer_text(plan, res), plan=plan, result=res,
                         explanation=explanation, planner_used=planner_name)
