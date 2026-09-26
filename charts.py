"""
Chart selection: pick a chart only when it genuinely helps.

Rules (from the dataviz method):
  * category + number  -> horizontal bar, sorted (names stay readable)
  * time + number      -> line with markers (change over time)
  * single number / row list -> no chart (the number or table is the answer)
  * one series -> one colour, no legend box; only the extreme gets a direct label
"""
from __future__ import annotations

import plotly.express as px

from agent.plan import Plan
from agent.tools import AGG_LABEL, ToolResult

SERIES_1 = "#2a78d6"          # reference palette, categorical slot 1


def _pretty(name: str) -> str:
    return name.replace("_", " ").capitalize()


def chart_for(plan: Plan, res: ToolResult):
    if res is None or res.table is None or plan.action not in ("group_aggregate", "top_n"):
        return None
    if not plan.group_by or len(res.table) < 2:
        return None

    x, y = plan.group_by, res.details["column"]
    table = res.table.copy()
    table[y] = table[y].astype(float)
    y_label = "Number of rows" if y == "rows" else f"{AGG_LABEL[plan.agg].capitalize()} {_pretty(y).lower()}"
    title = f"{y_label} per {plan.time_grain}" if plan.time_grain else f"{y_label} by {_pretty(x).lower()}"

    if plan.time_grain:
        fig = px.line(table, x=x, y=y, markers=True, title=title,
                      labels={x: plan.time_grain.capitalize(), y: y_label})
        fig.update_traces(line=dict(width=2, color=SERIES_1), marker=dict(size=8, color=SERIES_1))
        # "2026-01" must stay a label: otherwise Plotly turns it into a date
        # axis and shows ticks like "Dec 28 2025" that are not in the data.
        fig.update_xaxes(type="category")
        fig.update_yaxes(rangemode="tozero")
    else:
        table = table.sort_values(y, ascending=True)           # biggest bar on top
        fig = px.bar(table, x=y, y=x, orientation="h", title=title,
                     labels={x: _pretty(x), y: y_label})
        fig.update_traces(marker_color=SERIES_1, marker_cornerradius=4)
        top = table.iloc[-1]
        fig.add_annotation(x=top[y], y=top[x], text=f"{top[y]:,.0f}", showarrow=False,
                           xanchor="left", xshift=6)
        fig.update_layout(bargap=0.35)

    fig.update_layout(height=max(280, 60 * len(table) + 120) if not plan.time_grain else 340,
                      margin=dict(l=10, r=40, t=50, b=10), showlegend=False)
    fig.update_xaxes(showgrid=not plan.time_grain, gridwidth=1)
    fig.update_yaxes(showgrid=bool(plan.time_grain), gridwidth=1)
    return fig
