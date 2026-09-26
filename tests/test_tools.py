"""
Tool tests against the ground truth written in PROJECT_PLAN.md §10
(computed independently with plain Pandas BEFORE the tools were written).
"""
from pathlib import Path

import pandas as pd
import pytest

from agent.data_loader import load_csv
from agent.tools import (Filter, ToolError, aggregate, count_rows, filter_rows,
                         group_aggregate, top_n)

SAMPLE = Path(__file__).parent.parent / "data" / "aid_distributions.csv"
COMPLETED = [Filter("status", "==", "Completed")]


@pytest.fixture(scope="module")
def df():
    return load_csv(SAMPLE.read_bytes(), SAMPLE.name)[0]


# --- T1 count --------------------------------------------------------------
def test_T1_completed_count(df):
    r = count_rows(df, COMPLETED)
    assert r.value == 41
    assert r.filters == ["status == 'Completed'"]


# --- T2 filter -------------------------------------------------------------
def test_T2_pending_in_khan_younis(df):
    r = filter_rows(df, [Filter("governorate", "==", "Khan Younis"),
                         Filter("status", "==", "Pending")])
    assert r.rows_matched == 2
    assert sorted(r.table["distribution_id"]) == ["DST-030", "DST-036"]


def test_text_filter_is_case_insensitive(df):
    assert count_rows(df, [Filter("governorate", "==", "khan younis")]).value == \
           count_rows(df, [Filter("governorate", "==", "Khan Younis")]).value


# --- T3 sum with missing values -------------------------------------------
def test_T3_total_households_completed_reports_missing(df):
    r = aggregate(df, "households_reached", "sum", COMPLETED)
    assert r.value == 10332
    assert r.rows_excluded_missing == 2          # DST-008 and DST-034


# --- T4 average ------------------------------------------------------------
def test_T4_average_quantity(df):
    r = aggregate(df, "quantity", "mean")
    assert round(r.value, 1) == 289.3
    assert isinstance(r.value, float)            # plain Python, not numpy


# --- T5 group --------------------------------------------------------------
def test_T5_quantity_by_governorate(df):
    r = group_aggregate(df, "governorate", "sum", "quantity")
    got = dict(zip(r.table["governorate"], r.table["quantity"]))
    assert got == {"Rafah": 4560, "Deir al-Balah": 4000, "Gaza City": 3410,
                   "Khan Younis": 3030, "North Gaza": 2360}
    assert r.table.iloc[0]["governorate"] == "Rafah"          # sorted high -> low


def test_group_count_needs_no_column(df):
    r = group_aggregate(df, "status", "count")
    got = dict(zip(r.table["status"], r.table["rows"]))
    assert got == {"Completed": 41, "Pending": 10, "Cancelled": 9}


def test_group_by_month(df):
    r = group_aggregate(df, "date", "count", time_grain="month")
    assert list(r.table["date"]) == sorted(r.table["date"])   # chronological
    assert r.table["rows"].sum() == 60


# --- T6 top ----------------------------------------------------------------
def test_T6_top_partner_by_households(df):
    r = top_n(df, "households_reached", n=1, group_by="partner", filters=COMPLETED)
    assert r.table.iloc[0]["partner"] == "Sanad Aid"
    assert r.table.iloc[0]["households_reached"] == 3713


def test_top_rows_without_group(df):
    r = top_n(df, "quantity", n=3)
    assert len(r.table) == 3
    assert r.table["quantity"].iloc[0] == df["quantity"].max()


# --- T10 boundary ----------------------------------------------------------
def test_T10_boundary_greater_than_max_returns_empty(df):
    r = filter_rows(df, [Filter("quantity", ">", 600)])
    assert r.is_empty
    assert r.filters == ["quantity > 600"]


def test_boundary_greater_or_equal_includes_max(df):
    assert count_rows(df, [Filter("quantity", ">=", "600")]).value >= 1   # "600" coerced


def test_date_filter(df):
    r = count_rows(df, [Filter("date", ">=", "2026-03-01"), Filter("date", "<", "2026-04-01")])
    expected = df[(df["date"] >= "2026-03-01") & (df["date"] < "2026-04-01")].shape[0]
    assert r.value == expected


# --- safety / impossible requests -----------------------------------------
@pytest.mark.parametrize("call, message", [
    (lambda d: aggregate(d, "age", "mean"), "not found"),
    (lambda d: aggregate(d, "partner", "sum"), "not numeric"),
    (lambda d: count_rows(d, [Filter("quantity", ">", "a lot")]), "not a number"),
    (lambda d: count_rows(d, [Filter("partner", ">", "A")]), "only works on numbers"),
    (lambda d: group_aggregate(d, "partner", "sum"), "needs a numeric column"),
])
def test_impossible_requests_raise_tool_error(df, call, message):
    with pytest.raises(ToolError, match=message):
        call(df)


def test_tools_never_modify_the_dataframe(df):
    before = df.copy()
    filter_rows(df, COMPLETED, sort_by="quantity")
    group_aggregate(df, "date", "count", time_grain="month")
    top_n(df, "quantity", n=5)
    pd.testing.assert_frame_equal(df, before)
