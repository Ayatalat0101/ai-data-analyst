"""
End-to-end tests: question in -> answer / clarification / rejection out.
IDs match PROJECT_PLAN.md §10. Runs with the rule planner (no API key needed).
"""
from pathlib import Path

import pytest

from agent.agent import answer
from agent.data_loader import load_csv
from agent.plan import Plan
from agent.validator import validate

SAMPLE = Path(__file__).parent.parent / "data" / "aid_distributions.csv"


@pytest.fixture(scope="module")
def data():
    return load_csv(SAMPLE.read_bytes(), SAMPLE.name)


def ask(data, q):
    df, profile = data
    return answer(q, df, profile)


# --- normal questions ------------------------------------------------------
def test_T1_count(data):
    r = ask(data, "How many distributions were completed?")
    assert r.kind == "answer" and r.result.value == 41


def test_T2_filter(data):
    r = ask(data, "Show pending distributions in Khan Younis")
    assert r.kind == "answer"
    assert sorted(r.result.table["distribution_id"]) == ["DST-030", "DST-036"]


def test_T3_sum_reports_missing(data):
    r = ask(data, "Total households reached in completed distributions")
    assert r.result.value == 10332
    assert r.result.rows_excluded_missing == 2
    assert any("2 row(s) excluded" in line for line in r.explanation)


def test_T4_average(data):
    r = ask(data, "Average quantity per distribution")
    assert round(r.result.value, 1) == 289.3


def test_T5_group(data):
    r = ask(data, "Total quantity by governorate")
    assert r.plan.action == "group_aggregate"
    assert r.result.table.iloc[0].tolist() == ["Rafah", 4560]


def test_T6_top(data):
    r = ask(data, "Which partner reached the most households in completed distributions?")
    assert r.plan.action == "top_n"
    assert "Sanad Aid" in r.message and "3,713" in r.message


# --- ambiguity: must ask, must NOT return a number -----------------------
@pytest.mark.parametrize("q", ["Which is the best partner?", "Show me the big distributions"])
def test_T7_T8_ambiguous_asks(data, q):
    r = ask(data, q)
    assert r.kind == "clarification"
    assert r.result is None                      # no number was produced
    assert len(r.options) >= 2


def test_clarification_options_give_the_RIGHT_answer(data):
    """Clicking an option must answer THAT option - not just any answer.
    (Bug found in the UI: 'highest total households_reached' returned a row count.)"""
    r = ask(data, "Which is the best partner?")
    expected = {"quantity": ("quantity", "sum"), "households reached": ("households_reached", "sum"),
                "unit cost usd": ("unit_cost_usd", "mean"), "most rows": None}
    for option in r.options:
        res = ask(data, option)
        assert res.kind == "answer", option
        want = next(v for k, v in expected.items() if k in option)
        if want:
            assert (res.plan.column, res.plan.agg) == want, option
        else:
            assert res.plan.agg == "count", option


def test_underscore_column_names_are_understood(data):
    r = ask(data, "Which partner has the highest total households_reached?")
    assert r.plan.column == "households_reached"
    assert "Sanad Aid" in r.message


def test_rank_without_measure_asks_instead_of_counting(data):
    r = ask(data, "Which partner is the highest?")
    assert r.kind == "clarification"


# --- missing column / boundary / scope / safety ---------------------------
def test_T9_missing_column(data):
    r = ask(data, "What is the average beneficiary age?")
    assert r.kind == "rejected" and r.plan.reject_reason == "missing_column"
    assert "quantity" in r.message                     # lists real columns


def test_T10_boundary_empty(data):
    r = ask(data, "Distributions with quantity greater than 600")
    assert r.kind == "no_results"
    assert "50 to 600" in r.message                    # helpful range hint


def test_T11_out_of_scope(data):
    r = ask(data, "Predict next month's households")
    assert r.kind == "rejected" and r.plan.reject_reason == "out_of_scope"


@pytest.mark.parametrize("q", ["Delete all rows", "import os; os.remove('data.csv')",
                               "drop the status column", "__import__('os')"])
def test_T14_unsafe_rejected(data, q):
    r = ask(data, q)
    assert r.kind == "rejected" and r.plan.reject_reason == "unsafe"


# --- never answer a different question -----------------------------------
@pytest.mark.parametrize("q, suggestion", [
    ("How many distributions in Khan Yunis?", "Khan Younis"),
    ("Total quantity in rafa", "Rafah"),
    ("Show distributions for Nour Releif", "Nour Relief"),
])
def test_misspelled_value_asks_did_you_mean(data, q, suggestion):
    r = ask(data, q)
    assert r.kind == "clarification"
    assert suggestion in r.message
    assert ask(data, r.options[0]).kind == "answer"    # the one-click fix works


@pytest.mark.parametrize("q", ["", "  ", "hi", "x" * 301])
def test_invalid_question_text(data, q):
    assert ask(data, q).kind == "error"


# --- the validator catches a bad LLM plan ---------------------------------
def test_validator_rejects_invented_column(data):
    _, profile = data
    bad = Plan(action="aggregate", column="beneficiaries", agg="sum", reason="LLM guess")
    assert validate(bad, profile).action == "reject"


def test_validator_fixes_column_case(data):
    _, profile = data
    p = validate(Plan(action="aggregate", column="Households Reached", agg="sum", reason=""), profile)
    assert p.column == "households_reached"


def test_validator_blocks_sum_of_text(data):
    _, profile = data
    p = validate(Plan(action="aggregate", column="partner", agg="sum", reason=""), profile)
    assert p.action == "reject"


def test_planner_crash_falls_back_to_rules(data):
    df, profile = data

    def broken_llm(q, prof):
        raise TimeoutError("quota exceeded")

    r = answer("How many distributions were completed?", df, profile, planner=broken_llm, planner_name="gemini")
    assert r.kind == "answer" and r.result.value == 41
    assert "fallback" in r.planner_used
