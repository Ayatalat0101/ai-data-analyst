"""
AI Data Analyst - Streamlit interface.

This file only does presentation and state. It never touches Pandas directly
and never decides anything: questions go to agent.answer(), which returns an
AgentResponse that we display.

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from agent.agent import AgentResponse, answer
from agent.data_loader import DataError, DatasetProfile, load_csv
from agent.llm_planner import planner_from_secrets
from charts import chart_for

SAMPLE = Path(__file__).parent / "data" / "aid_distributions.csv"

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
def init_state() -> None:
    defaults = {"df": None, "profile": None, "file_sig": None, "source": None,
                "history": [], "queued": None, "load_error": None, "use_llm": False,
                "ai_used": 0}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def set_dataset(raw: bytes, name: str, source: str) -> None:
    """Load a file. A DIFFERENT file resets the chat (T13): old answers
    belong to old data and must not be shown next to the new one."""
    sig = hashlib.md5(raw).hexdigest() + name
    if sig == st.session_state.file_sig:
        return                                     # same file, normal re-run
    try:
        df, profile = load_csv(raw, name)
    except DataError as err:
        st.session_state.update(df=None, profile=None, file_sig=sig, history=[],
                                load_error=str(err), source=source)
        return
    st.session_state.update(df=df, profile=profile, file_sig=sig, history=[],
                            load_error=None, source=source)


def clear_dataset() -> None:
    st.session_state.update(df=None, profile=None, file_sig=None, source=None,
                            history=[], load_error=None)


def ai_limit() -> int:
    """AI questions per visitor session (protects the shared free quota on the public demo)."""
    try:
        return int(st.secrets.get("AI_QUESTIONS_PER_SESSION", 20))
    except Exception:
        return 20


@st.cache_resource
def get_llm_planner():
    """Built once per server from secrets.toml. None = no key -> rules only."""
    try:
        secrets = dict(st.secrets)
    except Exception:                  # no secrets.toml file
        return None
    return planner_from_secrets(secrets)


def queue(question: str) -> None:
    """Button callback: the question is processed on the next re-run."""
    st.session_state.queued = question


# ----------------------------------------------------------------------------
# Sidebar: data source + helpers
# ----------------------------------------------------------------------------
def sidebar() -> None:
    with st.sidebar:
        st.header("1 · Your data")
        up = st.file_uploader("Upload a CSV file (max 10 MB)", type=["csv"],
                              help="Excel users: File → Save As → CSV UTF-8")
        if up is not None:
            set_dataset(up.getvalue(), up.name, "upload")
        elif st.session_state.source == "upload":
            clear_dataset()                        # user removed the file

        if st.button("Use sample data", use_container_width=True,
                     help="60 synthetic humanitarian aid distributions"):
            set_dataset(SAMPLE.read_bytes(), SAMPLE.name, "sample")

        st.divider()
        st.header("2 · Try a question")
        for q in ["How many distributions were completed?",
                  "Total quantity by governorate",
                  "Which partner reached the most households in completed distributions?",
                  "Total households reached per month",
                  "Which is the best partner?",
                  "What is the average beneficiary age?"]:
            st.button(q, on_click=queue, args=(q,), use_container_width=True,
                      disabled=st.session_state.df is None, key=f"ex_{q}")

        st.divider()
        st.header("3 · Planner")
        llm = get_llm_planner()
        if llm is None:
            st.caption("No API key found → **rule-based planner** (offline). "
                       "Add GEMINI_API_KEY to .streamlit/secrets.toml to enable AI planning.")
            st.session_state.use_llm = False
        else:
            choice = st.radio("Who plans the analysis?", ["🤖 AI (Gemini)", "📏 Rules (offline)"],
                              help="Both go through the same validator and tools. "
                                   "If the AI is unavailable, rules answer automatically.")
            st.session_state.use_llm = choice.startswith("🤖")
            st.caption(f"Chain: {llm.name} → rules")
            if st.session_state.use_llm:
                left = max(0, ai_limit() - st.session_state.ai_used)
                st.caption(f"AI questions left in this session: **{left}/{ai_limit()}** "
                           "(then the rule planner answers).")
        if st.session_state.history:
            st.button("🗑 Clear conversation", on_click=lambda: st.session_state.update(history=[]),
                      use_container_width=True)
        if st.session_state.use_llm:
            st.caption("🔒 Rows never leave this app. In AI mode, the question plus column names, "
                       "types and category values are sent to Google Gemini to plan the analysis. "
                       "Calculations run locally with Pandas.")
        else:
            st.caption("🔒 Nothing leaves this app. Calculations run locally with Pandas.")


# ----------------------------------------------------------------------------
# Dataset overview
# ----------------------------------------------------------------------------
def display_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes(include=["datetime"]).columns:
        out[c] = out[c].dt.date
    return out


def overview(df: pd.DataFrame, p: DatasetProfile) -> None:
    st.subheader(f"📄 {p.file_name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{p.rows:,}")
    c2.metric("Columns", len(p.columns))
    c3.metric("Missing cells", sum(c.missing for c in p.columns))
    c4.metric("Duplicate rows", p.duplicate_rows)
    for w in p.warnings:
        st.warning(w, icon="⚠️")

    with st.expander("Columns and types", expanded=not st.session_state.history):
        st.dataframe(pd.DataFrame([{
            "column": c.name, "type": c.kind, "missing": c.missing, "unique": c.unique,
            "values / range": ", ".join(c.values) if c.values else
                              (f"{c.min} → {c.max}" if c.min is not None else ", ".join(c.examples)),
        } for c in p.columns]), hide_index=True, use_container_width=True)
    with st.expander("Preview: first 5 rows", expanded=not st.session_state.history):
        st.dataframe(display_table(df.head()), hide_index=True, use_container_width=True)


# ----------------------------------------------------------------------------
# One assistant message
# ----------------------------------------------------------------------------
ICONS = {"clarification": "❓", "rejected": "🚫", "no_results": "🔎", "error": "⚠️"}


def render_response(r: AgentResponse, turn: int, is_last: bool) -> None:
    if r.kind == "answer":
        st.markdown(r.message)
        res = r.result
        if res.table is not None and len(res.table):
            fig = chart_for(r.plan, res)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{turn}")
            st.dataframe(display_table(res.table), hide_index=True, use_container_width=True)
            st.download_button("⬇ Download result (CSV)", res.table.to_csv(index=False),
                               file_name=f"result_{turn + 1}.csv", key=f"dl_{turn}")
    elif r.kind == "clarification":
        st.info(r.message, icon=ICONS[r.kind])
        if is_last and r.options:
            st.caption("Choose one, or type a clearer question:")
            for i, opt in enumerate(r.options):
                st.button(opt, on_click=queue, args=(opt,), key=f"opt_{turn}_{i}")
    elif r.kind in ("rejected", "error"):
        st.error(r.message, icon=ICONS[r.kind])
    elif r.kind == "no_results":
        st.warning(r.message, icon=ICONS[r.kind])

    if r.explanation:
        with st.expander("🔍 How I got this"):
            for line in r.explanation:
                st.markdown(f"- {line}")
    elif r.plan is not None:
        st.caption(f"Planner: {r.planner_used} · decision: `{r.plan.action}`"
                   + (f" ({r.plan.reject_reason})" if r.plan.reject_reason else ""))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    init_state()
    sidebar()

    st.title("📊 AI Data Analyst")
    st.caption("Ask questions about a CSV in plain English. The agent picks an approved "
               "analysis, checks it against your columns, and shows how every number was calculated.")

    if st.session_state.load_error:
        st.error(f"**This file can't be used:** {st.session_state.load_error}", icon="🚫")
    df, profile = st.session_state.df, st.session_state.profile
    if df is None:
        if not st.session_state.load_error:
            st.info("⬅ Upload a CSV file, or click **Use sample data** to start.", icon="📂")
        st.chat_input("Upload a file first", disabled=True)
        return

    # handle new input BEFORE drawing the history, so it appears in this run
    typed = st.chat_input("Ask a question about your data…")
    question = st.session_state.queued or typed
    st.session_state.queued = None
    if question:
        llm, name = None, "rules"
        if st.session_state.get("use_llm"):
            if st.session_state.ai_used < ai_limit():
                llm = get_llm_planner()
                st.session_state.ai_used += 1
            else:
                # Public demo: one visitor must not spend the whole free Gemini quota.
                name = "rules (AI limit for this session reached)"
        with st.spinner("Planning the analysis…"):
            st.session_state.history.append(
                {"q": question, "r": answer(question, df, profile, planner=llm, planner_name=name)})

    overview(df, profile)
    st.divider()
    history = st.session_state.history
    for i, turn in enumerate(history):
        with st.chat_message("user"):
            st.markdown(turn["q"])
        with st.chat_message("assistant", avatar="📊"):
            render_response(turn["r"], i, is_last=(i == len(history) - 1))


main()
