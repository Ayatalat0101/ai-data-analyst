"""
Stage A - practice page.

Goal: learn the 4 Streamlit building blocks the real app needs
(title, input, button, result) and confirm the Gemini key works.
Run:  streamlit run practice_app.py
"""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Practice - AI Data Analyst", page_icon="🧪")
st.title("🧪 Streamlit practice page")

# ---------------------------------------------------------------
# 1) Text input + button + result
#    Streamlit re-runs this whole file from top to bottom on EVERY
#    interaction. The button returns True only during the re-run
#    that happens right after the click.
# ---------------------------------------------------------------
st.header("1. Text input and button")
name = st.text_input("Your name", placeholder="Deema")

if st.button("Say hello"):
    if not name.strip():
        st.error("Please type a name first.")      # validation, not a crash
    else:
        st.success(f"Hello {name}, Streamlit is working ✅")

# ---------------------------------------------------------------
# 2) File upload -> Pandas -> preview
# ---------------------------------------------------------------
st.header("2. Upload a CSV")
uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded is None:
    st.info("No file yet. Try data/aid_distributions.csv")
else:
    try:
        df = pd.read_csv(uploaded)
        st.write(f"**{len(df)} rows × {len(df.columns)} columns**")
        st.dataframe(df.head())
    except Exception as err:                      # unreadable file
        st.error(f"Could not read this file: {err}")

# ---------------------------------------------------------------
# 3) session_state: a value that survives re-runs
#    A normal Python variable is reset on every re-run.
# ---------------------------------------------------------------
st.header("3. session_state counter")
if "clicks" not in st.session_state:
    st.session_state.clicks = 0
if st.button("Click me"):
    st.session_state.clicks += 1
st.write(f"You clicked {st.session_state.clicks} times")

# ---------------------------------------------------------------
# 4) Check the Gemini key (read from .streamlit/secrets.toml)
# ---------------------------------------------------------------
st.header("4. Test the Gemini connection")
if st.button("Test Gemini"):
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:              # no secrets.toml file at all
        key = ""
    if not key or key.startswith("paste-"):
        st.warning("No key found. Create .streamlit/secrets.toml from the example file.")
    else:
        try:
            from google import genai
            client = genai.Client(api_key=key)
            model = st.secrets.get("GEMINI_MODEL", "gemini-flash-latest")
            reply = client.models.generate_content(
                model=model,
                contents="Reply with exactly: Gemini is connected",
            )
            st.success(f"{model} says: {reply.text}")
        except Exception as err:
            st.error(f"Gemini call failed: {err}")
