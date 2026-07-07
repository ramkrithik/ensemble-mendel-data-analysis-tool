"""Streamlit UI for the data-analysis agent.

Run with:  uv run streamlit run app/streamlit_app.py

Upload (or point at) a CSV, ask a question, and watch the agent's reasoning
chain — model thoughts, generated code, execution results — stream into the page
as it works. State (dataset + conversation) persists across reruns via
``st.session_state``, giving simple conversation persistence within a session.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

# Allow `streamlit run app/streamlit_app.py` from the repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_agent.agent import DataAnalysisAgent  # noqa: E402
from data_agent.config import load_config  # noqa: E402
from data_agent.llm import build_client  # noqa: E402
from data_agent.tools.dataset import Dataset  # noqa: E402

st.set_page_config(page_title="Data Analysis Agent", page_icon="📊", layout="wide")
st.title("📊 Data Analysis Agent")
st.caption("Upload a CSV, ask a question — the agent writes & runs pandas to answer.")


@st.cache_resource(show_spinner=False)
def _config():
    return load_config()


@st.cache_resource(show_spinner=False)
def _llm(_provider: str, _model: str):
    # Keyed on provider+model so a config change rebuilds the client.
    return build_client(_config())


def _load_dataset(uploaded, fallback_path: str) -> Dataset | None:
    if uploaded is not None:
        tmp = Path(tempfile.gettempdir()) / uploaded.name
        tmp.write_bytes(uploaded.getvalue())
        return Dataset.from_csv(tmp)
    if fallback_path.strip():
        return Dataset.from_csv(fallback_path.strip())
    return None


cfg = _config()

with st.sidebar:
    st.header("Configuration")
    st.write(f"**Provider:** `{cfg.provider}`")
    st.write(f"**Model:** `{cfg.model}`")
    st.write(f"**Max steps:** {cfg.agent_max_steps}")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    fallback = st.text_input("…or path to a CSV", value="data/sales.csv")

try:
    dataset = _load_dataset(uploaded, fallback)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

if dataset is None:
    st.info("Upload a CSV or provide a path in the sidebar to begin.")
    st.stop()

with st.expander("Dataset profile", expanded=False):
    st.dataframe(dataset.df.head(20), use_container_width=True)
    st.code(dataset.profile(), language="text")

question = st.text_input(
    "Ask a question about this dataset",
    placeholder="e.g. Which region generated the most net revenue?",
)

if st.button("Analyse", type="primary", disabled=not question.strip()):
    llm = _llm(cfg.provider, cfg.model)
    agent = DataAnalysisAgent(dataset, llm, cfg)

    with st.status("Agent working…", expanded=True) as status:
        st.write("Running reason → plan → act → observe loop…")
        result = agent.run(question)
        status.update(label=f"Done — {result.status}", state="complete")

    if result.status == "needs_clarification":
        st.warning(f"**Clarifying question:** {result.clarifying_question}")
    elif result.status == "answered":
        st.subheader("Answer")
        st.write(result.answer)
        if result.key_findings:
            st.subheader("Key findings")
            for f in result.key_findings:
                st.markdown(f"- {f}")
        if result.methodology:
            st.caption(f"Methodology: {result.methodology}")
    else:
        st.error(result.answer)

    # Show the full reasoning trace read back from the JSONL file.
    with st.expander("Reasoning trace", expanded=False):
        trace_file = Path(result.trace_path)
        if trace_file.is_file():
            st.code(trace_file.read_text(encoding="utf-8"), language="json")
    st.caption(f"Run {result.run_id} · {result.steps_used} steps · trace: {result.trace_path}")
