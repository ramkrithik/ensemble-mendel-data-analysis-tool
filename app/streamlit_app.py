"""Streamlit UI for the PC-build agent.

Run with:  uv run streamlit run app/streamlit_app.py

A chat interface over the agent. The conversation (and the agent's own state)
persists across Streamlit reruns via ``st.session_state``, so follow-up messages
act as feedback and the agent amends the previous build. Each proposed build is
rendered as a table; the full reasoning trace for every turn is available in an
expander.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run app/streamlit_app.py` from the repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pc_agent.agent import PCBuildAgent  # noqa: E402
from pc_agent.catalog import Catalog  # noqa: E402
from pc_agent.config import load_config  # noqa: E402
from pc_agent.llm import build_client  # noqa: E402

st.set_page_config(page_title="PC Build Agent", page_icon="🖥️", layout="centered")
st.title("🖥️ PC Build Agent")
st.caption("Describe your ideal PC. Give feedback to amend the build — it remembers.")


@st.cache_resource(show_spinner="Loading catalog & model…")
def _make_agent():
    cfg = load_config()
    catalog = Catalog.load(cfg.data_dir)
    llm = build_client(cfg)
    return PCBuildAgent(catalog, llm, cfg), cfg


try:
    agent, cfg = _make_agent()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Startup failed: {exc}")
    st.stop()

with st.sidebar:
    st.header("Configuration")
    st.write(f"**Provider:** `{cfg.provider}`")
    st.write(f"**Model:** `{cfg.model}`")
    st.write(f"**Max steps:** {cfg.agent_max_steps}")
    st.write(f"**Dataset:** `{cfg.data_dir}/`")
    if st.button("🔄 New conversation"):
        st.session_state.pop("history", None)
        _make_agent.clear()
        st.rerun()

# Conversation history: list of dicts {role, kind, payload}
if "history" not in st.session_state:
    st.session_state.history = []


def _render_build(turn) -> None:
    build = turn.build
    rows = [
        {"Category": p.category, "Part": p.name,
         "Price": f"${p.price:,.2f}" if p.price is not None else "n/a"}
        for p in build.parts
    ]
    st.table(rows)
    st.markdown(f"**Total: ${build.total_price:,.2f}**")
    if turn.report and turn.report.warnings:
        for w in turn.report.warnings:
            st.info(w.detail)
    if turn.message:
        st.markdown(f"**Why:** {turn.message}")


# Replay history
for item in st.session_state.history:
    with st.chat_message(item["role"]):
        if item["kind"] == "text":
            st.markdown(item["payload"])
        elif item["kind"] == "build":
            _render_build(item["payload"])
            with st.expander("Reasoning trace"):
                tp = Path(item["payload"].trace_path)
                if tp.is_file():
                    st.code(tp.read_text(encoding="utf-8"), language="json")

prompt = st.chat_input("e.g. A 1080p gaming PC under $1200, prefer AMD")
if prompt:
    st.session_state.history.append({"role": "user", "kind": "text", "payload": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reasoning → searching → checking compatibility…"):
            turn = agent.chat(prompt)
        if turn.status == "proposed" and turn.build is not None:
            _render_build(turn)
            with st.expander("Reasoning trace"):
                tp = Path(turn.trace_path)
                if tp.is_file():
                    st.code(tp.read_text(encoding="utf-8"), language="json")
            st.session_state.history.append({"role": "assistant", "kind": "build", "payload": turn})
        elif turn.status == "needs_clarification":
            st.warning(f"❓ {turn.clarifying_question}")
            st.session_state.history.append(
                {"role": "assistant", "kind": "text", "payload": f"❓ {turn.clarifying_question}"})
        else:
            st.error(turn.message)
            st.session_state.history.append(
                {"role": "assistant", "kind": "text", "payload": turn.message})
