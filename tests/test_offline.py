"""Offline tests — no API key or network required.

These exercise the deterministic parts of the system (config, dataset profiling,
the code-execution sandbox and its guard-rails, and the full agent loop against a
scripted fake LLM) so the whole reason->act->observe->respond cycle is verifiable
without hitting a provider.

Run with:  uv run pytest        (or)  uv run python -m pytest tests/test_offline.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from data_agent.agent import DataAnalysisAgent
from data_agent.config import load_config
from data_agent.llm.base import LLMError, LLMResponse, ToolCall
from data_agent.tools.code_executor import CodeExecutor
from data_agent.tools.dataset import Dataset

DATA = Path(__file__).resolve().parent.parent / "data" / "sales.csv"


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path):
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
    cfg = load_config(env_file=None)
    # Redirect traces into the test's tmp dir.
    object.__setattr__(cfg, "trace_dir", tmp_path / "traces")
    return cfg


@pytest.fixture
def dataset():
    return Dataset.from_csv(DATA)


class ScriptedLLM:
    """A fake LLMClient that replays a fixed list of LLMResponses.

    Lets us drive the agent loop deterministically — including an error->retry
    path — without any network call.
    """

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.model = "scripted-model"
        self.calls: list[dict] = []

    def complete(self, *, system, messages, tools=None):
        self.calls.append({"system": system, "messages": messages})
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        return self._responses.pop(0)


# ── Dataset / profile ────────────────────────────────────────────────────────

def test_dataset_loads_and_profiles(dataset):
    assert dataset.df.shape == (30, 8)
    profile = dataset.profile()
    assert "region" in profile
    assert "Numeric summary" in profile
    assert "30 rows" in profile


def test_dataset_missing_file():
    with pytest.raises(FileNotFoundError):
        Dataset.from_csv("does/not/exist.csv")


# ── Code executor: happy path, error path, guard-rails ───────────────────────

def test_executor_runs_pandas(dataset):
    ex = CodeExecutor(dataset.df)
    res = ex.run("print(df['category'].nunique())")
    assert res.ok
    assert res.stdout.strip() == "3"


def test_executor_captures_traceback(dataset):
    ex = CodeExecutor(dataset.df)
    res = ex.run("print(df['nonexistent_column'].sum())")
    assert not res.ok
    assert "KeyError" in res.error
    assert "STATUS: error" in res.as_tool_content()


@pytest.mark.parametrize("bad", [
    "import os",
    "open('/etc/passwd')",
    "__import__('os').system('ls')",
    "df.to_csv('/tmp/leak.csv')",
    "print(df.__class__.__bases__)",
])
def test_executor_blocks_dangerous_code(dataset, bad):
    ex = CodeExecutor(dataset.df)
    res = ex.run(bad)
    assert not res.ok
    assert "Refused" in res.error


def test_executor_does_not_mutate_source(dataset):
    original_rows = len(dataset.df)
    ex = CodeExecutor(dataset.df)
    ex.run("df.drop(df.index, inplace=True); print(len(df))")
    assert len(dataset.df) == original_rows  # source DataFrame untouched


# ── Agent loop with a scripted LLM ────────────────────────────────────────────

def _text(t):  # helper: assistant text-only turn
    return LLMResponse(text=t, tool_calls=[], stop_reason="end_turn", raw_content=[{"type": "text", "text": t}])


def _tool(tc: ToolCall, stop="tool_use"):
    return LLMResponse(text="", tool_calls=[tc], stop_reason=stop, raw_content=[])


def test_agent_full_loop_with_self_correction(config, dataset):
    """Model runs broken code, sees the traceback, fixes it, then finalises."""
    scripted = ScriptedLLM([
        _tool(ToolCall("t1", "run_python",
                       {"code": "print(df['bad'].mean())", "intent": "avg"})),
        _tool(ToolCall("t2", "run_python",
                       {"code": "print(df['unit_price'].mean())", "intent": "avg"})),
        _tool(ToolCall("t3", "final_answer",
                       {"summary": "Average unit price computed.",
                        "key_findings": ["Mean unit price is 442.0"],
                        "methodology": "mean of unit_price"})),
    ])
    agent = DataAnalysisAgent(dataset, scripted, config)
    result = agent.run("What is the average unit price?", include_few_shot=False)

    assert result.status == "answered"
    assert result.key_findings
    # The error turn + the fix turn + the finalise turn = 3 model calls.
    assert len(scripted.calls) == 3
    # Trace file exists and records the error observation.
    trace_text = Path(result.trace_path).read_text()
    assert "KeyError" in trace_text
    assert "final_answer" in trace_text


def test_agent_clarifying_question(config, dataset):
    scripted = ScriptedLLM([
        _tool(ToolCall("c1", "ask_clarifying_question",
                       {"question": "Best by revenue or by units?",
                        "why": "changes the ranking"})),
    ])
    agent = DataAnalysisAgent(dataset, scripted, config)
    result = agent.run("Which product is best?", include_few_shot=False)
    assert result.status == "needs_clarification"
    assert "revenue" in result.clarifying_question.lower()


def test_agent_handles_llm_failure_gracefully(config, dataset):
    class BrokenLLM:
        model = "broken"

        def complete(self, **_):
            raise LLMError("simulated outage")

    agent = DataAnalysisAgent(dataset, BrokenLLM(), config)
    result = agent.run("anything", include_few_shot=False)
    assert result.status == "error"
    assert "unavailable" in result.answer.lower()


def test_agent_rejects_empty_question(config, dataset):
    agent = DataAnalysisAgent(dataset, ScriptedLLM([]), config)
    result = agent.run("   ", include_few_shot=False)
    assert result.status == "error"


def test_agent_enforces_step_budget(config, dataset):
    object.__setattr__(config, "agent_max_steps", 2)
    # Always asks to run code, never finalises -> should hit the budget.
    looping = ScriptedLLM([
        _tool(ToolCall(f"t{i}", "run_python",
                       {"code": "print(1)", "intent": "loop"}))
        for i in range(5)
    ])
    agent = DataAnalysisAgent(dataset, looping, config)
    result = agent.run("loop forever", include_few_shot=False)
    assert result.status == "max_steps"
    assert result.steps_used == 2
