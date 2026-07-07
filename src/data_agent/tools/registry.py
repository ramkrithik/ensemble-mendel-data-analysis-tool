"""Tool schemas (the model's contract) and their runtime bindings.

Three tools shape the agent's behaviour:

* ``run_python`` — the workhorse: execute pandas code, observe stdout/traceback.
* ``ask_clarifying_question`` — lets the agent pause and ask the user when the
  request is ambiguous, rather than guessing.
* ``final_answer`` — a structured close-out so the last turn is machine-checkable
  (used by the evaluation harness) instead of free-form prose.

The executors for ``run_python`` are bound per-run to a specific DataFrame by the
agent; the schemas below are static and safe to hand to the LLM as-is.
"""

from __future__ import annotations

from typing import Any

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "run_python",
        "description": (
            "Execute Python (pandas as `pd`, numpy as `np`) against the loaded "
            "dataset, which is already available as the DataFrame `df`. "
            "You MUST `print(...)` any value you want to see — return values are "
            "discarded. Do not read or write files, import os/sys, or access the "
            "network. If the code errors, you'll get the traceback back and should "
            "fix it and try again."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Self-contained Python to execute. Reference `df`.",
                },
                "intent": {
                    "type": "string",
                    "description": "One sentence: what this snippet is trying to learn.",
                },
            },
            "required": ["code", "intent"],
        },
    },
    {
        "name": "ask_clarifying_question",
        "description": (
            "Ask the user ONE focused clarifying question when the request is "
            "ambiguous enough that guessing risks the wrong analysis (e.g. an "
            "undefined metric, unclear grouping, or missing time window). Prefer "
            "reasonable assumptions for minor ambiguity; reserve this for genuine "
            "forks. Do not ask more than necessary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The single clarifying question to ask the user.",
                },
                "why": {
                    "type": "string",
                    "description": "Briefly, why the answer changes the analysis.",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "final_answer",
        "description": (
            "Deliver the final analysis. Call this exactly once, when you have "
            "enough evidence from run_python to answer the user's question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Plain-language answer to the user's question.",
                },
                "key_findings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Bullet-point findings, each grounded in computed output.",
                },
                "methodology": {
                    "type": "string",
                    "description": "Brief note on how the analysis was performed.",
                },
            },
            "required": ["summary", "key_findings"],
        },
    },
]

# Names that terminate or interrupt the loop rather than producing an observation.
TERMINAL_TOOLS = frozenset({"final_answer"})
INTERRUPTING_TOOLS = frozenset({"ask_clarifying_question"})

# run_python is bound at runtime (needs the DataFrame). ask_clarifying_question
# and final_answer are handled directly by the agent loop, so the executor map
# holds only what is genuinely stateless-callable here. Kept for symmetry and to
# make the "schema <-> executor" split explicit.
TOOL_EXECUTORS: dict[str, Any] = {}
