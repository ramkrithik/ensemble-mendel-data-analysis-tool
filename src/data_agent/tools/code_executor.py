"""Sandboxed-ish Python execution over the loaded DataFrame.

This is the agent's "act" surface: the model writes pandas code, we run it, and
the captured stdout (or traceback) becomes the "observe" the model reasons about
next. It is the mechanism that *forces* the full agent loop — when generated code
raises, the model sees the error and must self-correct.

Guard-rails (defence in depth, not a true security sandbox):
  * A denylist of obviously dangerous names/imports blocks the most common
    footguns (os, sys, subprocess, open, eval, __import__, network, dunder access).
  * Execution runs with a restricted global namespace exposing only pandas,
    numpy, and the DataFrame ``df``.
  * stdout is captured and truncated so a runaway ``print(df)`` can't blow up
    the context window.

For a production system you would replace this with a real sandbox (subprocess
with seccomp, a container, or the Anthropic server-side code-execution tool).
The interface here is deliberately narrow so that swap is localised.
"""

from __future__ import annotations

import contextlib
import io
import re
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Substrings that, if present, cause us to refuse execution outright.
_DENYLIST = (
    "import os", "import sys", "import subprocess", "import shutil",
    "import socket", "import requests", "import urllib", "importlib",
    "__import__", "eval(", "exec(", "compile(", "open(", "input(",
    "globals(", "locals(", "getattr(", "setattr(", "delattr(",
    "os.", "sys.", "subprocess", "__builtins__", "__globals__",
    "__subclasses__", "__class__", "__bases__", "__mro__",
    "to_csv", "to_pickle", "to_parquet", "read_pickle",
)

# Output is truncated to keep tool results (and thus context) bounded.
_MAX_OUTPUT_CHARS = 6000


@dataclass
class ExecResult:
    """Outcome of one code-execution attempt."""

    ok: bool
    stdout: str
    error: str  # traceback text when ok is False, else ""

    def as_tool_content(self) -> str:
        """Render for feeding back to the model as a tool_result."""
        if self.ok:
            body = self.stdout if self.stdout.strip() else "(code ran, no output printed)"
            return f"STATUS: success\nOUTPUT:\n{body}"
        return (
            "STATUS: error\n"
            "The code raised an exception. Read the traceback, fix the code, "
            "and call run_python again.\nTRACEBACK:\n"
            f"{self.error}"
        )


def _screen(code: str) -> str | None:
    """Return a refusal reason if the code trips a guard-rail, else None."""
    lowered = code.lower()
    for bad in _DENYLIST:
        if bad in lowered:
            return f"Refused: code contains a disallowed operation ({bad!r})."
    # Block dunder attribute access used for sandbox escapes.
    if re.search(r"__\w+__", code):
        return "Refused: dunder attribute access is not allowed."
    return None


class CodeExecutor:
    """Runs model-authored pandas code against a fixed DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        # Work on a defensive copy so a mutating snippet can't corrupt the
        # canonical dataset across steps.
        self._df = df

    def run(self, code: str) -> ExecResult:
        refusal = _screen(code)
        if refusal is not None:
            return ExecResult(ok=False, stdout="", error=refusal)

        sandbox_globals = {
            "__builtins__": _safe_builtins(),
            "pd": pd,
            "np": np,
            "df": self._df.copy(),
        }
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(code, "<agent_code>", "exec"), sandbox_globals)  # noqa: S102
        except Exception:  # noqa: BLE001 - surface any error back to the model
            return ExecResult(
                ok=False,
                stdout=_truncate(stdout.getvalue()),
                error=_truncate(traceback.format_exc()),
            )
        return ExecResult(ok=True, stdout=_truncate(stdout.getvalue()), error="")


def _safe_builtins() -> dict[str, object]:
    """A minimal builtins map — enough for data work, nothing filesystem/eval."""
    allowed = [
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
        "float", "int", "len", "list", "map", "max", "min", "print", "range",
        "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple",
        "zip", "isinstance", "type", "repr", "format",
    ]
    import builtins as _b

    return {name: getattr(_b, name) for name in allowed}


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[:limit]
    return f"{head}\n... [output truncated at {limit} chars]"
