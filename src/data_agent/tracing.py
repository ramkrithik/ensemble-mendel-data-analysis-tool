"""Structured, inspectable tracing of every agent step.

Each run gets a :class:`Tracer` that both logs human-readable lines (for the
console) and appends structured JSONL records to ``TRACE_DIR/<run_id>.jsonl``.
The JSONL trace is the artefact the evaluation harness and the "agent run
report" read back — it captures the full reasoning chain: model thoughts, tool
calls with arguments, observations, and the final answer.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("data_agent.trace")


@dataclass
class TraceEvent:
    """One entry in the reasoning trace."""

    step: int
    kind: str  # "run_start" | "model_thought" | "tool_call" | "observation" |
    #            "clarifying_question" | "final_answer" | "error" | "run_end"
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class Tracer:
    """Collects trace events in memory and mirrors them to a JSONL file."""

    def __init__(self, run_id: str, trace_dir: Path) -> None:
        self.run_id = run_id
        self.events: list[TraceEvent] = []
        trace_dir.mkdir(parents=True, exist_ok=True)
        self._path = trace_dir / f"{run_id}.jsonl"
        # Truncate any prior file for this run id.
        self._path.write_text("", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def record(self, step: int, kind: str, **data: Any) -> None:
        event = TraceEvent(step=step, kind=kind, data=data)
        self.events.append(event)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(event), default=str) + "\n")
        log.info("[step %d] %s %s", step, kind, _short(data))


def _short(data: dict[str, Any], limit: int = 160) -> str:
    text = json.dumps(data, default=str)
    return text if len(text) <= limit else text[:limit] + "..."
