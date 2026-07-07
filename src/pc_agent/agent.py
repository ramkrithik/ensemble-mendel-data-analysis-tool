"""The reason -> plan -> act -> observe -> respond loop, with a feedback loop.

:class:`PCBuildAgent` holds one catalog + LLM client and turns a customer request
into a compatibility-checked build. It keeps conversation state across turns, so
:meth:`chat` can be called repeatedly — the second and later calls are how the
customer gives feedback ("cheaper", "make it Intel", "add storage") and the agent
amends the build.

Terminal outcomes per turn:
  * ``proposed``            — a build passed the compatibility engine and was delivered.
  * ``needs_clarification`` — the agent asked the customer one question.
  * ``max_steps``           — the step budget was exhausted (guard-rail).
  * ``error``               — LLM failure or invalid input (graceful fallback).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from pc_agent.catalog import Catalog
from pc_agent.compatibility import CompatibilityChecker
from pc_agent.config import AppConfig
from pc_agent.llm.base import LLMClient, LLMError, LLMResponse
from pc_agent.models import Build, CompatibilityReport, Requirements
from pc_agent.prompts import FEW_SHOT_MESSAGES, build_system_prompt
from pc_agent.tools.registry import (
    INTERRUPTING_TOOLS,
    TERMINAL_TOOLS,
    TOOL_SCHEMAS,
    ToolKit,
)
from pc_agent.tracing import Tracer

_MAX_INPUT_CHARS = 6000


@dataclass
class TurnResult:
    """Outcome of one conversational turn."""

    status: str  # "proposed" | "needs_clarification" | "max_steps" | "error"
    message: str = ""                       # rationale, question, or error text
    build: Build | None = None
    report: CompatibilityReport | None = None
    clarifying_question: str = ""
    steps_used: int = 0
    trace_path: str = ""
    run_id: str = ""

    def render(self) -> str:
        if self.status == "needs_clarification":
            return f"❓ {self.clarifying_question}"
        if self.status != "proposed" or self.build is None:
            return self.message
        lines = ["🖥️  Proposed build:"]
        for p in self.build.parts:
            price = f"${p.price:,.2f}" if p.price is not None else "price n/a"
            lines.append(f"  • {p.category:20s} {p.name}  ({price})")
        lines.append(f"\n  Total: ${self.build.total_price:,.2f}")
        if self.report and self.report.warnings:
            lines.append("\n  Notes:")
            lines.extend(f"    - {w.detail}" for w in self.report.warnings)
        if self.message:
            lines.append(f"\n  Why: {self.message}")
        return "\n".join(lines)


class PCBuildAgent:
    """Drives the agent loop over a catalog, preserving conversation state."""

    def __init__(self, catalog: Catalog, llm: LLMClient, config: AppConfig) -> None:
        self._catalog = catalog
        self._llm = llm
        self._config = config
        self._toolkit = ToolKit(catalog)
        self._checker = CompatibilityChecker(catalog)
        self._system_prompt = build_system_prompt(catalog.summary())
        # Conversation state, preserved across chat() calls for the feedback loop.
        self._messages: list[dict[str, Any]] = list(FEW_SHOT_MESSAGES)
        # Best-effort requirements memory, refined as the conversation continues.
        self._requirements = Requirements()
        self._last_build: Build | None = None

    # ── public API ─────────────────────────────────────────────────────────
    def chat(self, user_message: str) -> TurnResult:
        """One turn: a new request, or feedback on the previous build."""
        run_id = uuid.uuid4().hex[:12]
        tracer = Tracer(run_id, self._config.trace_dir)
        is_feedback = self._last_build is not None
        tracer.record(
            0, "turn_start", message=user_message, is_feedback=is_feedback,
            config=self._config.redacted(),
        )

        problem = _validate(user_message)
        if problem is not None:
            tracer.record(0, "error", reason=problem)
            return TurnResult(status="error", message=problem, run_id=run_id,
                              trace_path=str(tracer.path))

        self._maybe_capture_requirements(user_message)
        self._messages.append({"role": "user", "content": user_message})

        return self._run_loop(tracer, run_id)

    def run(self, user_message: str) -> TurnResult:
        """Alias for a single-shot request (fresh callers)."""
        return self.chat(user_message)

    # ── loop ─────────────────────────────────────────────────────────────────
    def _run_loop(self, tracer: Tracer, run_id: str) -> TurnResult:
        for step in range(1, self._config.agent_max_steps + 1):
            try:
                resp = self._llm.complete(
                    system=self._system_prompt,
                    messages=self._messages,
                    tools=TOOL_SCHEMAS,
                )
            except LLMError as exc:
                tracer.record(step, "error", reason=str(exc))
                return TurnResult(
                    status="error",
                    message=(f"The language model was unavailable after retries: {exc}"),
                    steps_used=step, run_id=run_id, trace_path=str(tracer.path),
                )

            if resp.text:
                tracer.record(step, "model_thought", text=resp.text)

            if resp.stop_reason == "refusal":
                tracer.record(step, "error", reason="model_refusal")
                return TurnResult(status="error",
                                  message="The model declined this request.",
                                  steps_used=step, run_id=run_id,
                                  trace_path=str(tracer.path))

            if not resp.wants_tool:
                # Nudge the model to act through tools rather than free-prose.
                tracer.record(step, "model_thought", note="no_tool_call", text=resp.text)
                self._messages.append({"role": "assistant", "content": resp.raw_content})
                self._messages.append({
                    "role": "user",
                    "content": ("Use the tools: search_components to find parts, "
                                "check_compatibility to verify, then propose_build."),
                })
                continue

            self._messages.append({"role": "assistant", "content": resp.raw_content})
            terminal = self._handle_tools(resp, tracer, step)
            if terminal is not None:
                terminal.steps_used = step
                terminal.run_id = run_id
                terminal.trace_path = str(tracer.path)
                tracer.record(step, "turn_end", status=terminal.status)
                return terminal

        tracer.record(self._config.agent_max_steps, "turn_end", status="max_steps")
        return TurnResult(
            status="max_steps",
            message=("Reached the maximum number of steps without a final build. "
                     "Try adding a budget or narrowing the requirements."),
            steps_used=self._config.agent_max_steps, run_id=run_id,
            trace_path=str(tracer.path),
        )

    def _handle_tools(
        self, resp: LLMResponse, tracer: Tracer, step: int
    ) -> TurnResult | None:
        """Execute every tool call, then decide the turn outcome.

        Crucially, we emit exactly one ``tool_result`` for every ``tool_use`` in
        the assistant turn — including terminal/interrupting tools — *before*
        returning. Leaving a ``tool_use`` without a matching ``tool_result`` makes
        the saved history invalid and the API rejects the *next* (feedback) turn.
        """
        tool_results: list[dict[str, Any]] = []
        outcome: TurnResult | None = None  # set by a terminal/interrupting tool

        for call in resp.tool_calls:
            tracer.record(step, "tool_call", tool=call.name, arguments=call.arguments)

            if call.name in INTERRUPTING_TOOLS:
                question = str(call.arguments.get("question", "")).strip()
                tracer.record(step, "clarifying_question", question=question,
                              why=call.arguments.get("why", ""))
                tool_results.append(_tool_result(
                    call.id, "Clarifying question relayed to the customer; awaiting their reply."))
                outcome = outcome or TurnResult(
                    status="needs_clarification", clarifying_question=question)

            elif call.name in TERMINAL_TOOLS:
                result_text, turn = self._finalise(call.arguments, tracer, step)
                tool_results.append(_tool_result(call.id, result_text))
                if turn is not None:
                    outcome = outcome or turn

            elif call.name == "search_components":
                out = self._toolkit.search_components(call.arguments)
                tracer.record(step, "observation", tool="search_components",
                              category=call.arguments.get("category"),
                              result_preview=out[:500])
                tool_results.append(_tool_result(call.id, out))
            elif call.name == "check_compatibility":
                out = self._toolkit.check_compatibility(call.arguments, self._requirements)
                tracer.record(step, "observation", tool="check_compatibility",
                              parts=call.arguments.get("parts"), report=out)
                tool_results.append(_tool_result(call.id, out))
            else:
                msg = f"Unknown tool {call.name!r}."
                tracer.record(step, "error", reason=msg)
                tool_results.append(_tool_result(call.id, msg, is_error=True))

        # Always close out the tool_use blocks with their results.
        if tool_results:
            self._messages.append({"role": "user", "content": tool_results})
        return outcome

    def _finalise(
        self, args: dict[str, Any], tracer: Tracer, step: int
    ) -> tuple[str, TurnResult | None]:
        """Validate a proposed build. Returns (tool_result_text, terminal_or_None).

        A ``None`` terminal means the build failed the compatibility gate; the
        returned text tells the model to fix it and the loop continues.
        """
        parts = args.get("parts", {}) or {}
        rationale = str(args.get("rationale", "")).strip()
        build = self._toolkit.build_from_parts(parts)
        # Re-run the deterministic check ourselves — never trust the model's word
        # that a build is compatible. This is the self-critique safety net.
        report = self._checker.check(build, self._requirements)
        build.rationale = rationale

        tracer.record(step, "propose_build", parts=parts, total=build.total_price,
                      compatible=report.compatible, report=report.summary())

        if not report.compatible:
            tracer.record(step, "reject_proposal", errors=[e.detail for e in report.errors])
            reject_text = (
                "That build FAILED the compatibility check and was NOT delivered:\n"
                f"{report.summary()}\n"
                "Fix the errors (search for compatible replacements) and call "
                "propose_build again."
            )
            return reject_text, None  # continue the loop

        self._last_build = build
        turn = TurnResult(status="proposed", message=rationale, build=build, report=report)
        return "Build delivered to the customer.", turn

    # ── requirement capture (lightweight, deterministic heuristics) ───────────
    def _maybe_capture_requirements(self, text: str) -> None:
        """Cheap keyword/number extraction to seed budget & preferences.

        This is a deterministic pre-pass; the LLM still does the real gathering.
        It just ensures the compatibility engine's budget check has a value to use
        even before the model formalises it.
        """
        import re

        low = text.lower()
        # Budget: "$1500", "1500 dollars", "budget 1200", "under 1000"
        m = re.search(r"\$?\s*(\d{3,5})(?:\s*(?:usd|dollars|budget|bucks))?", low)
        if m and any(k in low for k in ("$", "budget", "under", "spend", "dollar", "cost")):
            try:
                self._requirements.budget_usd = float(m.group(1))
            except ValueError:
                pass
        if "amd" in low or "ryzen" in low:
            self._requirements.cpu_brand = "AMD"
        elif "intel" in low or "core i" in low:
            self._requirements.cpu_brand = "Intel"
        for uc, kws in {
            "gaming": ("gaming", "game", "fps", "esports"),
            "workstation": ("workstation", "render", "editing", "cad", "compile", "3d"),
            "office": ("office", "browsing", "word", "excel", "email"),
            "htpc": ("htpc", "home theatre", "home theater", "media center"),
        }.items():
            if any(k in low for k in kws):
                from pc_agent.models import UseCase
                self._requirements.use_case = UseCase(uc)
                break


def _validate(message: str) -> str | None:
    if not message or not message.strip():
        return "Please describe what you'd like to build."
    if len(message) > _MAX_INPUT_CHARS:
        return (f"Message is too long ({len(message)} chars; limit "
                f"{_MAX_INPUT_CHARS}). Please shorten it.")
    return None


def _tool_result(tool_use_id: str, content: str, *, is_error: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result", "tool_use_id": tool_use_id, "content": content,
    }
    if is_error:
        block["is_error"] = True
    return block
