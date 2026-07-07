"""The reason -> plan -> act -> observe -> respond loop.

:class:`DataAnalysisAgent` owns one dataset and one LLM client and turns a natural
-language question into a grounded answer by iterating:

    model turn  ->  (tool call?)  ->  execute  ->  feed observation back  ->  ...

The loop terminates when the model calls ``final_answer`` (success), asks a
clarifying question (pauses for the user), exhausts ``agent_max_steps`` (guard-
rail), or the LLM call fails unrecoverably (surfaced as a fallback answer).

Everything the loop does is written to a :class:`~data_agent.tracing.Tracer`, so
the full reasoning chain is inspectable after the fact.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from data_agent.config import AppConfig
from data_agent.llm.base import LLMClient, LLMError, LLMResponse
from data_agent.prompts import FEW_SHOT_MESSAGES, build_system_prompt
from data_agent.tools.code_executor import CodeExecutor
from data_agent.tools.dataset import Dataset
from data_agent.tools.registry import (
    INTERRUPTING_TOOLS,
    TERMINAL_TOOLS,
    TOOL_SCHEMAS,
)
from data_agent.tracing import Tracer

# Rough guard-rail: refuse to start if the profile + question obviously blow the
# budget. Real token counting would use messages.count_tokens; this cheap check
# catches pathological inputs (a pasted 10MB question) without a network call.
_MAX_QUESTION_CHARS = 8000


@dataclass
class AgentResult:
    """The outcome of a run, plus everything needed to inspect it."""

    status: str  # "answered" | "needs_clarification" | "max_steps" | "error"
    answer: str
    key_findings: list[str] = field(default_factory=list)
    methodology: str = ""
    clarifying_question: str = ""
    steps_used: int = 0
    trace_path: str = ""
    run_id: str = ""

    def render(self) -> str:
        """Human-readable rendering for the CLI."""
        if self.status == "needs_clarification":
            return f"❓ Clarifying question:\n  {self.clarifying_question}"
        lines = [self.answer.strip()]
        if self.key_findings:
            lines.append("\nKey findings:")
            lines.extend(f"  • {f}" for f in self.key_findings)
        if self.methodology:
            lines.append(f"\nMethodology: {self.methodology}")
        return "\n".join(lines)


class DataAnalysisAgent:
    """Drives the agent loop over a single dataset."""

    def __init__(
        self,
        dataset: Dataset,
        llm: LLMClient,
        config: AppConfig,
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> None:
        self._dataset = dataset
        self._llm = llm
        self._config = config
        self._executor = CodeExecutor(dataset.df)
        self._on_token = on_token  # optional streaming/preview hook for UIs
        self._system_prompt = build_system_prompt(dataset.profile())

    def run(self, question: str, *, include_few_shot: bool = True) -> AgentResult:
        run_id = uuid.uuid4().hex[:12]
        tracer = Tracer(run_id, self._config.trace_dir)
        tracer.record(
            0, "run_start", question=question, config=self._config.redacted(),
            dataset=str(self._dataset.path),
        )

        # ── Guard-rail: input validation ────────────────────────────────────
        problem = _validate_question(question)
        if problem is not None:
            tracer.record(0, "error", reason=problem)
            return AgentResult(
                status="error", answer=problem, run_id=run_id,
                trace_path=str(tracer.path),
            )

        messages: list[dict[str, Any]] = list(FEW_SHOT_MESSAGES) if include_few_shot else []
        messages.append({"role": "user", "content": question})

        for step in range(1, self._config.agent_max_steps + 1):
            try:
                resp = self._llm.complete(
                    system=self._system_prompt,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                )
            except LLMError as exc:
                # Graceful degradation: don't crash the caller on an API failure.
                tracer.record(step, "error", reason=str(exc))
                return AgentResult(
                    status="error",
                    answer=(
                        "The language model was unavailable after retries, so I "
                        f"couldn't complete the analysis. Details: {exc}"
                    ),
                    steps_used=step, run_id=run_id, trace_path=str(tracer.path),
                )

            if resp.text:
                tracer.record(step, "model_thought", text=resp.text)
                if self._on_token:
                    self._on_token(resp.text)

            # Handle a safety refusal explicitly rather than reading empty content.
            if resp.stop_reason == "refusal":
                tracer.record(step, "error", reason="model_refusal")
                return AgentResult(
                    status="error",
                    answer="The model declined to answer this request.",
                    steps_used=step, run_id=run_id, trace_path=str(tracer.path),
                )

            if not resp.wants_tool:
                # Model answered in prose without calling final_answer. Nudge it
                # once toward the structured tool; if it already has content, use it.
                tracer.record(step, "model_thought", note="no_tool_call", text=resp.text)
                messages.append({"role": "assistant", "content": resp.raw_content})
                messages.append({
                    "role": "user",
                    "content": (
                        "Please deliver your conclusion by calling the final_answer "
                        "tool so it is structured."
                    ),
                })
                continue

            # Persist the assistant turn (with its tool_use blocks) verbatim.
            messages.append({"role": "assistant", "content": resp.raw_content})

            terminal = self._handle_tool_calls(resp, messages, tracer, step)
            if terminal is not None:
                terminal.steps_used = step
                terminal.run_id = run_id
                terminal.trace_path = str(tracer.path)
                tracer.record(step, "run_end", status=terminal.status)
                return terminal

        # ── Guard-rail: step budget exhausted ───────────────────────────────
        tracer.record(
            self._config.agent_max_steps, "run_end", status="max_steps"
        )
        return AgentResult(
            status="max_steps",
            answer=(
                "I reached the maximum number of analysis steps without a final "
                "answer. Try narrowing the question or raising AGENT_MAX_STEPS."
            ),
            steps_used=self._config.agent_max_steps,
            run_id=run_id, trace_path=str(tracer.path),
        )

    # ------------------------------------------------------------------ #

    def _handle_tool_calls(
        self,
        resp: LLMResponse,
        messages: list[dict[str, Any]],
        tracer: Tracer,
        step: int,
    ) -> AgentResult | None:
        """Execute the model's tool calls; return an AgentResult iff terminal.

        The model may emit several tool calls in one turn; per the Messages API
        contract we must return one tool_result per tool_use, all in a single
        user message.
        """
        tool_results: list[dict[str, Any]] = []

        for call in resp.tool_calls:
            tracer.record(step, "tool_call", tool=call.name, arguments=call.arguments)

            if call.name in TERMINAL_TOOLS:
                return self._finalise(call.arguments, tracer, step)

            if call.name in INTERRUPTING_TOOLS:
                question = str(call.arguments.get("question", "")).strip()
                tracer.record(
                    step, "clarifying_question", question=question,
                    why=call.arguments.get("why", ""),
                )
                return AgentResult(
                    status="needs_clarification",
                    answer="",
                    clarifying_question=question,
                )

            if call.name == "run_python":
                result = self._executor.run(str(call.arguments.get("code", "")))
                tracer.record(
                    step, "observation", tool="run_python", ok=result.ok,
                    intent=call.arguments.get("intent", ""),
                    stdout=result.stdout, error=result.error,
                )
                tool_results.append(_tool_result_block(call.id, result.as_tool_content(), is_error=not result.ok))
            else:
                # Unknown tool name — tell the model so it can recover.
                msg = f"Unknown tool {call.name!r}."
                tracer.record(step, "error", reason=msg)
                tool_results.append(_tool_result_block(call.id, msg, is_error=True))

        # Feed all observations back as one user message and continue the loop.
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        return None

    def _finalise(
        self, args: dict[str, Any], tracer: Tracer, step: int
    ) -> AgentResult:
        summary = str(args.get("summary", "")).strip()
        findings = [str(f) for f in args.get("key_findings", []) or []]
        methodology = str(args.get("methodology", "")).strip()
        tracer.record(
            step, "final_answer", summary=summary, key_findings=findings,
            methodology=methodology,
        )
        return AgentResult(
            status="answered",
            answer=summary,
            key_findings=findings,
            methodology=methodology,
        )


def _validate_question(question: str) -> str | None:
    """Return an error message if the question is unusable, else None."""
    if not question or not question.strip():
        return "No question provided. Ask something about the dataset."
    if len(question) > _MAX_QUESTION_CHARS:
        return (
            f"Question is too long ({len(question)} chars; limit "
            f"{_MAX_QUESTION_CHARS}). Please shorten it."
        )
    return None


def _tool_result_block(tool_use_id: str, content: str, *, is_error: bool) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if is_error:
        block["is_error"] = True
    return block
