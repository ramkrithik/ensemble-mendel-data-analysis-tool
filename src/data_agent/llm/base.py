"""Provider-neutral LLM interface and value types.

The agent loop is written against these types alone. A provider implementation
(Anthropic direct, Bedrock) only has to turn a normalised request into a
provider call and normalise the response back into an :class:`LLMResponse`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    """Normalised result of one model turn.

    ``stop_reason`` is the provider's reason string (e.g. ``"tool_use"``,
    ``"end_turn"``, ``"max_tokens"``, ``"refusal"``). ``text`` is the
    concatenation of any text blocks; ``tool_calls`` holds any tool requests.
    ``raw_content`` is the provider's assistant-turn content, passed back
    verbatim on the next request to preserve conversation state.
    """

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    raw_content: Any
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(Protocol):
    """What the agent needs from any LLM backend.

    Implementations must be resilient: apply timeouts, retry transient failures,
    and raise :class:`~data_agent.llm.base.LLMError` on unrecoverable ones.
    """

    model: str

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Run one model turn and return the normalised response."""
        ...


class LLMError(RuntimeError):
    """Raised when the LLM call cannot be completed after retries/fallback."""
