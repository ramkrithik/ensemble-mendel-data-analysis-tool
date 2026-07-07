"""Provider-neutral LLM interface and value types.

The agent loop is written against these types alone. A provider implementation
(Anthropic direct, Bedrock) only turns a normalised request into a provider call
and normalises the response back into an :class:`LLMResponse`.
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
    """Normalised result of one model turn."""

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    raw_content: Any  # provider assistant-turn content, echoed back verbatim
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(Protocol):
    """What the agent needs from any LLM backend."""

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
    """Raised when the LLM call cannot be completed after retries."""
