"""Anthropic-backed LLM clients: direct API and Amazon Bedrock.

Both share one request/response normalisation path and differ only in how the
underlying SDK client is constructed. The Anthropic SDK already retries
408/409/429/5xx with exponential backoff and honours the request timeout; we map
its exceptions to :class:`LLMError` so the agent only ever sees one error type.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from pc_agent.config import AppConfig
from pc_agent.llm.base import LLMError, LLMResponse, ToolCall

log = logging.getLogger(__name__)


class _BaseAnthropicClient:
    """Shared normalisation for direct-API and Bedrock Anthropic clients."""

    def __init__(self, sdk_client: Any, config: AppConfig) -> None:
        self._client = sdk_client
        self._config = config
        self.model = config.model

    def _supports_temperature(self) -> bool:
        # Newer adaptive-thinking models reject explicit sampling params. Only
        # send temperature to models where it's safe; when unsure, omit it.
        m = self.model.lower()
        safe_markers = (
            "claude-3", "claude-sonnet-4-0", "claude-opus-4-0", "claude-opus-4-1",
            "sonnet-4-20", "opus-4-20",
        )
        return any(marker in m for marker in safe_markers)

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._config.max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if self._supports_temperature():
            kwargs["temperature"] = self._config.temperature

        try:
            resp = self._client.messages.create(**kwargs)
        except anthropic.APIStatusError as exc:
            raise LLMError(
                f"LLM request failed (HTTP {exc.status_code}): {exc.message}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"LLM connection error: {exc}") from exc
        except anthropic.AnthropicError as exc:
            raise LLMError(f"LLM error: {exc}") from exc

        return self._normalise(resp)

    @staticmethod
    def _normalise(resp: Any) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(resp.usage, "input_tokens", 0),
                "output_tokens": getattr(resp.usage, "output_tokens", 0),
            }

        return LLMResponse(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason or "",
            raw_content=resp.content,
            usage=usage,
        )


class AnthropicDirectClient(_BaseAnthropicClient):
    """Direct Anthropic API (console.anthropic.com)."""

    def __init__(self, config: AppConfig) -> None:
        client = anthropic.Anthropic(
            api_key=config.anthropic_api_key,  # None -> SDK resolves from env/profile
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        super().__init__(client, config)


class AnthropicBedrockClient(_BaseAnthropicClient):
    """Anthropic on Amazon Bedrock. Credentials from the standard AWS chain."""

    def __init__(self, config: AppConfig) -> None:
        client = anthropic.AnthropicBedrock(
            aws_region=config.aws_region,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        super().__init__(client, config)
