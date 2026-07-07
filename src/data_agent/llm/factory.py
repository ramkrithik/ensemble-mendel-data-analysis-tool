"""Select and construct the configured LLM client."""

from __future__ import annotations

from data_agent.config import AppConfig
from data_agent.llm.anthropic_client import (
    AnthropicBedrockClient,
    AnthropicDirectClient,
)
from data_agent.llm.base import LLMClient


def build_client(config: AppConfig) -> LLMClient:
    """Return the LLM client for the configured provider."""
    if config.provider == "bedrock":
        return AnthropicBedrockClient(config)
    return AnthropicDirectClient(config)
