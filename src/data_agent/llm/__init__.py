"""LLM provider abstraction.

The agent talks to :class:`~data_agent.llm.base.LLMClient`, never to a concrete
SDK. :func:`build_client` picks the implementation from config, so swapping
between the direct Anthropic API and Amazon Bedrock is a one-line env change.
"""

from data_agent.llm.base import LLMClient, LLMResponse, ToolCall
from data_agent.llm.factory import build_client

__all__ = ["LLMClient", "LLMResponse", "ToolCall", "build_client"]
