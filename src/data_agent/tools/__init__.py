"""Tool definitions and executors for the agent.

A *tool schema* is the JSON contract the model sees; a *tool executor* is the
Python that runs when the model calls it. :data:`TOOL_SCHEMAS` is the list handed
to the LLM; :data:`TOOL_EXECUTORS` maps tool name -> callable.
"""

from data_agent.tools.registry import TOOL_EXECUTORS, TOOL_SCHEMAS

__all__ = ["TOOL_SCHEMAS", "TOOL_EXECUTORS"]
