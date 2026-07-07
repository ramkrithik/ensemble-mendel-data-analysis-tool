"""Tool schemas (the model's contract) and their runtime executors.

The agent has four tools:
  * ``search_components``     — query the catalog for parts under constraints.
  * ``check_compatibility``   — run the deterministic engine on a candidate build.
  * ``propose_build``         — deliver the final, checked configuration.
  * ``ask_clarifying_question`` — pause for the customer on genuine ambiguity.

Schemas are static; executors are bound to a live :class:`Catalog` /
:class:`CompatibilityChecker` per run by :class:`ToolKit`.
"""

from pc_agent.tools.registry import TOOL_SCHEMAS, ToolKit

__all__ = ["TOOL_SCHEMAS", "ToolKit"]
