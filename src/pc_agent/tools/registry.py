"""Tool schemas and the :class:`ToolKit` that executes them against the catalog."""

from __future__ import annotations

import json
from typing import Any

from pc_agent.catalog import Catalog
from pc_agent.catalog.catalog import CATEGORIES
from pc_agent.compatibility import CompatibilityChecker
from pc_agent.models import Build, BuildPart, Requirements

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_components",
        "description": (
            "Search the components dataset for parts in one category that match "
            "constraints. Returns up to `limit` matches (priced items only), "
            "sorted by price ascending by default. Use this to find candidate "
            "parts before proposing a build — never invent part names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": list(CATEGORIES),
                    "description": "Which component category to search.",
                },
                "max_price": {"type": "number", "description": "Max unit price (USD)."},
                "min_price": {"type": "number", "description": "Min unit price (USD)."},
                "keyword": {
                    "type": "string",
                    "description": "Case-insensitive substring on the part name, e.g. 'Ryzen 7' or 'RTX 4070'.",
                },
                "socket": {
                    "type": "string",
                    "description": "Filter CPUs/motherboards by socket, e.g. 'AM5', 'LGA1700'.",
                },
                "ddr_gen": {
                    "type": "integer",
                    "description": "Filter memory/motherboards by DDR generation (4 or 5).",
                },
                "form_factors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter motherboards by canonical form factor, e.g. ['ATX','Micro ATX'].",
                },
                "min_wattage": {"type": "integer", "description": "Min PSU wattage."},
                "min_total_gb": {"type": "integer", "description": "Min total memory (GB) for a memory kit."},
                "min_capacity_gb": {"type": "integer", "description": "Min storage capacity (GB)."},
                "sort_by": {"type": "string", "description": "Column to sort by (default 'price')."},
                "ascending": {"type": "boolean", "description": "Sort ascending (default true)."},
                "limit": {"type": "integer", "description": "Max results (default 8)."},
            },
            "required": ["category"],
        },
    },
    {
        "name": "check_compatibility",
        "description": (
            "Run the deterministic compatibility engine on a candidate build. "
            "Pass the parts you selected. Returns a report listing any "
            "socket/form-factor/memory/power/budget errors or warnings. ALWAYS "
            "call this and resolve every error before proposing the build."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "parts": {
                    "type": "object",
                    "description": (
                        "Map of category -> the part's `uid` from search results "
                        "(e.g. {'cpu': 'cpu#12', 'motherboard': 'motherboard#340'}). "
                        "Use the uid, NOT the name — names are not unique."
                    ),
                    "additionalProperties": {"type": "string"},
                }
            },
            "required": ["parts"],
        },
    },
    {
        "name": "ask_clarifying_question",
        "description": (
            "Ask the customer ONE focused question when a requirement is missing "
            "or ambiguous in a way that would change the build (e.g. no budget "
            "given, or 'good for games' without a resolution/FPS target). Prefer "
            "reasonable assumptions for minor gaps; reserve this for real forks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "why": {"type": "string", "description": "Why the answer changes the build."},
            },
            "required": ["question"],
        },
    },
    {
        "name": "propose_build",
        "description": (
            "Deliver the final PC configuration. Call this only after "
            "check_compatibility reports no errors. Include a short rationale "
            "tying the choices back to the customer's requirements."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "parts": {
                    "type": "object",
                    "description": "Map of category -> the part's `uid` from search results.",
                    "additionalProperties": {"type": "string"},
                },
                "rationale": {
                    "type": "string",
                    "description": "1–4 sentences explaining how the build meets the needs.",
                },
            },
            "required": ["parts", "rationale"],
        },
    },
]

TERMINAL_TOOLS = frozenset({"propose_build"})
INTERRUPTING_TOOLS = frozenset({"ask_clarifying_question"})

# Salient specs to surface per category in a BuildPart, for rationale/UI.
_SPEC_KEYS = {
    "cpu": ["socket", "core_count", "boost_clock", "tdp", "microarchitecture"],
    "motherboard": ["socket", "form_factor", "max_memory", "memory_slots", "ddr_gen"],
    "memory": ["total_gb", "ddr_gen", "ddr_mhz", "cas_latency"],
    "video-card": ["chipset", "memory", "length"],
    "power-supply": ["wattage", "efficiency", "modular"],
    "case": ["type", "size_canon"],
    "internal-hard-drive": ["capacity", "type", "form_factor", "interface"],
    "cpu-cooler": ["rpm", "noise_level", "size"],
}


class ToolKit:
    """Binds tool executors to a catalog + compatibility checker for one session."""

    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog
        self._checker = CompatibilityChecker(catalog)

    # ── executors ────────────────────────────────────────────────────────────
    def search_components(self, args: dict[str, Any]) -> str:
        category = args.get("category")
        if category not in CATEGORIES:
            return f"ERROR: unknown category {category!r}. Valid: {list(CATEGORIES)}"
        try:
            results = self._catalog.search(
                category,
                max_price=args.get("max_price"),
                min_price=args.get("min_price"),
                keyword=args.get("keyword"),
                socket=args.get("socket"),
                ddr_gen=args.get("ddr_gen"),
                form_factors=args.get("form_factors"),
                min_wattage=args.get("min_wattage"),
                min_total_gb=args.get("min_total_gb"),
                min_capacity_gb=args.get("min_capacity_gb"),
                sort_by=args.get("sort_by", "price"),
                ascending=args.get("ascending", True),
                limit=int(args.get("limit", 8)),
            )
        except KeyError as exc:
            return f"ERROR: {exc}"
        if not results:
            return "No matching components found. Loosen the constraints and retry."
        trimmed = [_trim_row(category, r) for r in results]
        return json.dumps({"category": category, "count": len(trimmed), "results": trimmed}, default=str)

    def check_compatibility(self, args: dict[str, Any], requirements: Requirements | None) -> str:
        build = self.build_from_parts(args.get("parts", {}))
        report = self._checker.check(build, requirements)
        return report.summary()

    def build_from_parts(self, parts: dict[str, str]) -> Build:
        """Materialise a Build from a {category: uid-or-name} map.

        Prefers the ``uid`` returned by search; falls back to a unique name.
        An unresolved/ambiguous reference is kept as an unpriced part so the
        compatibility engine flags the build rather than silently mispricing it.
        """
        build_parts: list[BuildPart] = []
        total = 0.0
        for category, ref in parts.items():
            if category not in CATEGORIES:
                continue
            row = self._catalog.get(category, str(ref))
            if row is None:
                # Keep the (unresolved) part so compatibility can flag it later.
                build_parts.append(BuildPart(category=category, name=str(ref)))
                continue
            price = row.get("price")
            if isinstance(price, (int, float)):
                total += float(price)
            specs = {
                k: str(row.get(k)) for k in _SPEC_KEYS.get(category, [])
                if row.get(k) is not None
            }
            build_parts.append(BuildPart(
                category=category, name=str(row.get("name", ref)),
                price=float(price) if isinstance(price, (int, float)) else None,
                specs=specs,
            ))
        return Build(parts=build_parts, total_price=round(total, 2))


def _trim_row(category: str, row: dict[str, Any]) -> dict[str, Any]:
    """Return uid + name + price + salient specs (keeps context small).

    ``uid`` is first so the model reliably passes it back to check_compatibility
    and propose_build — names in this dataset are not unique.
    """
    out: dict[str, Any] = {
        "uid": row.get("uid"),
        "name": row.get("name"),
        "price": row.get("price"),
    }
    for k in _SPEC_KEYS.get(category, []):
        if row.get(k) is not None:
            out[k] = row.get(k)
    return out
