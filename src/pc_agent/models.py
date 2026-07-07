"""Pydantic models — the typed contracts of the system.

These serve three jobs at once:
  * validate/normalise structured tool inputs from the LLM (e.g. Requirements),
  * give the compatibility engine and agent typed objects to reason over, and
  * define the shape of the final answer so it is machine-checkable in eval.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class UseCase(str, Enum):
    """Coarse usage buckets that drive component priorities."""

    GAMING = "gaming"
    WORKSTATION = "workstation"          # content creation, 3D, compile-heavy
    OFFICE = "office"                    # browsing, docs, light multitask
    HTPC = "htpc"                        # home theatre / small form factor
    GENERAL = "general"


class Requirements(BaseModel):
    """Structured customer requirements, gathered by the agent.

    Everything except a rough intent is optional — the agent fills what it can
    from the conversation and asks about the rest only when it materially changes
    the build.
    """

    use_case: UseCase = UseCase.GENERAL
    budget_usd: float | None = Field(
        default=None, description="Total budget for the parts covered by this tool."
    )
    cpu_brand: str | None = Field(
        default=None, description="'AMD' or 'Intel' if the customer has a preference."
    )
    gpu_preference: str | None = Field(
        default=None, description="Free-text GPU wish, e.g. 'RTX 4070' or 'NVIDIA'."
    )
    min_ram_gb: int | None = None
    min_storage_gb: int | None = None
    form_factor: str | None = Field(
        default=None, description="'ATX', 'Micro ATX', or 'Mini ITX' if specified."
    )
    notes: str | None = Field(
        default=None, description="Any other constraint or preference, verbatim."
    )

    @field_validator("cpu_brand")
    @classmethod
    def _norm_brand(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip().lower()
        if v in ("amd", "ryzen"):
            return "AMD"
        if v in ("intel", "core"):
            return "Intel"
        return v.title()


class BuildPart(BaseModel):
    """One selected component in a build."""

    category: str            # "cpu", "motherboard", ...
    name: str
    price: float | None = None
    # A few salient specs, category-dependent, for the human-facing rationale.
    specs: dict[str, str] = Field(default_factory=dict)


class Build(BaseModel):
    """A proposed PC configuration."""

    parts: list[BuildPart] = Field(default_factory=list)
    total_price: float = 0.0
    rationale: str = ""

    def part(self, category: str) -> BuildPart | None:
        return next((p for p in self.parts if p.category == category), None)


class CompatibilityIssue(BaseModel):
    """A single compatibility or constraint problem found in a build."""

    severity: str  # "error" (build won't work) | "warning" (works, but note it)
    rule: str      # short machine-readable rule id, e.g. "cpu_socket_mismatch"
    detail: str    # human-readable explanation


class CompatibilityReport(BaseModel):
    """Result of checking a build against the compatibility rules."""

    compatible: bool
    issues: list[CompatibilityIssue] = Field(default_factory=list)

    @property
    def errors(self) -> list[CompatibilityIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[CompatibilityIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> str:
        if self.compatible and not self.issues:
            return "COMPATIBLE — no issues found."
        lines = ["COMPATIBLE" if self.compatible else "NOT COMPATIBLE"]
        for i in self.issues:
            lines.append(f"  [{i.severity}] {i.rule}: {i.detail}")
        return "\n".join(lines)
