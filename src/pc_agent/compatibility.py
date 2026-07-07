"""Deterministic compatibility & sanity engine.

Compatibility is decided by **code, not the LLM** — this is the correctness core.
The agent proposes a build; this module checks it against explicit rules and
returns a typed :class:`CompatibilityReport`. Keeping the rules deterministic
means the same build always yields the same verdict, and the LLM cannot "reason"
its way past a genuine mismatch.

Rules implemented (over the loaded catalog):
  * CPU socket must equal motherboard socket           (error)
  * Motherboard form factor must fit the case size     (error)
  * Memory DDR generation must match the board's        (warning — data is inferred)
  * Total memory must not exceed the board's max_memory (error)
  * PSU wattage must cover an estimated system draw     (error) with headroom (warning)
  * Budget: total price must be within the stated budget (warning)
  * Essential parts present for a functioning build      (error)
"""

from __future__ import annotations

from typing import Any

from pc_agent.catalog import Catalog
from pc_agent.catalog import normalize as nz
from pc_agent.models import (
    Build,
    CompatibilityIssue,
    CompatibilityReport,
    Requirements,
)

# Parts required for a machine that actually boots and is useful.
ESSENTIAL = ("cpu", "motherboard", "memory", "power-supply", "case")
# internal-hard-drive is essential too (needs an OS target); cooler/GPU optional.
ESSENTIAL_WITH_STORAGE = ESSENTIAL + ("internal-hard-drive",)

# Rough TDP-based power model. Real builds vary, but a defensible estimate lets us
# catch obviously-undersized PSUs. GPU draw dominates and the dataset has no GPU
# TDP column, so we estimate from the chipset tier via keyword.
_BASE_SYSTEM_DRAW_W = 90  # motherboard + RAM + storage + fans, ballpark
_HEADROOM = 1.3           # want PSU >= 1.3x estimated draw


def _gpu_draw_estimate(chipset: Any) -> int:
    """Estimate GPU power draw (watts) from its chipset name."""
    if not isinstance(chipset, str):
        return 0
    c = chipset.lower()
    # High end
    if any(k in c for k in ("4090", "3090", "4080", "3080", "7900", "6900")):
        return 350
    if any(k in c for k in ("4070", "3070", "4070", "6800", "7800", "5070", "5080", "5090")):
        return 250
    if any(k in c for k in ("4060", "3060", "2060", "1660", "6600", "7600", "3050")):
        return 170
    return 130  # unknown discrete GPU — conservative default


def estimate_power_draw(build: Build) -> int:
    """Estimate total system power draw in watts for the proposed build."""
    draw = _BASE_SYSTEM_DRAW_W
    cpu = build.part("cpu")
    if cpu is not None:
        tdp = nz.coerce_float(cpu.specs.get("tdp"))
        draw += int(tdp) if tdp else 95
    gpu = build.part("video-card")
    if gpu is not None:
        draw += _gpu_draw_estimate(gpu.specs.get("chipset"))
    return draw


class CompatibilityChecker:
    """Runs the rule set over a Build using the catalog for spec lookups."""

    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog

    def check(
        self, build: Build, requirements: Requirements | None = None
    ) -> CompatibilityReport:
        issues: list[CompatibilityIssue] = []

        issues += self._check_essentials(build)
        issues += self._check_cpu_mobo_socket(build)
        issues += self._check_form_factor(build)
        issues += self._check_memory(build)
        issues += self._check_power(build)
        if requirements is not None:
            issues += self._check_budget(build, requirements)

        compatible = not any(i.severity == "error" for i in issues)
        return CompatibilityReport(compatible=compatible, issues=issues)

    # ── individual rules ─────────────────────────────────────────────────────
    def _check_essentials(self, build: Build) -> list[CompatibilityIssue]:
        issues: list[CompatibilityIssue] = []
        present = {p.category for p in build.parts}
        missing = [c for c in ESSENTIAL_WITH_STORAGE if c not in present]
        if missing:
            issues.append(CompatibilityIssue(
                severity="error", rule="missing_essential_parts",
                detail=f"Build is missing essential part(s): {', '.join(missing)}.",
            ))
        # A part we couldn't resolve in the catalog has no specs -> it can't be
        # priced or checked. Flag it so the model re-selects it by uid.
        for p in build.parts:
            if not p.specs and p.price is None:
                issues.append(CompatibilityIssue(
                    severity="error", rule="part_not_found",
                    detail=(f"Could not resolve {p.category} '{p.name}' in the catalog. "
                            "Select it again using the exact `uid` from search results."),
                ))
        return issues

    def _lookup(self, category: str, part) -> dict[str, Any] | None:
        if part is None:
            return None
        return self._catalog.get(category, part.name)

    def _check_cpu_mobo_socket(self, build: Build) -> list[CompatibilityIssue]:
        cpu = self._lookup("cpu", build.part("cpu"))
        mobo = self._lookup("motherboard", build.part("motherboard"))
        if not cpu or not mobo:
            return []
        cpu_socket = cpu.get("socket")
        mobo_socket = mobo.get("socket")
        if cpu_socket is None:
            return [CompatibilityIssue(
                severity="warning", rule="cpu_socket_unknown",
                detail=(f"Could not derive a socket for CPU "
                        f"'{cpu.get('name')}' ({cpu.get('microarchitecture')}); "
                        "socket compatibility unverified."),
            )]
        if mobo_socket is None:
            return [CompatibilityIssue(
                severity="warning", rule="mobo_socket_unknown",
                detail=f"Motherboard '{mobo.get('name')}' has no socket listed.",
            )]
        if str(cpu_socket) != str(mobo_socket):
            return [CompatibilityIssue(
                severity="error", rule="cpu_socket_mismatch",
                detail=(f"CPU socket {cpu_socket} does not match motherboard "
                        f"socket {mobo_socket}."),
            )]
        return []

    def _check_form_factor(self, build: Build) -> list[CompatibilityIssue]:
        mobo = self._lookup("motherboard", build.part("motherboard"))
        case = self._lookup("case", build.part("case"))
        if not mobo or not case:
            return []
        board_ff = mobo.get("ff_canon") or nz.canon_board_form_factor(mobo.get("form_factor"))
        case_size = case.get("size_canon") or nz.canon_case_size(case.get("type"))
        if board_ff is None or case_size is None:
            return [CompatibilityIssue(
                severity="warning", rule="form_factor_unknown",
                detail="Could not determine board/case form factor; fit unverified.",
            )]
        accepts = nz.CASE_ACCEPTS.get(case_size, set())
        if board_ff not in accepts:
            return [CompatibilityIssue(
                severity="error", rule="form_factor_mismatch",
                detail=(f"{board_ff} motherboard does not fit a {case_size} case "
                        f"('{case.get('type')}')."),
            )]
        return []

    def _check_memory(self, build: Build) -> list[CompatibilityIssue]:
        mem = self._lookup("memory", build.part("memory"))
        mobo = self._lookup("motherboard", build.part("motherboard"))
        out: list[CompatibilityIssue] = []
        if not mem or not mobo:
            return out
        mem_gen = mem.get("ddr_gen")
        board_gen = mobo.get("ddr_gen")
        if mem_gen and board_gen and int(mem_gen) != int(board_gen):
            out.append(CompatibilityIssue(
                severity="warning", rule="memory_ddr_mismatch",
                detail=(f"Memory is DDR{mem_gen} but the board's socket "
                        f"({mobo.get('socket')}) typically uses DDR{board_gen}. "
                        "Verify the specific board's supported memory."),
            ))
        total = nz.coerce_float(mem.get("total_gb"))
        max_mem = nz.coerce_float(mobo.get("max_memory"))
        if total and max_mem and total > max_mem:
            out.append(CompatibilityIssue(
                severity="error", rule="memory_exceeds_max",
                detail=(f"Memory total {int(total)}GB exceeds the motherboard's "
                        f"max of {int(max_mem)}GB."),
            ))
        return out

    def _check_power(self, build: Build) -> list[CompatibilityIssue]:
        psu = self._lookup("power-supply", build.part("power-supply"))
        if not psu:
            return []
        wattage = nz.coerce_float(psu.get("wattage"))
        if not wattage:
            return [CompatibilityIssue(
                severity="warning", rule="psu_wattage_unknown",
                detail="PSU wattage not listed; cannot verify it powers the build.",
            )]
        draw = estimate_power_draw(build)
        if wattage < draw:
            return [CompatibilityIssue(
                severity="error", rule="psu_undersized",
                detail=(f"PSU {int(wattage)}W is below the estimated system draw "
                        f"of ~{draw}W."),
            )]
        if wattage < draw * _HEADROOM:
            return [CompatibilityIssue(
                severity="warning", rule="psu_low_headroom",
                detail=(f"PSU {int(wattage)}W covers the ~{draw}W estimate but "
                        f"leaves little headroom (recommend >= {int(draw * _HEADROOM)}W)."),
            )]
        return []

    def _check_budget(
        self, build: Build, req: Requirements
    ) -> list[CompatibilityIssue]:
        if req.budget_usd is None:
            return []
        if build.total_price > req.budget_usd:
            over = build.total_price - req.budget_usd
            return [CompatibilityIssue(
                severity="warning", rule="over_budget",
                detail=(f"Build total ${build.total_price:.2f} exceeds the stated "
                        f"budget of ${req.budget_usd:.2f} by ${over:.2f}."),
            )]
        return []
