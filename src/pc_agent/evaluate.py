"""Lightweight evaluation harness.

Runs each scenario in ``tests/scenarios.json`` through the agent and scores the
output against expectations that are cheap to check without another LLM:

* status (single or set),
* deterministic **compatibility** of the delivered build (re-checked here, so a
  "proposed" build that isn't actually buildable fails),
* presence of essential categories, budget adherence, CPU brand,
* keyword presence (used for the infeasible-request scenario), and
* the feedback loop: the amended build must be compatible and cheaper.

Each scenario prints its trace path so a human can audit the reasoning chain.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pc_agent.agent import PCBuildAgent, TurnResult
from pc_agent.catalog import Catalog
from pc_agent.compatibility import CompatibilityChecker
from pc_agent.config import load_config
from pc_agent.llm import build_client


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class ScenarioReport:
    id: str
    checks: list[Check]
    trace_paths: list[str]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _brand_of(build) -> str | None:
    cpu = build.part("cpu") if build else None
    if not cpu:
        return None
    n = cpu.name.lower()
    if "amd" in n or "ryzen" in n or "athlon" in n or "fx-" in n:
        return "AMD"
    if "intel" in n or "core" in n or "pentium" in n or "celeron" in n or "xeon" in n:
        return "Intel"
    return None


def _score(scenario: dict, first: TurnResult, checker: CompatibilityChecker,
           follow: TurnResult | None) -> list[Check]:
    checks: list[Check] = []

    if "expect_status" in scenario:
        checks.append(Check("status", first.status == scenario["expect_status"],
                            f"got {first.status!r}, want {scenario['expect_status']!r}"))
    if "expect_status_any" in scenario:
        want = scenario["expect_status_any"]
        checks.append(Check("status_any", first.status in want,
                            f"got {first.status!r}, want one of {want}"))

    # Deterministic compatibility of the delivered build.
    if scenario.get("expect_compatible") and first.build is not None:
        rep = checker.check(first.build)
        checks.append(Check("compatible", rep.compatible, rep.summary().splitlines()[0]))
    if scenario.get("expect_if_proposed_compatible") and first.status == "proposed" and first.build:
        rep = checker.check(first.build)
        checks.append(Check("compatible_if_proposed", rep.compatible,
                            rep.summary().splitlines()[0]))

    if "expect_has_categories" in scenario and first.build is not None:
        present = {p.category for p in first.build.parts}
        missing = [c for c in scenario["expect_has_categories"] if c not in present]
        checks.append(Check("has_categories", not missing,
                            "all present" if not missing else f"missing {missing}"))

    if "expect_within_budget" in scenario and first.build is not None:
        cap = float(scenario["expect_within_budget"])
        checks.append(Check("within_budget", first.build.total_price <= cap,
                            f"${first.build.total_price:.2f} <= ${cap:.2f}"))

    if "expect_cpu_brand" in scenario and first.build is not None:
        want = scenario["expect_cpu_brand"]
        got = _brand_of(first.build)
        checks.append(Check("cpu_brand", got == want, f"got {got}, want {want}"))

    if "expect_keywords_any" in scenario:
        hay = " ".join([first.message, first.clarifying_question]).lower()
        kws = [k.lower() for k in scenario["expect_keywords_any"]]
        hit = next((k for k in kws if k in hay), None)
        checks.append(Check("keywords_any", hit is not None,
                            f"matched {hit!r}" if hit else f"none of {kws}"))

    # Feedback loop.
    if follow is not None:
        if "expect_followup_status" in scenario:
            checks.append(Check("followup_status",
                                follow.status == scenario["expect_followup_status"],
                                f"got {follow.status!r}"))
        if scenario.get("expect_followup_compatible") and follow.build is not None:
            rep = checker.check(follow.build)
            checks.append(Check("followup_compatible", rep.compatible,
                                rep.summary().splitlines()[0]))
        if scenario.get("expect_followup_cheaper_than_first") and (
            follow.build is not None and first.build is not None
        ):
            cheaper = follow.build.total_price < first.build.total_price
            checks.append(Check(
                "followup_cheaper",
                cheaper,
                f"${follow.build.total_price:.2f} < ${first.build.total_price:.2f}",
            ))

    if not checks:
        checks.append(Check("ran", first.status != "error", f"status={first.status}"))
    return checks


def run_evaluation(scenarios_path: str | Path) -> list[ScenarioReport]:
    spec = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
    config = load_config()
    catalog = Catalog.load(config.data_dir)
    checker = CompatibilityChecker(catalog)
    llm = build_client(config)

    reports: list[ScenarioReport] = []
    for scenario in spec["scenarios"]:
        # Fresh agent per scenario -> independent conversation + trace.
        agent = PCBuildAgent(catalog, llm, config)
        first = agent.chat(scenario["query"])
        traces = [first.trace_path]
        follow = None
        if scenario.get("followup"):
            follow = agent.chat(scenario["followup"])
            traces.append(follow.trace_path)
        reports.append(ScenarioReport(
            id=scenario["id"],
            checks=_score(scenario, first, checker, follow),
            trace_paths=traces,
        ))
    return reports


def _print_reports(reports: list[ScenarioReport]) -> bool:
    all_pass = True
    for r in reports:
        mark = "PASS" if r.passed else "FAIL"
        all_pass = all_pass and r.passed
        print(f"\n[{mark}] {r.id}")
        for c in r.checks:
            print(f"       {'✓' if c.passed else '✗'} {c.name}: {c.detail}")
        for tp in r.trace_paths:
            print(f"       trace: {tp}")
    passed = sum(r.passed for r in reports)
    print(f"\n{'='*60}\nResult: {passed}/{len(reports)} scenarios passed")
    return all_pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pc-agent-eval", description="Run PC-build agent evaluation scenarios."
    )
    p.add_argument("--scenarios", default="tests/scenarios.json")
    args = p.parse_args(argv)

    try:
        reports = run_evaluation(args.scenarios)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0 if _print_reports(reports) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
