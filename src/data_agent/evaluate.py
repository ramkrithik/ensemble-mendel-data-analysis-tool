"""Lightweight evaluation harness.

Runs each scenario in ``tests/scenarios.json`` through the agent and scores the
output against expectations that are cheap to check without another LLM:

* ``expect_status`` / ``expect_status_any`` — did the agent end the right way?
* ``expect_keywords_any`` — does the answer mention at least one expected token?
* ``expect_numeric`` — is a number within tolerance of a known ground truth
  present anywhere in the answer/findings?
* ``expect_findings_min`` — did it produce at least N grounded findings?

This is intentionally qualitative-plus-a-little-quantitative: the assignment
values "compare outputs against expected results", not a perfect grader. Each
scenario's trace path is printed so a human can audit the reasoning chain.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from data_agent.agent import AgentResult, DataAnalysisAgent
from data_agent.config import load_config
from data_agent.llm import build_client
from data_agent.tools.dataset import Dataset


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class ScenarioReport:
    id: str
    query: str
    status: str
    checks: list[Check]
    trace_path: str

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _numbers_in(text: str) -> list[float]:
    """Extract numeric literals (with optional commas/decimals) from text."""
    out: list[float] = []
    for raw in re.findall(r"-?\d[\d,]*\.?\d*", text):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


def _score(scenario: dict, result: AgentResult) -> list[Check]:
    checks: list[Check] = []
    haystack = " ".join(
        [result.answer, " ".join(result.key_findings), result.clarifying_question]
    ).lower()

    # Status
    if "expect_status" in scenario:
        want = scenario["expect_status"]
        checks.append(Check("status", result.status == want,
                            f"got {result.status!r}, want {want!r}"))
    elif "expect_status_any" in scenario:
        want = scenario["expect_status_any"]
        checks.append(Check("status_any", result.status in want,
                            f"got {result.status!r}, want one of {want}"))

    # Keywords
    if "expect_keywords_any" in scenario:
        kws = [k.lower() for k in scenario["expect_keywords_any"]]
        hit = next((k for k in kws if k in haystack), None)
        checks.append(Check("keywords_any", hit is not None,
                            f"matched {hit!r}" if hit else f"none of {kws} present"))

    # Numeric ground truth
    if "expect_numeric" in scenario:
        spec = scenario["expect_numeric"]
        target, tol = float(spec["value"]), float(spec.get("tolerance", 1.0))
        found = _numbers_in(haystack)
        close = next((n for n in found if abs(n - target) <= tol), None)
        checks.append(Check(
            "numeric", close is not None,
            f"found {close} within {tol} of {target}" if close is not None
            else f"no number within {tol} of {target} (saw {found[:8]})",
        ))

    # Minimum findings
    if "expect_findings_min" in scenario:
        need = int(scenario["expect_findings_min"])
        got = len(result.key_findings)
        checks.append(Check("findings_min", got >= need, f"{got} >= {need}"))

    if not checks:
        checks.append(Check("ran", result.status != "error", f"status={result.status}"))
    return checks


def run_evaluation(scenarios_path: str | Path) -> list[ScenarioReport]:
    spec = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
    base = Path(scenarios_path).resolve().parent.parent  # project root
    csv_path = (base / spec["dataset"]).resolve()

    config = load_config()
    dataset = Dataset.from_csv(csv_path)
    llm = build_client(config)

    reports: list[ScenarioReport] = []
    for scenario in spec["scenarios"]:
        # Fresh agent per scenario -> independent conversation + trace.
        agent = DataAnalysisAgent(dataset, llm, config)
        result = agent.run(scenario["query"])
        reports.append(ScenarioReport(
            id=scenario["id"], query=scenario["query"], status=result.status,
            checks=_score(scenario, result), trace_path=result.trace_path,
        ))
    return reports


def _print_reports(reports: list[ScenarioReport]) -> bool:
    all_pass = True
    for r in reports:
        mark = "PASS" if r.passed else "FAIL"
        all_pass = all_pass and r.passed
        print(f"\n[{mark}] {r.id}  (status={r.status})")
        print(f"       q: {r.query}")
        for c in r.checks:
            cm = "✓" if c.passed else "✗"
            print(f"       {cm} {c.name}: {c.detail}")
        print(f"       trace: {r.trace_path}")

    passed = sum(r.passed for r in reports)
    print(f"\n{'='*60}\nResult: {passed}/{len(reports)} scenarios passed")
    return all_pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="data-agent-eval", description="Run agent evaluation scenarios."
    )
    p.add_argument(
        "--scenarios", default="tests/scenarios.json",
        help="Path to the scenarios JSON file.",
    )
    args = p.parse_args(argv)

    try:
        reports = run_evaluation(args.scenarios)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0 if _print_reports(reports) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
