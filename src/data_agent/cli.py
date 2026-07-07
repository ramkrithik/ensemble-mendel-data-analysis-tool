"""Command-line entry point: ``data-agent --csv ... --query ...``.

Wires config -> LLM client -> dataset -> agent, runs one question, and prints the
result plus where the trace was written. Kept thin: all real logic lives in the
library modules so it is equally usable from the Streamlit UI and the evaluator.
"""

from __future__ import annotations

import argparse
import logging
import sys

from data_agent.agent import DataAnalysisAgent
from data_agent.config import load_config
from data_agent.llm import build_client
from data_agent.tools.dataset import Dataset


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="data-agent",
        description="Agentic data-analysis assistant: ask questions about a CSV.",
    )
    p.add_argument("--csv", required=True, help="Path to the CSV file to analyse.")
    p.add_argument("--query", required=True, help="Natural-language question.")
    p.add_argument(
        "--domain",
        default="data-analysis",
        help="Domain label (for parity with the assignment CLI; informational).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show the step-by-step reasoning trace on stderr.",
    )
    p.add_argument(
        "--no-few-shot", action="store_true",
        help="Disable the few-shot example (useful for ablation).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
        stream=sys.stderr,
    )

    try:
        config = load_config()
        dataset = Dataset.from_csv(args.csv)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    llm = build_client(config)
    agent = DataAnalysisAgent(dataset, llm, config)

    print(f"Analysing {dataset.path.name} with {config.provider}:{config.model}\n")
    result = agent.run(args.query, include_few_shot=not args.no_few_shot)

    print(result.render())
    print(f"\n[status={result.status} steps={result.steps_used} "
          f"trace={result.trace_path}]", file=sys.stderr)

    return 0 if result.status in ("answered", "needs_clarification") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
