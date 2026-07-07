"""Command-line entry point.

Two modes:
  * one-shot:   pc-agent --query "gaming PC under $1200"
  * interactive: pc-agent            (chat; type feedback to amend the build)

Wires config -> LLM client -> catalog -> agent. All real logic lives in the
library so the CLI stays thin and is mirrored by the Streamlit UI and evaluator.
"""

from __future__ import annotations

import argparse
import logging
import sys

from pc_agent.agent import PCBuildAgent
from pc_agent.catalog import Catalog
from pc_agent.config import load_config
from pc_agent.llm import build_client


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pc-agent",
        description="Agentic PC-build configurator: requirements -> compatible build.",
    )
    p.add_argument("--query", help="One-shot request. Omit to enter interactive chat.")
    p.add_argument("--data-dir", help="Override the dataset directory (default: data/).")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Show the step-by-step reasoning trace on stderr.")
    return p


def _make_agent(args) -> PCBuildAgent:
    config = load_config()
    if args.data_dir:
        object.__setattr__(config, "data_dir", __import__("pathlib").Path(args.data_dir))
    catalog = Catalog.load(config.data_dir)
    llm = build_client(config)
    print(f"Loaded catalog from {config.data_dir}/ · provider {config.provider}:{config.model}\n",
          file=sys.stderr)
    return PCBuildAgent(catalog, llm, config)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s", stream=sys.stderr,
    )

    try:
        agent = _make_agent(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.query:
        result = agent.chat(args.query)
        print(result.render())
        print(f"\n[status={result.status} steps={result.steps_used} "
              f"trace={result.trace_path}]", file=sys.stderr)
        return 0 if result.status in ("proposed", "needs_clarification") else 1

    # Interactive mode — the feedback loop shines here.
    print("Interactive PC-build agent. Describe your ideal PC; give feedback to amend.")
    print("Type 'exit' or Ctrl-D to quit.\n")
    while True:
        try:
            msg = input("you › ").strip()
        except EOFError:
            print()
            break
        if msg.lower() in ("exit", "quit"):
            break
        if not msg:
            continue
        result = agent.chat(msg)
        print("\n" + result.render() + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
