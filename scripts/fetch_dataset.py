"""Fetch the component CSVs the agent needs into ``data/``.

The agent only uses the 8 build-relevant categories. This script copies them from
a local clone of the dataset, or downloads them directly from GitHub.

Usage:
    # from a local clone
    uv run python scripts/fetch_dataset.py --from /path/to/Computer_Components_Dataset

    # or straight from GitHub (raw files)
    uv run python scripts/fetch_dataset.py --github

Dataset: https://github.com/vinayak-ensemble/Computer_Components_Dataset
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

FILES = [
    "cpu.csv", "motherboard.csv", "memory.csv", "video-card.csv",
    "power-supply.csv", "case.csv", "internal-hard-drive.csv", "cpu-cooler.csv",
]
RAW_BASE = (
    "https://raw.githubusercontent.com/vinayak-ensemble/"
    "Computer_Components_Dataset/main/data/csv/"
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch component CSVs into data/.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from", dest="local", help="Path to a local dataset clone.")
    src.add_argument("--github", action="store_true", help="Download from GitHub raw.")
    p.add_argument("--out", default="data", help="Output directory (default: data/).")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        dest = out / fname
        if args.local:
            candidate = Path(args.local) / "data" / "csv" / fname
            if not candidate.is_file():
                candidate = Path(args.local) / fname
            if not candidate.is_file():
                print(f"ERROR: {fname} not found under {args.local}", file=sys.stderr)
                return 2
            shutil.copyfile(candidate, dest)
        else:
            url = RAW_BASE + fname
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dest)  # noqa: S310 - known host
        print(f"  -> {dest}")

    print(f"\nDone. {len(FILES)} files in {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
