"""CSV loading and profiling.

The dataset is loaded once, up front, and a compact *profile* (shape, columns,
dtypes, sample rows, basic stats) is put into the system prompt so the model can
plan without spending a tool call just to look at the data. The full DataFrame
is made available to the code-execution tool as ``df``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Dataset:
    """A loaded CSV plus a human/model-readable profile of it."""

    path: Path
    df: pd.DataFrame

    @classmethod
    def from_csv(cls, path: str | Path) -> "Dataset":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"CSV not found: {p}")
        try:
            df = pd.read_csv(p)
        except Exception as exc:  # pragma: no cover - surfaced to the user
            raise ValueError(f"Could not parse CSV {p}: {exc}") from exc
        if df.empty:
            raise ValueError(f"CSV {p} loaded but contains no rows.")
        return cls(path=p, df=df)

    def profile(self, max_sample_rows: int = 5) -> str:
        """A compact textual profile suitable for embedding in a prompt."""
        buf = io.StringIO()
        df = self.df
        buf.write(f"File: {self.path.name}\n")
        buf.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n")

        buf.write("Columns (name : dtype : non-null : example):\n")
        for col in df.columns:
            non_null = int(df[col].notna().sum())
            example = df[col].dropna().iloc[0] if non_null else "<all null>"
            example_str = str(example)
            if len(example_str) > 40:
                example_str = example_str[:37] + "..."
            buf.write(
                f"  - {col} : {df[col].dtype} : {non_null}/{len(df)} : {example_str}\n"
            )

        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            buf.write("\nNumeric summary:\n")
            desc = numeric.describe().round(3).to_string()
            buf.write(desc + "\n")

        buf.write(f"\nFirst {max_sample_rows} rows:\n")
        buf.write(df.head(max_sample_rows).to_string(max_cols=20) + "\n")
        return buf.getvalue()
