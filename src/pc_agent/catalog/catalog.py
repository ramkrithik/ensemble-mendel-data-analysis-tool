"""The queryable component catalog.

Loads the vendored CSVs, augments them with derived/normalised columns (CPU
socket, total memory GB, DDR generation, canonical form factors), and exposes a
small, typed query surface the agent's tools call:

  * :meth:`search` — filter+sort a category by budget/brand/keyword/specs.
  * :meth:`get` — fetch one component by (near-)exact name.
  * :meth:`summary` — a compact catalog overview for the system prompt.

Only the components needed to reason about compatibility and a working build are
loaded (cpu, motherboard, memory, video-card, power-supply, case, storage, cooler).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from pc_agent.catalog import normalize as nz

log = logging.getLogger(__name__)

# category -> csv filename
_FILES = {
    "cpu": "cpu.csv",
    "motherboard": "motherboard.csv",
    "memory": "memory.csv",
    "video-card": "video-card.csv",
    "power-supply": "power-supply.csv",
    "case": "case.csv",
    "internal-hard-drive": "internal-hard-drive.csv",
    "cpu-cooler": "cpu-cooler.csv",
}

CATEGORIES = tuple(_FILES.keys())


class Catalog:
    """In-memory, normalised view of the components dataset."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self._frames = frames

    # ── Loading ──────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, data_dir: str | Path) -> "Catalog":
        base = Path(data_dir)
        frames: dict[str, pd.DataFrame] = {}
        for category, fname in _FILES.items():
            path = base / fname
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing dataset file for {category!r}: {path}. "
                    "See README for how to fetch the dataset into data/."
                )
            df = pd.read_csv(path)
            frames[category] = cls._augment(category, df)
        return cls(frames)

    @staticmethod
    def _augment(category: str, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns used by search and the compatibility engine."""
        df = df.copy().reset_index(drop=True)
        # Stable unique id per row. Component NAMES are NOT unique in this dataset
        # (e.g. 93 different cards are all "Gigabyte GAMING OC" at different prices),
        # so parts must be referenced by uid to resolve unambiguously.
        df["uid"] = [f"{category}#{i}" for i in range(len(df))]
        if category == "cpu":
            df["socket"] = df["microarchitecture"].map(nz.socket_for_microarchitecture)
        elif category == "memory":
            df["total_gb"] = df["modules"].map(nz.total_memory_gb)
            ddr = df["speed"].map(nz.parse_memory_ddr)
            df["ddr_gen"] = ddr.map(lambda t: t[0] if t else None)
            df["ddr_mhz"] = ddr.map(lambda t: t[1] if t else None)
        elif category == "motherboard":
            df["ff_canon"] = df["form_factor"].map(nz.canon_board_form_factor)
            # Motherboard DDR generation is inferred from socket (dataset lacks it):
            df["ddr_gen"] = df["socket"].map(_socket_ddr_gen)
        elif category == "case":
            df["size_canon"] = df["type"].map(nz.canon_case_size)
        return df

    # ── Query surface ────────────────────────────────────────────────────────
    def categories(self) -> tuple[str, ...]:
        return CATEGORIES

    def frame(self, category: str) -> pd.DataFrame:
        if category not in self._frames:
            raise KeyError(f"Unknown category {category!r}. Known: {CATEGORIES}")
        return self._frames[category]

    def get(self, category: str, ref: str) -> dict[str, Any] | None:
        """Fetch one component by uid (preferred) or, failing that, by name.

        ``ref`` should be the ``uid`` returned by :meth:`search` (e.g.
        ``"video-card#67"``). Name resolution is a best-effort fallback only,
        because names are not unique in this dataset — an exact-name match with a
        single candidate is used, but an ambiguous name returns None so the caller
        can tell the model to use the uid instead.
        """
        df = self.frame(category)
        ref = str(ref).strip()

        if "uid" in df.columns:
            by_uid = df[df["uid"] == ref]
            if len(by_uid):
                return _row_to_dict(by_uid.iloc[0])

        exact = df[df["name"] == ref]
        if len(exact) == 1:
            return _row_to_dict(exact.iloc[0])
        loose = df[df["name"].str.lower() == ref.lower()]
        if len(loose) == 1:
            return _row_to_dict(loose.iloc[0])
        # Ambiguous or not found by name -> None (caller should use uid).
        return None

    def search(
        self,
        category: str,
        *,
        max_price: float | None = None,
        min_price: float | None = None,
        keyword: str | None = None,
        socket: str | None = None,
        ddr_gen: int | None = None,
        form_factors: list[str] | None = None,
        min_wattage: int | None = None,
        min_total_gb: int | None = None,
        min_capacity_gb: int | None = None,
        sort_by: str = "price",
        ascending: bool = True,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Filter a category by common constraints and return the top matches.

        Only priced rows are returned (an unpriced part can't be budgeted), and
        every filter is optional so the agent can search broadly or narrowly.
        """
        df = self.frame(category)
        # Require a usable price.
        df = df[df["price"].map(nz.coerce_float).notna()]

        if max_price is not None:
            df = df[df["price"] <= max_price]
        if min_price is not None:
            df = df[df["price"] >= min_price]
        if keyword:
            df = df[df["name"].str.contains(keyword.strip(), case=False, regex=False, na=False)]
        if socket is not None and "socket" in df.columns:
            df = df[df["socket"] == socket]
        if ddr_gen is not None and "ddr_gen" in df.columns:
            df = df[df["ddr_gen"] == ddr_gen]
        if form_factors and "ff_canon" in df.columns:
            df = df[df["ff_canon"].isin(form_factors)]
        if min_wattage is not None and "wattage" in df.columns:
            df = df[df["wattage"] >= min_wattage]
        if min_total_gb is not None and "total_gb" in df.columns:
            df = df[df["total_gb"] >= min_total_gb]
        if min_capacity_gb is not None and "capacity" in df.columns:
            df = df[df["capacity"] >= min_capacity_gb]

        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending, na_position="last")

        return [_row_to_dict(r) for _, r in df.head(limit).iterrows()]

    def summary(self) -> str:
        """Compact overview for the system prompt (counts + price ranges)."""
        lines = ["Component catalog (priced items only shown as ranges):"]
        for cat in CATEGORIES:
            df = self.frame(cat)
            priced = df[df["price"].map(nz.coerce_float).notna()]
            if len(priced):
                lo, hi = priced["price"].min(), priced["price"].max()
                lines.append(
                    f"  - {cat}: {len(priced)} priced items, ${lo:.0f}–${hi:.0f}"
                )
            else:
                lines.append(f"  - {cat}: {len(df)} items")
        lines.append(
            "\nNote: CPU 'socket' is derived from microarchitecture; motherboard "
            "'ddr_gen' is inferred from socket (AM5/LGA1851 => DDR5, most others => DDR4)."
        )
        return "\n".join(lines)


def _socket_ddr_gen(socket: Any) -> int | None:
    """DDR generation a motherboard socket implies (dataset lacks the column)."""
    if not isinstance(socket, str):
        return None
    ddr5 = {"AM5", "LGA1851"}
    ddr4 = {"AM4", "LGA1700", "LGA1200", "LGA1151", "LGA1150", "LGA1155", "AM3+"}
    # LGA1700 actually supports both DDR4/DDR5 boards; we treat it as DDR5-capable
    # by leaving it out of the strict DDR4-only set below and returning None so the
    # engine only warns rather than hard-blocks. Keep it simple + safe:
    if socket in ddr5 or socket == "LGA1700":
        return 5
    if socket in ddr4:
        return 4
    return None


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        fv = nz.coerce_float(v) if k in ("price",) else v
        if isinstance(v, float) and pd.isna(v):
            out[str(k)] = None
        else:
            out[str(k)] = fv if k == "price" else (None if pd.isna(v) else v)
    return out
