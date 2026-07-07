"""Dataset-specific normalisation helpers.

Isolated here so the quirks of *this* dataset live in one place. If the dataset
changes, this is the file to touch.

Key facts about the dataset that these helpers encode:
  * ``cpu.csv`` has NO socket column -> we derive it from ``microarchitecture``.
  * ``memory.csv`` encodes ``modules`` as ``"count,size_gb"`` (e.g. "2,16" = 2x16GB)
    and ``speed`` as ``"ddr_gen,MHz"`` (e.g. "5,6000" = DDR5-6000).
  * ``motherboard.form_factor`` and ``case.type`` use different vocabularies for
    the same physical sizes ("Micro ATX" vs "MicroATX Mini Tower").
"""

from __future__ import annotations

import math
from typing import Any

# ── CPU microarchitecture -> socket ─────────────────────────────────────────
# Covers the modern and common architectures in the dataset. Unlisted (mostly
# very old) architectures resolve to None -> the compatibility engine treats an
# unknown socket as a *warning*, not a hard error, so the agent still functions.
ARCH_TO_SOCKET: dict[str, str] = {
    # AMD
    "Zen 5": "AM5",
    "Zen 4": "AM5",
    "Zen 3": "AM4",
    "Zen 2": "AM4",
    "Zen+": "AM4",
    "Zen": "AM4",
    "Excavator": "AM4",
    "Piledriver": "AM3+",
    "Bulldozer": "AM3+",
    # Intel (modern desktop)
    "Arrow Lake": "LGA1851",
    "Raptor Lake": "LGA1700",
    "Raptor Lake Refresh": "LGA1700",
    "Alder Lake": "LGA1700",
    "Rocket Lake": "LGA1200",
    "Comet Lake": "LGA1200",
    "Coffee Lake Refresh": "LGA1151",
    "Coffee Lake": "LGA1151",
    "Kaby Lake": "LGA1151",
    "Skylake": "LGA1151",
    "Broadwell": "LGA1150",
    "Haswell Refresh": "LGA1150",
    "Haswell": "LGA1150",
    "Ivy Bridge": "LGA1155",
    "Sandy Bridge": "LGA1155",
}


def socket_for_microarchitecture(arch: Any) -> str | None:
    """Return the CPU socket for a microarchitecture, or None if unknown."""
    if not isinstance(arch, str):
        return None
    return ARCH_TO_SOCKET.get(arch.strip())


# ── Motherboard/case form-factor normalisation ──────────────────────────────
# Reduce both vocabularies to a canonical physical size so a board can be matched
# to a case that physically accepts it.
_FF_CANON = {
    "atx": "ATX",
    "micro atx": "Micro ATX",
    "microatx": "Micro ATX",
    "mini itx": "Mini ITX",
    "mini-itx": "Mini ITX",
    "eatx": "EATX",
    "xl atx": "XL ATX",
}

# Which board sizes a case of a given size physically accepts (superset rule):
# larger cases fit smaller boards. Keyed by canonical case size.
CASE_ACCEPTS: dict[str, set[str]] = {
    "EATX": {"EATX", "ATX", "Micro ATX", "Mini ITX"},
    "ATX": {"ATX", "Micro ATX", "Mini ITX"},
    "Micro ATX": {"Micro ATX", "Mini ITX"},
    "Mini ITX": {"Mini ITX"},
}


def canon_board_form_factor(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    return _FF_CANON.get(raw.strip().lower())


def canon_case_size(raw: Any) -> str | None:
    """Map a case ``type`` string (e.g. 'ATX Mid Tower') to a canonical size."""
    if not isinstance(raw, str):
        return None
    low = raw.lower()
    if "eatx" in low or "e-atx" in low:
        return "EATX"
    if "microatx" in low or "micro atx" in low or "matx" in low:
        return "Micro ATX"
    if "mini itx" in low or "mini-itx" in low or "itx" in low:
        return "Mini ITX"
    if "atx" in low:
        return "ATX"
    return None


# ── Memory encoding ──────────────────────────────────────────────────────────
def parse_memory_modules(raw: Any) -> tuple[int, int] | None:
    """Parse ``"count,size_gb"`` (e.g. '2,16') -> (num_modules, size_per_module_gb)."""
    if not isinstance(raw, str) or "," not in raw:
        return None
    try:
        count_s, size_s = raw.split(",", 1)
        return int(count_s), int(size_s)
    except ValueError:
        return None


def total_memory_gb(raw_modules: Any) -> int | None:
    parsed = parse_memory_modules(raw_modules)
    if parsed is None:
        return None
    count, size = parsed
    return count * size


def parse_memory_ddr(raw_speed: Any) -> tuple[int, int] | None:
    """Parse ``"ddr_gen,MHz"`` (e.g. '5,6000') -> (ddr_generation, mhz)."""
    if not isinstance(raw_speed, str) or "," not in raw_speed:
        return None
    try:
        gen_s, mhz_s = raw_speed.split(",", 1)
        return int(gen_s), int(mhz_s)
    except ValueError:
        return None


def coerce_float(value: Any) -> float | None:
    """Best-effort float, returning None for NaN / blanks / non-numerics."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f
