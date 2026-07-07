"""Component catalog: load, normalise, and query the dataset.

The dataset (PCPartPicker-style CSVs) is loaded once into a :class:`Catalog`,
which normalises the quirky encodings (memory ``"count,size"`` strings, DDR
generation from ``speed``) and — crucially — **derives each CPU's socket from its
microarchitecture**, since ``cpu.csv`` has no socket column. That derived socket
is what makes CPU<->motherboard compatibility checkable.
"""

from pc_agent.catalog.catalog import Catalog

__all__ = ["Catalog"]
