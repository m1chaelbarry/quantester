"""Macro / FX overlays — not StreamingDataHandler bar feeds.

Load World Bank, NBP, and GUS series as tidy DataFrames, then optionally align
them onto a trading calendar with :func:`as_daily_reindex`.

Requires ``pip install "quantester[data]"`` (for ``requests``). GUS accepts an
optional ``X-ClientId`` via ``api_key=`` / ``QUANTESTER_GUS_API_KEY``.
"""

from .align import as_daily_reindex
from .gus import load_gus_variable
from .nbp import load_nbp_fx
from .worldbank import load_world_bank

__all__ = [
    "as_daily_reindex",
    "load_world_bank",
    "load_nbp_fx",
    "load_gus_variable",
]
