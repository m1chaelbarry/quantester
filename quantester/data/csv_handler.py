"""HistoricCSVDataHandler: per-symbol OHLCV CSVs over a unified master calendar.

Multi-symbol alignment uses an outer-join timestamp union with per-symbol
availability masks: a missing bar marks the asset untradeable at that timestamp
instead of erasing the timestamp (Cross-Ref-2 section 4.3 supersedes Report 1's
incomplete-bar dropping rule, which deletes high-stress/illiquid periods and
induces selection bias). Streaming/firewall semantics live in
StreamingDataHandler.

CSV schema: datetime,open,high,low,close,volume
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .streaming import StreamingDataHandler


class HistoricCSVDataHandler(StreamingDataHandler):
    def __init__(self, csv_map: dict):
        """csv_map: symbol -> path to CSV (or pre-loaded DataFrame indexed by datetime)."""
        frames = {}
        for symbol, source in csv_map.items():
            if isinstance(source, pd.DataFrame):
                frames[symbol] = source.copy()
            else:
                frames[symbol] = pd.read_csv(
                    Path(source), parse_dates=["datetime"], index_col="datetime"
                )
        super().__init__(frames)
