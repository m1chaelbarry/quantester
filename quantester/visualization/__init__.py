"""Visualization for backtesting and strategy development.

Static charts (``plot_candles``, ``plot_equity``, ``plot_trade_analysis``,
``plot_monthly_returns``, ``plot_rolling_metrics``, ``plot_path_distribution``)
plus a scrollable, interactive matplotlib viewer (``interactive_view``) with
scroll-wheel zoom, drag panning, and a crosshair readout. Indicator helpers
(SMA/EMA/RSI/MACD/Bollinger/ATR) live in ``quantester.indicators`` and are
re-exported from ``visualization.indicators`` for charts.
"""

from . import indicators
from .interactive import InteractiveChartViewer, interactive_view
from .static import (
    plot_candles,
    plot_equity,
    plot_monthly_returns,
    plot_path_distribution,
    plot_rolling_metrics,
    plot_trade_analysis,
    trade_stats,
)

__all__ = [
    "indicators",
    "InteractiveChartViewer",
    "interactive_view",
    "plot_candles",
    "plot_equity",
    "plot_monthly_returns",
    "plot_path_distribution",
    "plot_rolling_metrics",
    "plot_trade_analysis",
    "trade_stats",
]
