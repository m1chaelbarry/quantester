"""Quantester: institutional-grade event-driven quantitative backtesting engine.

Trader-friendly imports live here so you do not need deep module paths::

    from quantester import (
        run_backtest,
        MovingAverageCrossStrategy,
        CostModel,
        summarize,
    )
"""

from .engine import BacktestEngine
from .events import EXIT, LONG, SHORT, SignalEvent
from .execution import CostModel, SimulatedExecutionHandler
from .portfolio import (
    FixedUnitSizer,
    FractionalRiskSizer,
    PercentEquitySizer,
    PortfolioManager,
)
from .simple import BacktestResult, run_backtest
from .strategy import (
    BuyAndHoldStrategy,
    DonchianBreakoutStrategy,
    MovingAverageCrossStrategy,
    Strategy,
    TranchePullbackStrategy,
)

try:
    from .analytics.performance import summarize
except ImportError:  # pragma: no cover - analytics always present in-tree
    summarize = None  # type: ignore[assignment]

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # One-call API
    "run_backtest",
    "BacktestResult",
    # Core wiring
    "BacktestEngine",
    "Strategy",
    "PortfolioManager",
    "SimulatedExecutionHandler",
    "CostModel",
    # Sizing
    "FixedUnitSizer",
    "PercentEquitySizer",
    "FractionalRiskSizer",
    # Built-in strategies
    "BuyAndHoldStrategy",
    "MovingAverageCrossStrategy",
    "DonchianBreakoutStrategy",
    "TranchePullbackStrategy",
    # Signals
    "SignalEvent",
    "LONG",
    "SHORT",
    "EXIT",
    # Results
    "summarize",
]
