"""Quantester: institutional-grade event-driven quantitative backtesting engine.

Trader-friendly imports live here so you do not need deep module paths::

    from quantester import (
        run_backtest,
        MovingAverageCrossStrategy,
        make_synthetic_ohlcv,
        load_yahoo,
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
from .simple import BacktestResult, load_crypto, load_yahoo, run_backtest
from .strategy import (
    BuyAndHoldStrategy,
    DonchianBreakoutStrategy,
    MovingAverageCrossStrategy,
    Strategy,
    TranchePullbackStrategy,
)
from .utils.synthetic import make_synthetic_ohlcv

try:
    from .analytics.performance import summarize
except ImportError:  # pragma: no cover
    summarize = None  # type: ignore[assignment]

try:
    from .analytics.tearsheet import generate_tearsheet
except ImportError:  # pragma: no cover
    generate_tearsheet = None  # type: ignore[assignment]

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # One-call API
    "run_backtest",
    "BacktestResult",
    "load_yahoo",
    "load_crypto",
    "make_synthetic_ohlcv",
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
    "generate_tearsheet",
]
