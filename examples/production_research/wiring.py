"""How to wire a Quantester backtest (the five modules).

Read this file before ``run.py``. It answers: "what pieces do I need, and how
do they connect?"

Quantester is event-driven. Five modules talk only through a shared queue::

    DataHandler  →  Strategy  →  PortfolioManager  →  ExecutionSimulator
                           ↑________ BacktestEngine queue ________↑

Lifecycle (one bar at a time)::

    MarketEvent → SignalEvent → OrderEvent → FillEvent

You never call Strategy → Portfolio directly. The engine drains the queue.

Two ways to run the *same* strategy
-----------------------------------
1. **Event engine** (``event_backtest``) — full fidelity: delay, costs, risk
   overlays, tearsheets. Slow; use for the champion and for leak checks.
2. **Fast-track** (``fast_sharpe`` / ``fast_equity``) — vectorized twin via
   ``momentum_positions`` + ``fast_backtest``. Fast; use for grids, MCPT,
   walk-forward. Only allowed after parity with the event engine passes.

Sizing note
-----------
Research-matrix share count is set so |target|=1 ≈ 95% of equity
(``FixedUnitSizer`` / ``fast_backtest``). A hard-coded 100 shares on a $100
name leaves ~90% cash and makes OOS Sharpe look artificially dead.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import kurtosis, skew

from quantester.analytics.performance import annualized_sharpe
from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.events import EXIT, LONG, SHORT, SignalEvent
from quantester.execution.costs import CostModel
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.montecarlo.fast_track import fast_backtest
from quantester.portfolio.portfolio import FixedUnitSizer, PortfolioManager
from quantester.portfolio.risk import DailyDrawdownBreaker, MarginMonitor

from strategy import TrendMomentumStrategy, momentum_positions

# ---------------------------------------------------------------------------
# Shared constants used by every stage in run.py
# ---------------------------------------------------------------------------
SYMBOL = "EDGE"
INITIAL_CAPITAL = 100_000.0
SEED = 8

# Realistic retail-ish friction. ``fill_price`` already embeds spread /
# slippage / impact; cash is charged qty * fill_price + commission once.
# Never double-deduct ``slippage_cost`` — it is analytics-only.
BASE_COSTS = CostModel(
    fixed_commission=0.0,
    per_share_commission=0.005,
    spread_pct=0.0004,
    slippage_vol_coef=0.05,
    impact_coef=0.05,
)


def units_for(df: pd.DataFrame) -> float:
    """Share count so a full-size signal ≈ 95% of starting equity."""
    px = float(df["close"].iloc[0])
    return max(INITIAL_CAPITAL * 0.95 / px, 1.0)


# ===========================================================================
# EVENT ENGINE — the canonical five-module wiring
# ===========================================================================
def event_backtest(
    df: pd.DataFrame,
    lookback: int,
    *,
    sizer=None,
    cost_model=None,
    use_risk_overlays: bool = False,
    invert: bool = False,
) -> PortfolioManager:
    """Run one full event-driven backtest; return the filled PortfolioManager.

    Needed pieces
    -------------
    1. ``HistoricCSVDataHandler`` — only way strategies may see prices.
    2. ``TrendMomentumStrategy`` — emits ``SignalEvent``s (delay=1).
    3. ``PortfolioManager`` — turns signals into ``OrderEvent``s + tracks cash.
    4. ``SimulatedExecutionHandler`` — fills with the cost model.
    5. ``BacktestEngine`` — owns the queue and drives open/close bar phases.
    """
    # 1) Data: wrap the OHLCV frame. Downstream code must NOT read ``df`` raw.
    handler = HistoricCSVDataHandler({SYMBOL: df})

    # 2) Strategy: delay=1 → signal on bar T close, fill at bar T+1 open.
    #    delay=0 needs BacktestEngine(..., allow_same_print_fills=True) (D4).
    strategy = TrendMomentumStrategy(
        handler, SYMBOL, lookback=lookback, allow_short=True,
    )
    if invert:
        # Decoy family for demos: same lookback, flipped side (mean-reversion).
        strategy = InvertedStrategy(strategy)

    # 3) Portfolio: sizer chooses quantity; optional risk overlays clip exposure.
    portfolio = PortfolioManager(
        handler,
        INITIAL_CAPITAL,
        sizer=sizer or FixedUnitSizer(units_for(df)),
        margin_monitor=MarginMonitor(max_leverage=3.0) if use_risk_overlays else None,
        drawdown_breaker=(
            # D11: session-close roll (16:00 America/New_York). Daily bars
            # stamped 00:00 UTC still map to their own date's session.
            DailyDrawdownBreaker(max_intraday_dd=0.08) if use_risk_overlays else None
        ),
    )

    # 4) Execution: applies CostModel to produce FillEvents.
    execution = SimulatedExecutionHandler(cost_model or BASE_COSTS)

    # 5) Engine: Market → Signal → Order → Fill until the data is exhausted.
    BacktestEngine(handler, strategy, portfolio, execution).run_backtest()
    return portfolio


class InvertedStrategy:
    """Flip LONG↔SHORT while preserving delay (used only as a rejected decoy).

    We reject mean-reversion *a priori* for this hypothesis. Do **not** log
    known-junk decoys into the TrialsRegistry — that games or wrecks DSR.
    """

    def __init__(self, inner: TrendMomentumStrategy):
        self.inner = inner
        self.delay = inner.delay
        self.symbol = inner.symbol

    def matches_phase(self, phase: str) -> bool:
        return self.inner.matches_phase(phase)

    def calculate_signals(self, event, events_queue):
        # Capture the inner strategy's signals, then flip the side.
        class _Capture:
            def __init__(self):
                self.items = []

            def put(self, item):
                self.items.append(item)

        captured = _Capture()
        self.inner.calculate_signals(event, captured)
        for sig in captured.items:
            if sig.signal_type == LONG:
                flipped = SHORT
            elif sig.signal_type == SHORT:
                flipped = LONG
            else:
                flipped = EXIT
            events_queue.put(
                SignalEvent(
                    sig.timestamp, sig.symbol, flipped,
                    strength=sig.strength, delay=sig.delay,
                )
            )

    def vectorized_signals(self, data: dict):
        out = self.inner.vectorized_signals(data)
        return {k: -v for k, v in out.items()}


# ===========================================================================
# FAST-TRACK — vectorized twin of the event engine (for grids / MCPT)
# ===========================================================================
def fast_sharpe(
    df: pd.DataFrame, lookback: int, cost_model=None, *, invert: bool = False
) -> float:
    """Tearsheet Sharpe from the vectorized twin (D7: FastResult.sharpe is
    ``annualized_sharpe`` — simple returns, measured bar calendar)."""
    target = momentum_positions(df["close"], lookback=lookback)
    if invert:
        target = -target
    return fast_backtest(
        df, target, cost_model or BASE_COSTS,
        initial_capital=INITIAL_CAPITAL, units=units_for(df),
    ).sharpe


def fast_equity(
    df: pd.DataFrame, lookback: int, cost_model=None, *, invert: bool = False
) -> pd.Series:
    """Equity curve from the vectorized twin."""
    target = momentum_positions(df["close"], lookback=lookback)
    if invert:
        target = -target
    return fast_backtest(
        df, target, cost_model or BASE_COSTS,
        initial_capital=INITIAL_CAPITAL, units=units_for(df),
    ).equity


# ===========================================================================
# Small helpers used when logging trials / summarizing results
# ===========================================================================
def daily_pnl(equity: pd.Series) -> pd.Series:
    return equity.diff().fillna(0.0)


def period_sharpe(equity: pd.Series) -> float:
    """Per-period (daily) Sharpe — what the TrialsRegistry stores for DSR."""
    rets = equity.pct_change().dropna()
    if len(rets) < 2 or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std())


def moments(equity: pd.Series) -> dict:
    """Sharpe + higher moments needed by DSR / the registry."""
    rets = equity.pct_change().dropna()
    return {
        "sharpe_daily": period_sharpe(equity),
        "sharpe_ann": annualized_sharpe(equity),
        "skew": float(skew(rets)) if len(rets) > 2 else 0.0,
        "kurt": float(kurtosis(rets, fisher=False)) if len(rets) > 3 else 3.0,
        "n_obs": int(len(rets)),
    }
