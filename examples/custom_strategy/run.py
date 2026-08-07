"""Tutorial companion: build a momentum strategy from scratch and backtest it.

Follows docs/tutorials/creating-a-strategy.md step by step: synthetic data ->
custom Strategy subclass -> PercentEquitySizer -> cost-adjusted execution ->
tearsheet -> truncation test -> fast-track parity -> MCPT.

Run from the repo root:  python examples/custom_strategy/run.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"

import numpy as np

from quantester.analytics.performance import summarize
from quantester.analytics.tearsheet import generate_tearsheet
from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.events import EXIT, LONG, SignalEvent
from quantester.execution.costs import CostModel
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.montecarlo.fast_track import fast_backtest
from quantester.montecarlo.permutation import masters_p_value, permute_log_changes
from quantester.portfolio.portfolio import PercentEquitySizer, PortfolioManager
from quantester.strategy.base import Strategy
from quantester.utils.synthetic import make_synthetic_ohlcv
from quantester.validation.truncation import run_truncation_test

INITIAL_CAPITAL = 100_000.0
N_REPS = 200  # demo-scale; use >= 1,000 for production conclusions


# --------------------------------------------------------------------- Step 3
class MomentumStrategy(Strategy):
    """Long when the lookback-bar close-to-close momentum is positive; flat
    otherwise. delay=1: signal at close T, fill at open T+1."""

    def __init__(self, data_handler, symbol: str, lookback: int = 20):
        self.data_handler = data_handler
        self.symbol = symbol
        self.lookback = lookback
        self.delay = 1
        self._position = 0.0

    def calculate_signals(self, event, events_queue):
        if event.bars.get(self.symbol) is None:
            return  # availability mask: untradeable at this timestamp
        bars = self.data_handler.get_latest_bars(self.symbol, self.lookback + 1)
        if len(bars) < self.lookback + 1:
            return  # not enough history yet
        momentum = bars["close"].iloc[-1] / bars["close"].iloc[0] - 1.0
        if momentum > 0 and self._position <= 0:
            events_queue.put(SignalEvent(event.timestamp, self.symbol,
                                         LONG, strength=1.0, delay=self.delay))
            self._position = 1.0
        elif momentum <= 0 and self._position > 0:
            events_queue.put(SignalEvent(event.timestamp, self.symbol,
                                         EXIT, strength=1.0, delay=self.delay))
            self._position = 0.0

    def vectorized_signals(self, data: dict):
        close = data[self.symbol]["close"]
        momentum = close / close.shift(self.lookback) - 1.0
        return {self.symbol: (momentum > 0).astype(float)}


# ------------------------------------------------------------------ machinery
def run_backtest(df, lookback: int = 20, truncate_last: int | None = None):
    if truncate_last:
        df = df.iloc[:-truncate_last]
    handler = HistoricCSVDataHandler({"AAA": df})
    strategy = MomentumStrategy(handler, "AAA", lookback=lookback)
    portfolio = PortfolioManager(handler, INITIAL_CAPITAL,
                                 sizer=PercentEquitySizer(0.9))
    engine = BacktestEngine(handler, strategy, portfolio,
                            SimulatedExecutionHandler(CostModel()))
    engine.run_backtest()
    return portfolio


def main():
    print("=" * 72)
    print("Quantester tutorial: momentum strategy from scratch")
    print("=" * 72)

    # Step 1-2: data + handler live inside run_backtest.
    df = make_synthetic_ohlcv("AAA", n_bars=750, s0=100.0,
                              mu=0.10, sigma=0.22, seed=1)

    # Steps 4-7: size, execute, run, read results.
    portfolio = run_backtest(df, lookback=20)
    equity = portfolio.equity_curve
    stats = summarize(equity)
    print(f"Backtest: total return {stats['total_return']:+.2%}  "
          f"sharpe {stats['sharpe']:+.3f}  max DD {stats['max_drawdown']:+.2%}  "
          f"calmar {stats['calmar']:+.3f}")
    print(f"Trades: {len(portfolio.trades)} round-trips, "
          f"{len(portfolio.fills)} fills")

    # Step 8: tearsheet.
    generate_tearsheet(equity, OUTPUT_DIR / "momentum_tearsheet.png",
                       title="Momentum(20) on AAA")
    print(f"Tearsheet written to {OUTPUT_DIR / 'momentum_tearsheet.png'}")

    # Step 9: truncation test (look-ahead leak detector).
    result = run_truncation_test(
        lambda n: run_backtest(df, 20, truncate_last=n).positions_history,
        n_truncated=30,
    )
    print(result)

    # Step 3c check: fast-track parity between event form and vectorized twin.
    twin = MomentumStrategy(None, "AAA", lookback=20)
    target = twin.vectorized_signals({"AAA": df})["AAA"]
    fast = fast_backtest(df, target, CostModel(),
                         initial_capital=INITIAL_CAPITAL, units=1.0)
    # The event run used PercentEquitySizer; parity is exact under a fixed-unit
    # sizing, so re-run the event engine with a FixedUnitSizer of 1 share.
    from quantester.portfolio.portfolio import FixedUnitSizer

    handler = HistoricCSVDataHandler({"AAA": df})
    parity_portfolio = PortfolioManager(handler, INITIAL_CAPITAL,
                                        sizer=FixedUnitSizer(1.0))
    BacktestEngine(handler, MomentumStrategy(handler, "AAA", 20),
                   parity_portfolio, SimulatedExecutionHandler(CostModel())
                   ).run_backtest()
    diff = (parity_portfolio.equity_curve - fast.equity).abs().max()
    print(f"Fast-track parity: max |equity diff| = {diff:.2e}")

    # Step 10: MCPT on the fast-track (retrain lookback choice per permutation).
    close = df["close"]

    def optimizer(series):
        ohlc = df.copy()
        ohlc["close"] = series.reindex(ohlc.index).ffill().bfill()
        best = -np.inf
        for lookback in (10, 20, 40):
            tgt = MomentumStrategy(None, "AAA", lookback) \
                .vectorized_signals({"AAA": ohlc})["AAA"]
            best = max(best, fast_backtest(ohlc, tgt, CostModel()).sharpe)
        return best

    rng = np.random.default_rng(7)
    original = optimizer(close)
    permuted = np.array(
        [optimizer(permute_log_changes(close, rng)) for _ in range(N_REPS - 1)]
    )
    p_value = masters_p_value(original, permuted)
    verdict = "SIGNIFICANT p<0.05" if p_value < 0.05 else "not significant"
    print(f"MCPT p-value ({N_REPS} reps): {p_value:.4f} ({verdict})")


if __name__ == "__main__":
    main()
