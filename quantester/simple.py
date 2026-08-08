"""Trader-facing one-call backtest API.

Most of Quantester is deliberately low-level (five modules on an event queue).
This module is the readable front door for someone who wants to:

1. Pass price data + a strategy
2. Get equity, trades, and a plain-English summary back

It does **not** bypass the event queue — it wires the same five modules every
example uses, with safe defaults and checks that catch common mistakes early.

Example::

    from quantester import run_backtest, MovingAverageCrossStrategy
    from quantester.utils.synthetic import make_synthetic_ohlcv

    data = {"AAA": make_synthetic_ohlcv("AAA", seed=1)}
    result = run_backtest(
        data,
        MovingAverageCrossStrategy,
        symbol="AAA",
        fast=10,
        slow=40,
    )
    result.print_summary()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .analytics.performance import summarize
from .data.csv_handler import HistoricCSVDataHandler
from .engine import BacktestEngine
from .execution.costs import CostModel
from .execution.simulator import SimulatedExecutionHandler
from .portfolio.portfolio import PercentEquitySizer, PortfolioManager
from .strategy.base import Strategy

# Callable that receives the data handler and returns a ready Strategy.
StrategyFactory = Callable[[Any], Strategy]


@dataclass
class BacktestResult:
    """Readable wrapper around a finished PortfolioManager run.

    Traders usually care about four numbers: return, Sharpe, max drawdown, and
    how many trades fired. Everything else (fills ledger, position history) is
    still available on ``portfolio`` for deeper work.
    """

    portfolio: PortfolioManager
    stats: dict

    @property
    def equity(self) -> pd.Series:
        return self.portfolio.equity_curve

    @property
    def trades(self) -> list:
        return self.portfolio.trades

    @property
    def fills(self) -> list:
        return self.portfolio.fills

    @property
    def total_return(self) -> float:
        return float(self.stats.get("total_return", 0.0))

    @property
    def sharpe(self) -> float:
        return float(self.stats.get("sharpe", 0.0))

    @property
    def max_drawdown(self) -> float:
        return float(self.stats.get("max_drawdown", 0.0))

    def summary(self) -> str:
        """Plain-English one-screen summary of the run."""
        n_trades = len(self.trades)
        n_fills = len(self.fills)
        dd_days = self.stats.get("max_drawdown_duration_days", 0)
        calmar = self.stats.get("calmar", 0.0)
        lines = [
            "Backtest summary",
            "----------------",
            f"  Total return : {self.total_return:+.2%}",
            f"  Sharpe       : {self.sharpe:+.3f}  (higher is better; ~0 = no edge)",
            f"  Max drawdown : {self.max_drawdown:+.2%}  (worst peak-to-trough loss)",
            f"  Drawdown days: {dd_days}",
            f"  Calmar       : {calmar:+.3f}  (return / |drawdown|)",
            f"  Trades       : {n_trades} round-trips  ({n_fills} fills)",
        ]
        if n_trades == 0:
            lines.append(
                "  Note         : no trades — check warmup bars, signal rules, "
                "or whether the symbol had any available bars."
            )
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary())


def _coerce_data(data: dict | pd.DataFrame, symbol: str | None) -> dict:
    """Accept a single DataFrame or a {symbol: frame_or_path} map."""
    if isinstance(data, pd.DataFrame):
        if not symbol:
            raise ValueError(
                "When you pass a single price table, also pass symbol='TICKER' "
                "so Quantester knows what to call it."
            )
        return {symbol: data}
    if not isinstance(data, dict) or not data:
        raise ValueError(
            "data must be a non-empty dict of {symbol: DataFrame_or_csv_path}, "
            "or a single DataFrame plus symbol=..."
        )
    for key, value in data.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"data keys must be symbol names (strings); got {key!r}")
        if isinstance(value, (str, Path)):
            continue
        if not isinstance(value, pd.DataFrame):
            raise ValueError(
                f"data[{key!r}] must be a pandas DataFrame or a CSV path; "
                f"got {type(value).__name__}."
            )
        required = {"open", "high", "low", "close"}
        missing = required - set(value.columns)
        if missing:
            raise ValueError(
                f"data[{key!r}] is missing OHLCV columns {sorted(missing)}. "
                "Need at least open, high, low, close (volume recommended)."
            )
    return data


def _build_strategy(
    handler,
    strategy: Strategy | StrategyFactory | type,
    strategy_kwargs: dict | None,
) -> Strategy:
    """Turn a class / factory / (rejected) instance into a live Strategy."""
    kwargs = dict(strategy_kwargs or {})
    if isinstance(strategy, type) and issubclass(strategy, Strategy):
        try:
            return strategy(handler, **kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Could not construct {strategy.__name__}(handler, **{kwargs}). "
                "Pass the keyword arguments the strategy expects "
                "(e.g. symbol=, fast=, slow=), or pass a factory: "
                "lambda handler: MyStrategy(handler, ...)."
            ) from exc
    if isinstance(strategy, Strategy):
        raise TypeError(
            "Pass a strategy *class* or a factory function, not an already-built "
            "instance. The data handler must be created first, then the strategy.\n"
            "  OK:  run_backtest(data, MovingAverageCrossStrategy, symbol='AAA', ...)\n"
            "  OK:  run_backtest(data, lambda h: MyStrategy(h, 'AAA', lookback=20))\n"
            "  Bad: run_backtest(data, MyStrategy(some_old_handler, ...))"
        )
    if callable(strategy):
        built = strategy(handler)
        if not isinstance(built, Strategy):
            raise TypeError(
                "strategy factory must return a Strategy instance; "
                f"got {type(built).__name__}."
            )
        return built
    raise TypeError(
        "strategy must be a Strategy subclass, or a callable "
        "handler -> Strategy. "
        f"Got {type(strategy).__name__}."
    )


def run_backtest(
    data: dict | pd.DataFrame,
    strategy: Strategy | StrategyFactory | type,
    *,
    symbol: str | None = None,
    capital: float = 100_000.0,
    equity_pct: float = 0.9,
    sizer=None,
    costs: CostModel | None = None,
    strategy_kwargs: dict | None = None,
    **kwargs,
) -> BacktestResult:
    """Run a full event-driven backtest with safe defaults.

    Parameters
    ----------
    data
        ``{symbol: DataFrame_or_csv_path}`` map, **or** a single DataFrame
        (then pass ``symbol=``).
    strategy
        A ``Strategy`` subclass (plus kwargs), or ``lambda handler: Strategy(...)``.
    symbol
        Required when ``data`` is a single DataFrame. Also forwarded into
        ``strategy_kwargs`` if the strategy needs it and you did not pass it
        there already.
    capital
        Starting cash.
    equity_pct
        Fraction of equity to deploy per signal when using the default sizer
        (0.9 = use up to 90% of the account). Ignored if ``sizer=`` is set.
    sizer
        Optional custom sizer; default is ``PercentEquitySizer(equity_pct)``.
    costs
        Transaction-cost model. Default ``CostModel()`` is a liquid-equity
        starting point (commission + spread + slippage + impact).
    strategy_kwargs
        Extra keywords for a Strategy class (e.g. ``{"fast": 10, "slow": 40}``).
        Bare kwargs like ``fast=10`` are also accepted and merged here.
    """
    if capital <= 0:
        raise ValueError(f"capital must be positive (starting cash); got {capital!r}")
    if sizer is None and not (0.0 < float(equity_pct) <= 1.0):
        raise ValueError(
            f"equity_pct must be in (0, 1] — e.g. 0.9 means 'use 90% of equity'. "
            f"Got {equity_pct!r}."
        )

    # Bare kwargs (fast=, slow=, lookback=, ...) merge into strategy_kwargs so
    # call sites stay flat and readable for non-coders.
    merged_kwargs = dict(strategy_kwargs or {})
    merged_kwargs.update(kwargs)
    if symbol is not None and "symbol" not in merged_kwargs:
        merged_kwargs["symbol"] = symbol

    data_map = _coerce_data(data, symbol=merged_kwargs.get("symbol", symbol))
    handler = HistoricCSVDataHandler(data_map)
    built = _build_strategy(handler, strategy, merged_kwargs)
    portfolio = PortfolioManager(
        handler,
        capital,
        sizer=sizer if sizer is not None else PercentEquitySizer(equity_pct),
    )
    execution = SimulatedExecutionHandler(costs if costs is not None else CostModel())
    BacktestEngine(handler, built, portfolio, execution).run_backtest()
    equity = portfolio.equity_curve
    stats = summarize(equity) if len(equity) else {
        "total_return": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_duration_days": 0,
        "calmar": 0.0,
    }
    return BacktestResult(portfolio=portfolio, stats=stats)
