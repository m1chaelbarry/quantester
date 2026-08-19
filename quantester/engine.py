"""Centralized synchronous event loop (Report 1 section 3 pseudocode).

Look-ahead safety is enforced by a State-Based Temporal Firewall (Cross-Ref
section 3.1), not a hardcoded T+1:

- Every bar is processed in two phases: 'open' then 'close'.
- Open-phase MarketEvents expose only the open print to strategies (never
  high/low/close of the forming bar). Execution still sees the full bar for
  open fills, but stop/limit touch tests wait until close phase.
- Orders carry `earliest_fill_time`, derived from the strategy's explicit `delay`
  parameter and enforced by the execution ledger.
- delay=1 (default): signals computed on bar T's close fill at bar T+1's open.
- delay=0 (Delay-0 strategies): signals computed at bar T's open fill at bar T's
  open, under the intra-bar guard (bars up to T-1 plus T's open print only).

The DataHandler read-only stream never exposes data beyond simulated time t
regardless of delay.
"""

from __future__ import annotations

import queue

import pandas as pd

from .events import FILL, MARKET, OPEN, ORDER, SIGNAL, MarketEvent


def _open_visible_bars(bars: dict) -> dict:
    """Restrict open-phase event bars to the open print only (no H/L/C leak)."""
    out = {}
    for symbol, bar in bars.items():
        if bar is None:
            out[symbol] = None
            continue
        out[symbol] = pd.Series({"open": float(bar["open"])})
    return out


def _same_print_fill_error(what: str) -> ValueError:
    return ValueError(
        f"{what} requests delay=0: a fill at the same print the strategy just "
        "observed is unphysical without explicit latency modeling (Harris, "
        "Trading and Exchanges; ruling D4). The temporal-firewall delay-0 "
        "path stays available behind an explicit opt-in: pass "
        "allow_same_print_fills=True to BacktestEngine."
    )


def _require_callable(obj, name: str, role: str) -> None:
    if not hasattr(obj, name) or not callable(getattr(obj, name)):
        raise TypeError(
            f"{role} is missing required method {name}(). "
            f"Got {type(obj).__name__}. "
            "Did you pass the five pieces in the wrong order? "
            "BacktestEngine(data_handler, strategy, portfolio, execution_handler)."
        )


class BacktestEngine:
    """Centralized synchronous event loop.

    ``allow_same_print_fills`` (default False, ruling D4): delay-0 strategies
    fill at the same print they just observed, which is unphysical without
    latency modeling. Refused unless this flag is True; the intra-bar guard
    still applies when opted in.
    """

    def __init__(self, data_handler, strategies, portfolio, execution_handler,
                 allow_same_print_fills: bool = False):
        if data_handler is None:
            raise TypeError("data_handler is required (e.g. HistoricCSVDataHandler).")
        if portfolio is None:
            raise TypeError("portfolio is required (e.g. PortfolioManager).")
        if execution_handler is None:
            raise TypeError(
                "execution_handler is required (e.g. SimulatedExecutionHandler)."
            )
        if not isinstance(strategies, (list, tuple)):
            strategies = [strategies]
        if not strategies or any(s is None for s in strategies):
            raise TypeError(
                "Pass at least one Strategy (or a list of them). "
                "A strategy decides when to go long/short/flat."
            )
        _require_callable(data_handler, "prime_data", "data_handler")
        _require_callable(data_handler, "advance", "data_handler")
        _require_callable(portfolio, "update_from_signal", "portfolio")
        _require_callable(portfolio, "update_from_fill", "portfolio")
        _require_callable(execution_handler, "execute_order", "execution_handler")
        for i, strategy in enumerate(strategies):
            _require_callable(strategy, "calculate_signals", f"strategies[{i}]")
            delay = getattr(strategy, "delay", 1)
            if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0:
                raise ValueError(
                    f"strategies[{i}] ({type(strategy).__name__}).delay must be "
                    f"an integer >= 0; got {delay!r}."
                )
        self.allow_same_print_fills = bool(allow_same_print_fills)
        if not self.allow_same_print_fills:
            for i, strategy in enumerate(strategies):
                if getattr(strategy, "delay", 1) == 0:
                    raise _same_print_fill_error(
                        f"strategies[{i}] ({type(strategy).__name__})"
                    )
        self.events = queue.Queue()
        self.data_handler = data_handler
        self.strategies = list(strategies)
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        # Prefer wiring the handler so open-phase cost proxies can use prior bars.
        if getattr(self.execution_handler, "data_handler", None) is None:
            try:
                self.execution_handler.data_handler = data_handler
            except AttributeError:
                # Handler exposes a read-only data_handler; leave unwired.
                pass

    def run_backtest(self):
        """Main chronological causal loop: outer data stream, inner queue drain."""
        dh = self.data_handler
        dh.prime_data()

        while dh.continue_backtest:
            timestamp, bars = dh.advance()

            # Open phase: MARKET ledger first (full bars), then delay=0 strategies
            # see open-only MarketEvent bars.
            dh.set_phase(OPEN, timestamp)
            self.execution_handler.on_market(
                MarketEvent(timestamp=timestamp, bars=bars, phase=OPEN), self.events
            )
            self.events.put(
                MarketEvent(
                    timestamp=timestamp, bars=_open_visible_bars(bars), phase=OPEN
                )
            )
            self._drain_queue()

            # Close phase: mark-to-market, stop/limit ledger, then delay>=1 strategies
            dh.set_phase("close", timestamp)
            self.events.put(MarketEvent(timestamp=timestamp, bars=bars, phase="close"))
            self._drain_queue()

        return self.portfolio

    def _drain_queue(self):
        while True:
            try:
                event = self.events.get(block=False)
            except queue.Empty:
                break

            if event.type == MARKET:
                if event.phase == OPEN:
                    # Execution already drained for this open (see run_backtest).
                    for strategy in self.strategies:
                        if strategy.matches_phase(OPEN):
                            self._calculate_signals(strategy, event)
                else:
                    self.portfolio.update_portfolio_valuation(event, self.events)
                    # STOP/LIMIT touch tests once full OHLC is known.
                    self.execution_handler.on_market(event, self.events)
                    for strategy in self.strategies:
                        if strategy.matches_phase("close"):
                            self._calculate_signals(strategy, event)

            elif event.type == SIGNAL:
                if event.delay == 0 and not self.allow_same_print_fills:
                    raise _same_print_fill_error(
                        f"SignalEvent({event.symbol} {event.signal_type})"
                    )
                self.portfolio.update_from_signal(event, self.events)

            elif event.type == ORDER:
                self.execution_handler.execute_order(event, self.events)

            elif event.type == FILL:
                self.portfolio.update_from_fill(event)

            self.events.task_done()

    def _calculate_signals(self, strategy, event) -> None:
        """Run ``calculate_signals`` with ``source_ohlcv`` sealed on the handler."""
        with self.data_handler.seal_source_ohlcv():
            strategy.calculate_signals(event, self.events)
