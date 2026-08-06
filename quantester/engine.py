"""Centralized synchronous event loop (Report 1 section 3 pseudocode).

Look-ahead safety is enforced by a State-Based Temporal Firewall (Cross-Ref
section 3.1), not a hardcoded T+1:

- Every bar is processed in two phases: 'open' then 'close'.
- Orders carry `earliest_fill_time`, derived from the strategy's explicit `delay`
  parameter and enforced by the execution ledger.
- delay=1 (default): signals computed on bar T's close fill at bar T+1's open.
- delay=0 (Delay-0 strategies): signals computed at bar T's open fill at bar T's
  open, under the intra-bar guard (Cross-Ref-2 section 3.B): the DataHandler only
  exposes data strictly before the fill timestamp (bars up to T-1 plus T's open
  print), so a same-bar fill can never use same-bar close information.

The DataHandler read-only stream never exposes data beyond simulated time t
regardless of delay.
"""

from __future__ import annotations

import queue

from .events import FILL, MARKET, OPEN, ORDER, SIGNAL, MarketEvent


class BacktestEngine:
    def __init__(self, data_handler, strategies, portfolio, execution_handler):
        if not isinstance(strategies, (list, tuple)):
            strategies = [strategies]
        self.events = queue.Queue()
        self.data_handler = data_handler
        self.strategies = list(strategies)
        self.portfolio = portfolio
        self.execution_handler = execution_handler

    def run_backtest(self):
        """Main chronological causal loop: outer data stream, inner queue drain."""
        dh = self.data_handler
        dh.prime_data()

        while dh.continue_backtest:
            timestamp, bars = dh.advance()

            # Open phase: pending-order ledger drains first, then delay=0 strategies
            dh.set_phase(OPEN, timestamp)
            self.events.put(MarketEvent(timestamp=timestamp, bars=bars, phase=OPEN))
            self._drain_queue()

            # Close phase: mark-to-market, then delay>=1 strategies
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
                    self.execution_handler.on_market(event, self.events)
                    for strategy in self.strategies:
                        if strategy.matches_phase(OPEN):
                            strategy.calculate_signals(event, self.events)
                else:
                    self.portfolio.update_portfolio_valuation(event, self.events)
                    for strategy in self.strategies:
                        if strategy.matches_phase("close"):
                            strategy.calculate_signals(event, self.events)

            elif event.type == SIGNAL:
                self.portfolio.update_from_signal(event, self.events)

            elif event.type == ORDER:
                self.execution_handler.execute_order(event, self.events)

            elif event.type == FILL:
                self.portfolio.update_from_fill(event)

            self.events.task_done()
