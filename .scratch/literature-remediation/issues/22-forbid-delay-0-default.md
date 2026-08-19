# Forbid delay-0 fills by default

Type: task
Status: resolved
Part of: [Literature remediation decision map](../map.md)

## Goal

Same-print fills are unphysical by default. Keep the Temporal Firewall delay-0 **code path** behind an explicit opt-in.

Ruling: [Keep delay-0 as a firewall feature, or require minimum latency?](08-delay-0-policy.md) (notebook D4). Orthogonal to [Are delay-1 market entries and resting intra-bar stops orthogonal policies?](07-delay-entries-vs-stops.md).

## Files

- `quantester/engine.py` — `BacktestEngine.__init__` / `run_backtest` start checks
- `quantester/simple.py` — `run_backtest(...)` kwargs
- `quantester/strategy/base.py` — `delay` default already 1; `emit_target` still allows 0 (keep, gated at engine)
- `quantester/events.py` — `SignalEvent.delay` construction may stay `>= 0`; engine refuses to **run** delay-0 without the flag
- `tests/test_engine.py` — delay-0 strategy fixture
- `tests/test_portfolio.py` — delay-0 MOC rejection

## Acceptance criteria

- Add `allow_same_print_fills: bool = False` on `BacktestEngine` and `run_backtest`.
- If any strategy or queued `SignalEvent` has `delay == 0` and the flag is false, raise a clear `ValueError` naming Harris latency / unphysical same-print fills and how to opt in.
- `Strategy.delay` default remains 1. Do **not** delete open-phase intra-bar guard, `_open_visible_bars`, or delay-0 matching.
- Tests that demonstrate delay-0 pass `allow_same_print_fills=True`.
- Delay-1 entries after close stay the default live path (do not reopen).

## Tests

- Default engine + `delay=0` strategy → raises, no fills.
- Same setup with the flag → existing delay-0 look-ahead test still passes.
- Delay-1 strategies unchanged with the flag false.

## Out of scope

- Resting STOP vs EXIT-on-touch (already shipped).
- Changing `fill_at='close'` MOC for delay>=1.

## Answer

Shipped (commits 5eaf2d4, 4b0aa2b). `BacktestEngine(allow_same_print_fills=False)` refuses delay-0 strategies at construction AND delay-0 SignalEvents at dispatch (Harris latency message); `run_backtest(..., allow_same_print_fills=...)` forwards the flag. The intra-bar guard / `_open_visible_bars` / open-phase matching stay intact behind the opt-in; existing delay-0 tests opt in explicitly.
