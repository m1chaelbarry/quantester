"""GLD/GDX pairs-trading diagnostics.

Covers, on seeded synthetic cointegrated mock data (offline, no external API):

- hedge-ratio (beta_t) correctness against a direct scikit-learn OLS refit;
- simultaneous, opposite order placement for the two legs (signal, order, and
  fill level), with delay=1 fills at the next bar's open (temporal firewall);
- read-only firewall: the strategy never observes data after simulated time t;
- multi-symbol synchronization: a missing GDX bar pauses signals without
  erasing the timestamp from the master calendar;
- transaction costs (5 bps/leg/trade) strictly degrade Sharpe vs frictionless;
- Ernest Chan's historical truncation test (N=50) as a MANDATORY case, plus a
  negative control proving the runner detects deliberate look-ahead leakage.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from quantester.analytics.performance import summarize
from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.events import BUY, EXIT, LONG, SELL, SHORT, SignalEvent
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.portfolio.portfolio import HedgeRatioSizer, PortfolioManager
from quantester.strategy.base import Strategy
from quantester.strategy.pairs_trading import (
    FLAT,
    LONG_SPREAD,
    SHORT_SPREAD,
    PairsTradingStrategy,
    next_spread_state,
    ols_spread,
    zscore_of,
)
from quantester.utils.synthetic import make_cointegrated_pair, write_csvs
from quantester.validation.truncation_test import (
    FIVE_BPS_COST_MODEL,
    ZERO_COST_MODEL,
    run_pairs_backtest,
    run_pairs_truncation_test,
    truncate_on_master_calendar,
)

TRUE_BETA = 1.4
INITIAL_CAPITAL = 100_000.0


# --------------------------------------------------------------------------
# Instrumented harness components (test-only; the engine contract is untouched)
# --------------------------------------------------------------------------


class RecordingFirewallPairs(PairsTradingStrategy):
    """Pairs strategy that records every emitted signal and asserts, on each
    invocation, that the DataHandler exposes no data after simulated time t
    (read-only temporal firewall)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emitted: list[SignalEvent] = []

    def calculate_signals(self, event, events_queue):
        for symbol in (self.leg_y, self.leg_x):
            visible = self.data_handler.get_latest_bars(symbol, 10**9)
            if len(visible):
                assert visible.index.max() <= event.timestamp, (
                    f"firewall breach: {symbol} data up to "
                    f"{visible.index.max()} visible at {event.timestamp}"
                )
        strategy = self

        class _Tee:
            def put(self, item):
                strategy.emitted.append(item)
                events_queue.put(item)

        super().calculate_signals(event, _Tee())


class OrderTap(SimulatedExecutionHandler):
    """Execution handler that records every OrderEvent it receives."""

    def __init__(self, cost_model):
        super().__init__(cost_model)
        self.orders: list = []

    def execute_order(self, order, events_queue):
        self.orders.append(order)
        super().execute_order(order, events_queue)


# --------------------------------------------------------------------------
# Module-scoped fixtures: each expensive backtest runs exactly once
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pair_data() -> dict:
    """750 seeded mock GLD/GDX daily bars, cointegrated with beta=1.4."""
    return make_cointegrated_pair(n_bars=750, beta=TRUE_BETA, seed=7)


@pytest.fixture(scope="module")
def gross(pair_data):
    """Frictionless engine run with instrumented strategy and execution."""
    handler = HistoricCSVDataHandler(pair_data)
    strategy = RecordingFirewallPairs(handler)  # spec params: 252/20, +/-2.0, 0.5
    portfolio = PortfolioManager(
        handler, INITIAL_CAPITAL, sizer=HedgeRatioSizer(0.5)
    )
    execution = OrderTap(ZERO_COST_MODEL)
    BacktestEngine(handler, strategy, portfolio, execution).run_backtest()
    return SimpleNamespace(
        handler=handler, strategy=strategy, portfolio=portfolio,
        execution=execution,
    )


@pytest.fixture(scope="module")
def net(pair_data):
    """Identical run with the 5 bps/leg/trade transaction cost model."""
    return run_pairs_backtest(pair_data, cost_model=FIVE_BPS_COST_MODEL)


@pytest.fixture(scope="module")
def truncation(pair_data, tmp_path_factory):
    """Mandatory Chan truncation test artefacts (N=50), ingested from CSVs."""
    base = tmp_path_factory.mktemp("truncation")
    csv_map = write_csvs(pair_data, base / "data")  # CSV-path ingestion
    result = run_pairs_truncation_test(csv_map, base / "out", n_truncated=50)
    return SimpleNamespace(result=result, base=base)


# --------------------------------------------------------------------------
# Pure-helper correctness
# --------------------------------------------------------------------------


def test_ols_spread_recovers_known_coefficients():
    rng = np.random.default_rng(3)
    log_x = np.log(np.linspace(25.0, 40.0, 252))
    log_y = 1.7 + 1.35 * log_x + rng.normal(0.0, 1e-8, 252)
    alpha, beta, z = ols_spread(log_y, log_x)
    assert alpha == pytest.approx(1.7, abs=1e-5)
    assert beta == pytest.approx(1.35, abs=1e-5)
    assert abs(z) < 1e-6  # residual at the last point is pure noise


def test_zscore_of_matches_manual_sample_std():
    rng = np.random.default_rng(4)
    window = rng.normal(0.05, 0.02, 20)
    expected = (window[-1] - window.mean()) / window.std(ddof=1)
    assert zscore_of(window) == pytest.approx(expected, rel=1e-12)
    assert zscore_of(np.zeros(20)) is None  # zero variance -> undefined, not a signal


def test_next_spread_state_band_protocol():
    assert next_spread_state(FLAT, -2.3, 2.0, 0.5) == LONG_SPREAD
    assert next_spread_state(FLAT, 2.3, 2.0, 0.5) == SHORT_SPREAD
    assert next_spread_state(FLAT, -1.999, 2.0, 0.5) == FLAT
    assert next_spread_state(LONG_SPREAD, 0.4, 2.0, 0.5) == FLAT   # band exit
    assert next_spread_state(LONG_SPREAD, -0.5, 2.0, 0.5) == FLAT  # inclusive band
    assert next_spread_state(LONG_SPREAD, -1.7, 2.0, 0.5) == LONG_SPREAD  # hold
    assert next_spread_state(SHORT_SPREAD, 1.2, 2.0, 0.5) == SHORT_SPREAD  # hold


# --------------------------------------------------------------------------
# Hedge ratio correctness on mock GLD/GDX
# --------------------------------------------------------------------------


def test_hedge_ratio_matches_direct_sklearn_refit(gross, pair_data):
    """The strategy's trailing-window beta_t must equal a direct OLS refit of
    ln(GLD) on ln(GDX) over the same 252-day window (wiring correctness), and
    the beta path must track the data-generating beta=1.4."""
    diag = gross.strategy.diagnostics()
    log_gld = np.log(pair_data["GLD"]["close"].to_numpy()[-252:])
    log_gdx = np.log(pair_data["GDX"]["close"].to_numpy()[-252:])
    refit = LinearRegression().fit(log_gdx.reshape(-1, 1), log_gld)
    assert diag["beta"].iloc[-1] == pytest.approx(float(refit.coef_[0]), abs=1e-9)
    assert diag["alpha"].iloc[-1] == pytest.approx(float(refit.intercept_), abs=1e-9)
    # Stationary-spread construction: estimates track the data-generating
    # beta on average, with rolling-window estimation noise inside a wide
    # sanity band (no sign flips / degenerate fits).
    assert diag["beta"].mean() == pytest.approx(TRUE_BETA, abs=0.15)
    assert diag["beta"].between(0.9, 2.0).all()


# --------------------------------------------------------------------------
# Simultaneous, opposite order placement + temporal firewall
# --------------------------------------------------------------------------


def _by_timestamp(events):
    grouped: dict = {}
    for event in events:
        grouped.setdefault(event.timestamp, []).append(event)
    return grouped


def test_signals_are_simultaneous_and_opposite(gross):
    grouped = _by_timestamp(gross.strategy.emitted)
    assert grouped, "strategy emitted no signals on the mock pair"
    for timestamp, signals in grouped.items():
        assert len(signals) == 2, f"legs not simultaneous at {timestamp}"
        legs = {s.symbol: s.signal_type for s in signals}
        assert set(legs) == {"GLD", "GDX"}
        assert all(s.delay == 1 for s in signals)  # close T -> open T+1
        entries = {LONG, SHORT}
        if EXIT in legs.values():
            assert legs["GLD"] == legs["GDX"] == EXIT  # flat both, together
        else:
            assert set(legs.values()) == entries  # opposite directions
            y_sig = next(s for s in signals if s.symbol == "GLD")
            x_sig = next(s for s in signals if s.symbol == "GDX")
            assert y_sig.hedge_ratio == pytest.approx(1.0)
            assert x_sig.hedge_ratio > 0.0
            assert x_sig.hedge_ref_price == pytest.approx(y_sig.hedge_ref_price)


def test_orders_are_simultaneous_opposite_and_firewall_stamped(gross):
    grouped = _by_timestamp(gross.execution.orders)
    assert grouped, "portfolio placed no orders"
    for timestamp, orders in grouped.items():
        assert len(orders) == 2, f"legs not ordered together at {timestamp}"
        legs = {o.symbol: o for o in orders}
        assert set(legs) == {"GLD", "GDX"}
        directions = {o.direction for o in orders}
        assert directions == {BUY, SELL}  # opposite spread legs
        sigs = [s for s in gross.strategy.emitted if s.timestamp == timestamp]
        if EXIT not in {s.signal_type for s in sigs}:
            # q_X = -β q_Y: share ratio equals the X-leg hedge_ratio.
            x_ratio = next(s.hedge_ratio for s in sigs if s.symbol == "GDX")
            assert legs["GDX"].quantity / legs["GLD"].quantity == pytest.approx(
                x_ratio, rel=1e-9
            )
        for order in orders:
            assert order.quantity > 0
            # Temporal firewall: fill eligibility starts exactly one bar after
            # the signal bar (delay=1).
            assert order.earliest_fill_time == gross.handler.timestamp_at_offset(
                timestamp, 1
            )


def test_fills_execute_at_next_bar_open(gross):
    fills = gross.portfolio.fills
    assert fills, "no fills recorded"
    grouped = _by_timestamp(fills)
    for timestamp, leg_fills in grouped.items():
        assert len(leg_fills) == 2
        legs = {f.symbol: f for f in leg_fills}
        assert set(legs) == {"GLD", "GDX"}
        assert {f.direction for f in leg_fills} == {BUY, SELL}
        for fill in leg_fills:
            # Zero costs: fill price is exactly the T+1 open print.
            bar_open = float(
                gross.handler.bar_at(fill.symbol, timestamp)["open"]
            )
            assert fill.fill_price == pytest.approx(bar_open, rel=1e-12)
            assert fill.timestamp == timestamp  # stamped at the fill bar


def test_fill_timestamps_follow_signal_timestamps_by_one_bar(gross):
    signal_ts = sorted(_by_timestamp(gross.strategy.emitted))
    expected = {gross.handler.timestamp_at_offset(ts, 1) for ts in signal_ts}
    expected.discard(None)  # a signal on the final bar has no future fill bar
    actual = set(_by_timestamp(gross.portfolio.fills))
    assert actual == expected


# --------------------------------------------------------------------------
# Event form vs vectorized twin parity
# --------------------------------------------------------------------------


def test_event_signals_match_vectorized_twin(gross, pair_data):
    twin = gross.strategy.vectorized_signals(pair_data)
    for signal in gross.strategy.emitted:
        state_y = twin["GLD"].loc[: signal.timestamp].iloc[-1]
        state_x = twin["GDX"].loc[: signal.timestamp].iloc[-1]
        assert state_y == -state_x  # hedge symmetry
        expected = {LONG: 1.0, SHORT: -1.0, EXIT: 0.0}[signal.signal_type]
        actual = state_y if signal.symbol == "GLD" else state_x
        assert actual == expected


# --------------------------------------------------------------------------
# Multi-symbol synchronization under availability gaps
# --------------------------------------------------------------------------


def test_missing_leg_bars_pause_signals_without_erasing_timestamps():
    data = make_cointegrated_pair(n_bars=600, seed=11, gdx_missing_every=160)
    missing = data["GLD"].index.difference(data["GDX"].index)
    assert len(missing) == 4  # bars 50, 210, 370, 530 dropped from GDX only

    handler = HistoricCSVDataHandler(data)
    strategy = PairsTradingStrategy(handler, ols_window=60, zscore_window=10)
    handler.prime_data()
    emitted: list[SignalEvent] = []
    n_timestamps = 0
    while handler.continue_backtest:
        timestamp, bars = handler.advance()
        n_timestamps += 1
        handler.set_phase("close", timestamp)
        strategy.calculate_signals(
            SimpleNamespace(timestamp=timestamp, bars=bars),
            SimpleNamespace(put=emitted.append),
        )
    # Master calendar preserves every union timestamp (600), GDX has 596 bars.
    assert n_timestamps == 600
    assert len(data["GDX"]) == 596
    # No signal is ever emitted on a timestamp where a leg is untradeable...
    assert not any(s.timestamp in missing for s in emitted)
    # ...and the strategy resumes once both legs are tradeable again.
    assert emitted, "strategy never resumed after availability gaps"
    assert any(s.timestamp > missing[0] for s in emitted)


# --------------------------------------------------------------------------
# Transaction cost stress: 5 bps/leg must degrade performance logically
# --------------------------------------------------------------------------


def test_transaction_costs_degrade_sharpe(gross, net):
    gross_stats = summarize(gross.portfolio.equity_curve)
    net_stats = summarize(net.portfolio.equity_curve)
    for stats in (gross_stats, net_stats):
        for key in ("sharpe", "max_drawdown", "calmar"):
            assert np.isfinite(stats[key]), f"{key} not computable"
    total_costs = sum(f.total_cost for f in net.portfolio.fills)
    assert total_costs > 0.0  # friction model is actually engaged
    assert len(net.portfolio.fills) == len(gross.portfolio.fills)  # same trades
    assert net_stats["sharpe"] < gross_stats["sharpe"]  # costs degrade Sharpe
    assert net_stats["total_return"] < gross_stats["total_return"]
    assert net.portfolio.equity_curve.iloc[-1] < gross.portfolio.equity_curve.iloc[-1]


# --------------------------------------------------------------------------
# Mandatory: Ernest Chan's historical truncation test (N = 50)
# --------------------------------------------------------------------------


def test_truncation_no_lookahead_mandatory(truncation):
    result = truncation.result
    assert result.passed
    assert result.n_truncated == 50
    assert result.rows_compared == 700  # 750 master bars minus 50 truncated
    a = pd.read_csv(result.positions_a_path, parse_dates=["datetime"],
                    index_col="datetime")
    b = pd.read_csv(result.positions_b_path, parse_dates=["datetime"],
                    index_col="datetime")
    assert list(a.columns) == ["GLD", "GDX"]
    assert len(a) == 750 and len(b) == 700
    np.testing.assert_allclose(
        a.iloc[:-50].to_numpy(dtype=float), b.to_numpy(dtype=float),
        atol=1e-9, rtol=0.0,
    )


def test_truncation_detects_look_ahead_leakage(tmp_path):
    """Negative control: a strategy that peeks `horizon` bars ahead MUST be
    caught, with the first-divergence timestamp pinpointed in the error."""

    class FuturePeekingStrategy(Strategy):
        """Deliberately leaks: decides from closes AFTER bar T by reading the
        run's raw frame outside the DataHandler firewall (the exact failure
        mode the truncation test exists to catch)."""

        delay = 1

        def __init__(self, data_handler, horizon=5):
            self.data_handler = data_handler
            self._closes = data_handler._data["GLD"]["close"].to_numpy()
            self._index = data_handler._data["GLD"].index
            self._horizon = horizon
            self._state = FLAT

        def calculate_signals(self, event, events_queue):
            loc = self._index.get_loc(event.timestamp)
            if loc + self._horizon >= len(self._closes):
                return  # truncated run loses the peeked window: decisions differ
            target = LONG_SPREAD if (
                self._closes[loc + self._horizon] > self._closes[loc]
            ) else SHORT_SPREAD
            if target == self._state:
                return
            events_queue.put(
                SignalEvent(event.timestamp, "GLD",
                            LONG if target == LONG_SPREAD else SHORT, delay=1)
            )
            self._state = target

    data = make_cointegrated_pair(n_bars=400, seed=13)
    with pytest.raises(ValueError, match="Look-ahead leakage detected") as excinfo:
        run_pairs_truncation_test(
            data, tmp_path, n_truncated=20,
            strategy_factory=lambda handler: FuturePeekingStrategy(handler),
        )
    message = str(excinfo.value)
    assert "'GLD'" in message  # diverging symbol pinpointed
    b = pd.read_csv(tmp_path / "positions_B.csv", parse_dates=["datetime"],
                    index_col="datetime")
    match = re.search(r"diverge at (\d{4}-\d{2}-\d{2})", message)
    assert match is not None, "first-divergence timestamp not pinpointed"
    leaked_ts = pd.Timestamp(match.group(1), tz="UTC")
    # Divergence must surface in the tail of the overlap, where Run B no
    # longer has the future bars Run A peeked into.
    b_tail = b.index[-10]
    if b_tail.tzinfo is None:
        b_tail = b_tail.tz_localize("UTC")
    assert leaked_ts >= b_tail


def test_truncate_on_master_calendar_validation(pair_data):
    with pytest.raises(ValueError):
        truncate_on_master_calendar(pair_data, 0)
    with pytest.raises(ValueError):
        truncate_on_master_calendar(pair_data, 750)
    trimmed = truncate_on_master_calendar(pair_data, 50)
    assert len(trimmed["GLD"]) == 700
    assert trimmed["GLD"].index[-1] == pair_data["GLD"].index[-51]
