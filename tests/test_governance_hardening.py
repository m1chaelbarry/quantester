"""Margin restriction, portfolio accounting invariants, data audit, gates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantester.analytics.trials_registry import TrialsRegistry, auto_register_from_equity
from quantester.data.audit import FAIL, PASS, WARN, audit_ohlcv_frame, ensure_utc_index
from quantester.data.streaming import StreamingDataHandler, normalize_ohlcv_frame
from quantester.events import BUY, LONG, EXIT, FillEvent, MarketEvent, SignalEvent
from quantester.portfolio.portfolio import FixedUnitSizer, PortfolioManager
from quantester.portfolio.risk import MarginMonitor
from quantester.validation.gates import (
    NOT_VALIDATED,
    VALIDATED,
    build_validation_report,
    evaluate_gates,
)


class _Queue(list):
    def put(self, item):
        self.append(item)


def test_ensure_utc_and_normalize():
    naive = pd.bdate_range("2024-01-01", periods=3)
    utc = ensure_utc_index(naive)
    assert str(utc.tz) == "UTC"
    df = pd.DataFrame(
        {"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
        index=naive,
    )
    out = normalize_ohlcv_frame(df, "AAA")
    assert out.index.tz is not None


def test_data_audit_pass_warn_fail():
    idx = pd.bdate_range("2024-01-01", periods=5, tz="UTC")
    good = pd.DataFrame(
        {
            "open": [10, 11, 12, 13, 14],
            "high": [11, 12, 13, 14, 15],
            "low": [9, 10, 11, 12, 13],
            "close": [10.5, 11.5, 12.5, 13.5, 14.5],
            "volume": [100, 100, 100, 100, 100],
        },
        index=idx,
        dtype=float,
    )
    report = audit_ohlcv_frame(
        good,
        "AAA",
        adjustment_policy="split_dividend_adjusted",
        corporate_actions_documented=True,
        delistings_considered=True,
        survivorship_bias_considered=True,
        historical_universe_documented=True,
        trading_calendar_documented=True,
        expected_freq="B",
    )
    assert report.status in {PASS, WARN}
    assert not report.failures()

    bad = good.copy()
    bad.iloc[1, bad.columns.get_loc("high")] = 1.0  # high < open/close
    fail_report = audit_ohlcv_frame(bad, "AAA")
    assert fail_report.status == FAIL

    naive = good.copy()
    naive.index = pd.bdate_range("2024-01-01", periods=5)  # naive
    naive_report = audit_ohlcv_frame(naive, "AAA")
    assert any(c.name == "timestamp_timezone_explicit" and c.status == FAIL
               for c in naive_report.checks)


def test_margin_blocks_new_entries_until_recovery():
    idx = pd.bdate_range("2024-01-02", periods=5, tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1e6},
        index=idx,
    )
    handler = StreamingDataHandler({"AAA": df})
    monitor = MarginMonitor(max_leverage=1.5, liquidation_fraction=0.5)
    portfolio = PortfolioManager(
        handler, 100_000.0, sizer=FixedUnitSizer(5000), margin_monitor=monitor
    )
    # Large long: qty=3000 @ 100 → gross 300k / equity 100k = 3x leverage
    portfolio.update_from_fill(
        FillEvent(idx[0], "AAA", 3000, BUY, 100.0, 0.0, 0.0)
    )
    # cash = 100k - 300k = -200k; equity = -200k + 3000*100 = 100k; leverage=3
    queue = _Queue()
    bar = pd.Series({"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1e6})
    portfolio.update_portfolio_valuation(
        MarketEvent(idx[0], {"AAA": bar}, phase="close"), queue
    )
    assert monitor.restricted
    assert monitor.breach_count == 1
    cancels = [o for o in queue if o.order_type == "CANCEL"]
    liqs = [o for o in queue if o.order_type == "MARKET"]
    assert cancels and liqs

    n = len(queue)
    handler.set_phase("close", idx[0])
    # Target = +5000 would increase |position| from 3000 → blocked.
    portfolio.update_from_signal(SignalEvent(idx[0], "AAA", LONG, strength=1.0), queue)
    assert len(queue) == n  # entry blocked

    # Risk-reducing EXIT toward flat is allowed while restricted.
    portfolio.update_from_signal(SignalEvent(idx[0], "AAA", EXIT), queue)
    assert len(queue) > n

    # Apply liquidation fill to recover below max leverage.
    portfolio.update_from_fill(
        FillEvent(idx[1], "AAA", 1500, "SELL", 100.0, 0.0, 0.0)
    )
    queue2 = _Queue()
    portfolio.update_portfolio_valuation(
        MarketEvent(idx[1], {"AAA": bar}, phase="close"), queue2
    )
    # qty=1500, cash=-200k+150k=-50k, equity=-50k+150k=100k, leverage=1.5 → not >
    assert not monitor.restricted


def test_accounting_invariant_and_no_double_slippage():
    idx = pd.bdate_range("2024-01-02", periods=2, tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e6},
        index=idx,
    )
    handler = StreamingDataHandler({"AAA": df})
    portfolio = PortfolioManager(handler, 100_000.0)
    fill = FillEvent(
        idx[0], "AAA", 10, BUY, 100.5, commission=1.0, slippage_cost=5.0,
        reference_price=100.0,
    )
    portfolio.update_from_fill(fill)
    # Cash charged fill_price + commission only (slippage embedded, not double).
    assert portfolio.cash == pytest.approx(100_000.0 - 10 * 100.5 - 1.0)
    portfolio.last_prices["AAA"] = 100.0
    inv = portfolio.accounting_invariant()
    assert inv["ok"]
    assert inv["equity"] == pytest.approx(portfolio.cash + 10 * 100.0)


def test_validation_gates_block_validated_on_fail():
    gates = evaluate_gates(
        data_audit_status="PASS",
        truncation_passed=True,
        parity_passed=True,
        execution_stress_passed=True,
        cpcv_passed=True,
        pbo_passed=True,
        pbo_value=0.05,
        dsr_value=0.99,
        untouched_oos_passed=False,  # mandatory fail
        monte_carlo_passed=True,
        sensitivity_passed=True,
        accounting_invariant_passed=True,
        execution_assumptions_documented=True,
    )
    report = build_validation_report(
        gates,
        trial_count=12,
        code_version="0.1.0",
        assumptions={"spread_bps": 5},
    )
    assert report.status == NOT_VALIDATED
    assert not report.validated

    gates2 = evaluate_gates(
        data_audit_status="PASS",
        truncation_passed=True,
        parity_passed=None,
        execution_stress_passed=True,
        cpcv_passed=True,
        pbo_passed=True,
        pbo_value=0.05,
        dsr_value=0.99,
        untouched_oos_passed=True,
        monte_carlo_passed=True,
        sensitivity_passed=None,
        accounting_invariant_passed=True,
        execution_assumptions_documented=True,
    )
    report2 = build_validation_report(gates2, trial_count=3, code_version="0.1.0")
    assert report2.status == VALIDATED
    assert report2.validated


def test_auto_register_experiment():
    idx = pd.bdate_range("2024-01-01", periods=50, tz="UTC")
    equity = pd.Series(100_000 * np.cumprod(1 + np.full(50, 0.001)), index=idx)
    reg = TrialsRegistry()
    info = auto_register_from_equity(
        reg,
        equity,
        strategy_id="demo",
        params={"fast": 10},
        data_source="synthetic",
        universe=["AAA"],
        cost_model={"spread_bps": 5},
        random_seed=7,
        code_version="0.1.0",
    )
    assert reg.n_trials() == 1
    assert info["experiment_hash"]
    assert info["experiment_hash"] in reg.experiment_ids()
    best = reg.best_trial()
    assert best["strategy_id"] == "demo"
