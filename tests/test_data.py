"""DataHandler firewall, availability masks, bar sampling, ETF trick."""

import numpy as np
import pandas as pd
import pytest

from quantester.data.bars import dollar_bars, tick_imbalance_bars
from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.utils.etf_trick import ETFTrick
from quantester.utils.synthetic import make_synthetic_ohlcv


def test_outer_join_preserves_timestamps_and_masks_missing(ohlc_with_missing):
    handler = HistoricCSVDataHandler(ohlc_with_missing)
    expected = len(set(ohlc_with_missing["AAA"].index) | set(ohlc_with_missing["BBB"].index))
    assert len(handler._master_index) == expected

    missing_ts = ohlc_with_missing["AAA"].index.difference(ohlc_with_missing["BBB"].index)
    assert len(missing_ts) > 0
    handler.prime_data()
    seen_missing = False
    while handler.continue_backtest:
        ts, bars = handler.advance()
        if ts in set(missing_ts):
            seen_missing = True
            assert bars["BBB"] is None          # untradeable, not erased
            assert bars["AAA"] is not None
    assert seen_missing


def test_firewall_visibility_per_phase(ohlc):
    handler = HistoricCSVDataHandler({"AAA": ohlc})
    handler.prime_data()
    ts, _ = handler.advance()  # first bar
    ts2, _ = handler.advance()  # second bar

    handler.set_phase("open", ts2)
    visible_open = handler.get_latest_bars("AAA", 10)
    assert visible_open.index.max() < ts2       # intra-bar guard
    assert handler.get_current_open("AAA") == pytest.approx(float(ohlc.loc[ts2, "open"]))

    handler.set_phase("close", ts2)
    visible_close = handler.get_latest_bars("AAA", 10)
    assert visible_close.index.max() == ts2
    assert len(visible_close) == 2


def test_timestamp_at_offset(ohlc):
    handler = HistoricCSVDataHandler({"AAA": ohlc})
    idx = ohlc.index
    assert handler.timestamp_at_offset(idx[0], 1) == idx[1]
    assert handler.timestamp_at_offset(idx[-1], 1) is None


def _ticks(n=2_000, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    volumes = rng.lognormal(8, 0.5, n)
    return pd.DataFrame({"price": prices, "volume": volumes}, index=idx)


def test_dollar_bars_threshold():
    ticks = _ticks()
    threshold = 1e7
    bars = dollar_bars(ticks, threshold)
    assert len(bars) > 0
    assert bars.index.is_monotonic_increasing
    assert (bars["high"] >= bars["low"]).all()
    # Every bar except possibly the last crosses the dollar threshold.
    dollars = ticks["price"] * ticks["volume"]
    cumulative = 0.0
    crossings = 0
    for v in dollars:
        cumulative += v
        if cumulative >= threshold:
            crossings += 1
            cumulative = 0.0
    assert len(bars) in (crossings, crossings + 1)


def test_tick_imbalance_bars_structure():
    ticks = _ticks()
    bars = tick_imbalance_bars(ticks, span=5, warmup=2, initial_expected_len=20)
    assert len(bars) > 1
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-12).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-12).all()
    assert bars["close"].iloc[-1] == pytest.approx(ticks["price"].iloc[-1])


def test_tick_imbalance_bars_bar_frequency_ewma():
    """AFML ch.2 estimator: E_0[T] and the expected imbalance are BOTH
    bar-frequency EWMAs (one observation per completed bar). The pre-fix code
    EWMA'd concatenated per-tick flows with the same span, remembering ~span
    ticks instead of ~span bars (3rd-cross-ref synthesis §1.6, research 02).

    Hand-computed trace (span=2, warmup=1, initial_expected_len=2), with
    b = [+1, +1, -1, +1, +1, +1, +1, +1, +1, +1]:
    - bars 1: cap 2 -> [t0,t1], len 2, mean imb 1.0
      -> threshold = EWMA([2]) * |EWMA([1.0])| = 2
    - bar 2: [t2..t5] (theta dips to -1 then climbs to +2), len 4, imb 0.5
      -> threshold = EWMA([2,4]) * |EWMA([1.0,0.5])| = 3.5 * 0.625 = 2.1875
    - bar 3: [t6..t8], theta reaches 3 >= 2.1875, len 3
    - trailing tick t9 never reaches the next threshold -> flushed (len 1)
    The tick-unit EWMA (pre-fix) would emit bar 3 at length 4 instead of 3.
    """
    signs = [1, 1, -1, 1, 1, 1, 1, 1, 1, 1]
    prices = [100.0]
    for s in signs[1:]:
        prices.append(prices[-1] + 0.5 * s)
    idx = pd.date_range("2024-01-01", periods=len(signs), freq="1min", tz="UTC")
    ticks = pd.DataFrame({"price": prices, "volume": 1.0}, index=idx)

    bars = tick_imbalance_bars(ticks, span=2, warmup=1, initial_expected_len=2.0)
    assert list(bars["volume"]) == [2.0, 4.0, 3.0, 1.0]
    # Every emitted bar's close is the last tick inside it.
    assert bars["close"].iloc[-1] == pytest.approx(prices[-1])


def test_synthetic_ohlcv_ito_drift_correction():
    """GBM fixture must grow at rate mu in PRICE expectation (synthesis §1.11):
    log-increments average (mu - 0.5*sigma**2)/periods_per_year, so that
    E[S_T] = s0 * exp(mu * T). The pre-fix drift (mu/ppy) inflates E[S_T] by
    exp(0.5*sigma**2 * T)."""
    mu, sigma, ppy = 0.10, 0.50, 252.0
    df = make_synthetic_ohlcv(n_bars=200_000, mu=mu, sigma=sigma, seed=11)
    log_rets = np.diff(np.log(df["close"].to_numpy()))
    expected = (mu - 0.5 * sigma**2) / ppy
    se = sigma / np.sqrt(ppy * len(log_rets))  # standard error of the mean
    assert abs(log_rets.mean() - expected) < 4 * se
    # The pre-fix (no Itô term) drift sits far outside that band.
    assert abs(log_rets.mean() - mu / ppy) > 4 * se


def test_synthetic_ohlcv_periods_per_year_parameter():
    """periods_per_year scales the GBM discretization explicitly (§1.2)."""
    sigma = 0.40
    ppy = 365.0 * 24  # hourly 24/7 calendar
    df = make_synthetic_ohlcv(n_bars=100_000, mu=0.0, sigma=sigma,
                              periods_per_year=ppy, seed=5)
    log_rets = np.diff(np.log(df["close"].to_numpy()))
    expected_std = sigma / np.sqrt(ppy)
    se = expected_std / np.sqrt(2.0 * len(log_rets))  # se of the sample std
    assert abs(log_rets.std(ddof=1) - expected_std) < 5 * se


def _trick_inputs(rebalance=True):
    idx = pd.bdate_range("2024-01-01", periods=6, tz="UTC")
    close = pd.DataFrame({"F1": [100, 101, 102, 103, 104, 105],
                          "F2": [50, 50.5, 51, 51.5, 52, 52.5]}, index=idx, dtype=float)
    open_ = close.shift(1).bfill()
    w = pd.DataFrame({"F1": 1.0, "F2": -1.0}, index=idx)  # long/short spread
    B = [idx[0], idx[3]] if rebalance else [idx[0]]
    return idx, open_, close, w, B


def test_etf_trick_costs_external_to_index():
    idx, open_, close, w, B = _trick_inputs()
    base = ETFTrick(w, open_, close, B, cost_rates=0.0).compute()
    taxed = ETFTrick(w, open_, close, B, cost_rates=0.01).compute()

    # K must be identical regardless of cost rates: c_t is never embedded.
    assert np.allclose(base["K"], taxed["K"])
    # Rebalance costs are computed and strictly non-negative (never income).
    assert (taxed["c"] >= 0).all()
    assert taxed["c"].iloc[3] > 0
    assert taxed["c"].sum() > 0


def test_etf_trick_short_spread_no_fictitious_profit():
    idx, open_, close, w_long, B = _trick_inputs()
    w_short = -w_long
    tau = 0.01
    long_run = ETFTrick(w_long, open_, close, B, cost_rates=tau).compute()
    short_run = ETFTrick(w_short, open_, close, B, cost_rates=tau).compute()

    # K_0 = 1 and negated weights negate the pnl stream: K_short - 1 = -(K_long - 1)
    assert np.allclose(short_run["K"] - 1.0, -(long_run["K"] - 1.0))
    # Costs are identical under shorting (|h|), so they can never become income.
    assert np.allclose(long_run["c"], short_run["c"])
    net_short = (short_run["K"] - 1.0) - short_run["c"].cumsum()
    gross_short = short_run["K"] - 1.0
    assert (net_short <= gross_short + 1e-12).all()
