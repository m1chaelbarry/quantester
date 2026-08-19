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


def test_synthetic_ohlcv_log_drift_uses_ito_correction():
    """Hilpisch GBM: E[Δlog S] = (μ − ½σ²)Δt, not μΔt (synthesis §1.11)."""
    mu, sigma, n_bars, s0 = 0.20, 0.40, 80_000, 100.0
    frame = make_synthetic_ohlcv(
        "AAA", n_bars=n_bars, s0=s0, mu=mu, sigma=sigma, seed=1,
    )
    log_levels = np.log(frame["close"].to_numpy() / s0)
    daily = np.concatenate([[log_levels[0]], np.diff(log_levels)])
    mean_daily = float(daily.mean())
    ito_daily = (mu - 0.5 * sigma ** 2) / 252
    naive_daily = mu / 252
    assert abs(mean_daily - ito_daily) < abs(mean_daily - naive_daily)
    assert abs(mean_daily - ito_daily) < 0.0002
