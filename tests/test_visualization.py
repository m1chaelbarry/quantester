"""Visualization: indicators, static charts, interactive viewer mechanics."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.execution.costs import CostModel
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.montecarlo.synthetic import estimate_ou_params, generate_ou_paths
from quantester.portfolio.portfolio import FixedUnitSizer, PortfolioManager
from quantester.strategy.examples import MovingAverageCrossStrategy
from quantester.visualization import (
    indicators,
    interactive_view,
    plot_candles,
    plot_equity,
    plot_monthly_returns,
    plot_path_distribution,
    plot_rolling_metrics,
    plot_trade_analysis,
    trade_stats,
)
from quantester.visualization.interactive import MIN_WINDOW


def _run_backtest(ohlc):
    """Small MA-cross run producing fills, trades, equity, positions."""
    handler = HistoricCSVDataHandler({"AAA": ohlc})
    strategy = MovingAverageCrossStrategy(handler, "AAA", fast=3, slow=8)
    portfolio = PortfolioManager(handler, 100_000.0, sizer=FixedUnitSizer(100))
    engine = BacktestEngine(handler, strategy, portfolio,
                            SimulatedExecutionHandler(CostModel()))
    engine.run_backtest()
    return portfolio


# ------------------------------------------------------------------ indicators


def test_sma_ema_match_pandas(ohlc):
    close = ohlc["close"]
    pd.testing.assert_series_equal(indicators.sma(close, 10),
                                   close.rolling(10).mean())
    pd.testing.assert_series_equal(indicators.ema(close, 10),
                                   close.ewm(span=10, adjust=False).mean())


def test_rsi_known_limits():
    close = pd.Series(np.arange(1.0, 40.0))  # strictly rising: every gain
    r = indicators.rsi(close, window=14)
    assert r.dropna().iloc[-1] == pytest.approx(100.0)
    falling = pd.Series(np.arange(40.0, 1.0, -1.0))
    assert indicators.rsi(falling, 14).dropna().iloc[-1] == pytest.approx(0.0)
    bounded = indicators.rsi(
        pd.Series(np.sin(np.arange(200.0)) + 2.0), 14).dropna()
    assert ((0.0 <= bounded) & (bounded <= 100.0)).all()


def test_macd_and_bollinger_structure(ohlc):
    close = ohlc["close"]
    m = indicators.macd(close)
    assert list(m.columns) == ["macd", "signal", "histogram"]
    assert m["histogram"].dropna().equals(
        (m["macd"] - m["signal"]).dropna())

    bb = indicators.bollinger_bands(close, window=20, n_std=2.0)
    assert (bb["upper"].dropna() >= bb["mid"].dropna()).all()
    assert (bb["lower"].dropna() <= bb["mid"].dropna()).all()
    width = (bb["upper"] - bb["lower"]).dropna()
    assert width.iloc[-1] == pytest.approx(
        4.0 * close.rolling(20).std(ddof=0).iloc[-1])


def test_atr_non_negative_and_wilder(ohlc):
    a = indicators.atr(ohlc["high"], ohlc["low"], ohlc["close"], window=14)
    assert (a.dropna() >= 0).all()
    tr = pd.concat(
        [ohlc["high"] - ohlc["low"],
         (ohlc["high"] - ohlc["close"].shift(1)).abs(),
         (ohlc["low"] - ohlc["close"].shift(1)).abs()], axis=1).max(axis=1)
    expected = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    pd.testing.assert_series_equal(a, expected)


def test_adx_and_donchian(ohlc):
    a = indicators.adx(ohlc["high"], ohlc["low"], ohlc["close"], window=14)
    assert set(a.columns) == {"adx", "plus_di", "minus_di"}
    assert (a["adx"].dropna() >= 0).all()
    assert (a["adx"].dropna() <= 100).all()
    channel = indicators.donchian(ohlc["high"], ohlc["low"], window=20, shift=1)
    assert channel["upper"].iloc[40] == pytest.approx(
        ohlc["high"].iloc[20:40].max())
    assert channel["lower"].iloc[40] == pytest.approx(
        ohlc["low"].iloc[20:40].min())
    assert (channel["upper"].dropna() >= channel["lower"].dropna()).all()


# ------------------------------------------------------------- static charts


def test_plot_candles_full_stack(ohlc, tmp_path):
    portfolio = _run_backtest(ohlc)
    path = tmp_path / "candles.png"
    fig = plot_candles(
        ohlc,
        overlays={"SMA(10)": indicators.sma(ohlc["close"], 10)},
        subpanels={"RSI(14)": indicators.rsi(ohlc["close"]),
                   "MACD": indicators.macd(ohlc["close"])},
        trades=portfolio.trades,
        fills=portfolio.fills,
        positions=portfolio.positions_history["AAA"],
        title="AAA",
        path=path,
    )
    assert path.exists() and path.stat().st_size > 0
    assert len(fig.axes) == 5  # price + volume + 2 subpanels + positions


def test_plot_equity_with_positions(ohlc, tmp_path):
    portfolio = _run_backtest(ohlc)
    path = tmp_path / "equity.png"
    plot_equity(portfolio.equity_curve,
                positions_history=portfolio.positions_history, path=path)
    assert path.exists() and path.stat().st_size > 0


def test_plot_trade_analysis_and_stats(ohlc, tmp_path):
    portfolio = _run_backtest(ohlc)
    assert len(portfolio.trades) > 0
    stats = trade_stats(portfolio.trades)
    assert stats["n_trades"] == len(portfolio.trades)
    assert 0.0 <= stats["win_rate"] <= 1.0
    path = tmp_path / "trades.png"
    plot_trade_analysis(portfolio.trades, path=path)
    assert path.exists() and path.stat().st_size > 0


def test_trade_records_carry_direction(ohlc):
    portfolio = _run_backtest(ohlc)
    assert {t["direction"] for t in portfolio.trades} <= {-1, 1}


def test_plot_monthly_returns(ohlc, tmp_path):
    portfolio = _run_backtest(ohlc)
    path = tmp_path / "monthly.png"
    plot_monthly_returns(portfolio.equity_curve, path=path)
    assert path.exists() and path.stat().st_size > 0


def test_plot_rolling_metrics(ohlc, tmp_path):
    portfolio = _run_backtest(ohlc)
    path = tmp_path / "rolling.png"
    plot_rolling_metrics(portfolio.equity_curve, window=20, path=path)
    assert path.exists() and path.stat().st_size > 0


def test_plot_path_distribution(ohlc, tmp_path):
    params = estimate_ou_params(ohlc["close"].to_numpy())
    paths = generate_ou_paths(params, p0=float(ohlc["close"].iloc[-1]),
                              n_steps=60, n_paths=200, seed=7)
    path = tmp_path / "mc.png"
    plot_path_distribution(paths, path=path)
    assert path.exists() and path.stat().st_size > 0


# --------------------------------------------------------- interactive viewer


def test_interactive_viewer_zoom_pan_reset(ohlc, tmp_path):
    viewer = interactive_view(
        ohlc,
        overlays={"SMA(10)": indicators.sma(ohlc["close"], 10)},
        subpanels={"RSI(14)": indicators.rsi(ohlc["close"])},
        positions=pd.Series(1.0, index=ohlc.index),
    )
    full_lo, full_hi = viewer.view_window
    assert (full_lo, full_hi) == (-0.5, len(ohlc) - 0.5)

    viewer.zoom(0.5)
    lo, hi = viewer.view_window
    assert hi - lo == pytest.approx((full_hi - full_lo) * 0.5)

    viewer.pan(0.25)
    assert viewer.view_window[0] > lo

    width_before = viewer.view_window[1] - viewer.view_window[0]
    viewer.pan(-10.0)  # clamped at the left edge
    assert viewer.view_window[0] == -0.5
    assert viewer.view_window[1] - viewer.view_window[0] == pytest.approx(
        width_before)

    for _ in range(50):  # zoom-in clamp
        viewer.zoom(0.5)
    assert viewer.view_window[1] - viewer.view_window[0] >= MIN_WINDOW

    viewer.reset_view()
    assert viewer.view_window == (-0.5, len(ohlc) - 0.5)

    snap = viewer.save(tmp_path / "viewer.png")
    assert snap.exists() and snap.stat().st_size > 0


def test_interactive_viewer_scroll_and_key_events(ohlc):
    viewer = interactive_view(ohlc)

    class _Event:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    width_full = viewer.view_window[1] - viewer.view_window[0]
    viewer._on_scroll(_Event(inaxes=viewer.ax_price, xdata=60.0, step=1))
    assert viewer.view_window[1] - viewer.view_window[0] < width_full

    lo_before = viewer.view_window[0]
    viewer._on_key(_Event(key="right"))
    assert viewer.view_window[0] > lo_before
    viewer._on_key(_Event(key="home"))
    assert viewer.view_window[0] == -0.5
    viewer._on_key(_Event(key="end"))
    assert viewer.view_window[1] == len(ohlc) - 0.5
    viewer._on_key(_Event(key="r"))
    assert viewer.view_window[1] - viewer.view_window[0] == pytest.approx(
        width_full)


def test_interactive_viewer_headless_show_returns_false(ohlc, capsys):
    viewer = interactive_view(ohlc)
    assert viewer.show() is False
    assert "Headless" in capsys.readouterr().out


def test_interactive_viewer_drag_pan(ohlc):
    viewer = interactive_view(ohlc)

    class _Event:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    viewer.zoom(0.5)  # room to move; full-range windows clamp at the edges
    lo0, hi0 = viewer.view_window
    viewer._on_press(_Event(inaxes=viewer.ax_price, xdata=50.0, button=1))
    viewer._on_motion(_Event(inaxes=viewer.ax_price, xdata=40.0))
    viewer._on_release(_Event())
    lo1, hi1 = viewer.view_window
    assert lo1 > lo0 and hi1 > hi0  # dragged content left -> window moved right
