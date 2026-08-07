"""Static matplotlib charts for backtest review and strategy development.

All functions render with whatever backend is active (Agg-safe for headless
runs), return the ``Figure``, and save to ``path`` when given. Charts consume
post-run artifacts only — bars DataFrames, ``portfolio.equity_curve``,
``portfolio.positions_history``, ``portfolio.fills`` and ``portfolio.trades`` —
so visualization never touches the temporal firewall or the event loop.

Conventions:
- x-axes use integer bar positions with date tick labels, so business-day
  calendars render without weekend gaps and zoom math stays trivial.
- ``overlays`` map name -> Series/DataFrame drawn on the price panel;
  ``subpanels`` map name -> Series/DataFrame, one panel each (use a DataFrame
  for multi-line indicators such as MACD or Bollinger Bands).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

INTERACTIVE_BACKENDS = {
    "qtagg", "qt5agg", "qt6agg", "tkagg", "macosx", "gtk3agg", "gtk4agg",
    "webagg", "nbagg",
}

if matplotlib.get_backend().lower() not in INTERACTIVE_BACKENDS:
    matplotlib.use("Agg")
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter, MaxNLocator

UP_COLOR = "#2e8b57"
DOWN_COLOR = "#c0392b"
GRID = {"alpha": 0.3}


def _date_formatter(index: pd.DatetimeIndex) -> FuncFormatter:
    """Map integer bar positions back to date labels."""
    labels = index

    def _fmt(x, _pos):
        i = int(round(x))
        if 0 <= i < len(labels):
            return pd.Timestamp(labels[i]).strftime("%Y-%m-%d")
        return ""

    return FuncFormatter(_fmt)


def _style_date_axis(ax, index: pd.DatetimeIndex) -> None:
    ax.xaxis.set_major_formatter(_date_formatter(index))
    ax.xaxis.set_major_locator(MaxNLocator(8, integer=True))
    ax.set_xlim(-0.5, len(index) - 0.5)
    ax.grid(**GRID)


def _positions_of(index: pd.DatetimeIndex, timestamps) -> np.ndarray:
    """Integer bar positions for arbitrary timestamps (-1 when off-calendar)."""
    return index.get_indexer(pd.DatetimeIndex(pd.to_datetime(list(timestamps))))


def draw_candles(ax, bars: pd.DataFrame, width: float = 0.7) -> None:
    """OHLC candlesticks on integer x positions."""
    x = np.arange(len(bars))
    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    up = c >= o
    colors = np.where(up, UP_COLOR, DOWN_COLOR)

    wick_segments = [[(xi, lo), (xi, hi)] for xi, lo, hi in zip(x, low, h)]
    ax.add_collection(LineCollection(wick_segments, colors=colors, linewidths=0.8))

    body_bottom = np.minimum(o, c)
    body_height = np.abs(c - o)
    # Doji bars get a one-tick-tall body so they stay visible.
    tick = max(np.nanmedian(np.diff(np.sort(np.unique(c)))) if len(c) > 1 else 0.0,
               1e-12)
    body_height = np.maximum(body_height, tick * 0.05)
    ax.bar(x, body_height, bottom=body_bottom, width=width, color=colors,
           edgecolor=colors, linewidth=0.5, zorder=3)


def _plot_series_dict(ax, data: dict, index: pd.DatetimeIndex) -> None:
    """Plot name -> Series/DataFrame entries on one axis over bar positions."""
    x = np.arange(len(index))
    for name, series_or_df in data.items():
        aligned = series_or_df.reindex(index)
        if isinstance(aligned, pd.DataFrame):
            for col in aligned.columns:
                label = f"{name}.{col}" if len(aligned.columns) > 1 else name
                ax.plot(x, aligned[col].to_numpy(), lw=1.0, label=str(label))
        else:
            ax.plot(x, aligned.to_numpy(), lw=1.0, label=str(name))
    if data:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.6)


def _plot_fills(ax, fills, index: pd.DatetimeIndex) -> None:
    buys = [(t, p) for t, p in
            ((f.timestamp, f.fill_price) for f in fills if f.direction == "BUY")]
    sells = [(t, p) for t, p in
             ((f.timestamp, f.fill_price) for f in fills if f.direction == "SELL")]
    for points, marker, color, label in (
        (buys, "^", UP_COLOR, "BUY fill"),
        (sells, "v", DOWN_COLOR, "SELL fill"),
    ):
        if not points:
            continue
        ts, prices = zip(*points)
        xs = _positions_of(index, ts)
        keep = xs >= 0
        ax.scatter(xs[keep], np.asarray(prices)[keep], marker=marker, s=45,
                   color=color, edgecolor="black", linewidth=0.4, zorder=5,
                   label=label)


def _plot_trades(ax, trades: list, index: pd.DatetimeIndex) -> None:
    """Round-trip markers: entry/exit dots joined by a pnl-colored segment."""
    for trade in trades:
        xs = _positions_of(index, [trade["t0"], trade["t1"]])
        if (xs < 0).any():
            continue
        color = UP_COLOR if trade["pnl"] >= 0 else DOWN_COLOR
        ax.plot(xs, [trade["entry_price"], trade["exit_price"]], color=color,
                lw=1.2, alpha=0.8, zorder=4)
        ax.scatter(xs, [trade["entry_price"], trade["exit_price"]], s=22,
                   color=color, marker="o", zorder=5, alpha=0.9)


def plot_candles(bars: pd.DataFrame, overlays: dict | None = None,
                 subpanels: dict | None = None, trades: list | None = None,
                 fills: list | None = None, positions: pd.Series | None = None,
                 volume: bool = True, title: str | None = None, path=None,
                 figsize=(13, 9)):
    """Candlestick chart with indicator overlays, subpanels, and trade markers.

    ``positions`` (target or held quantity per timestamp) gets a thin step
    panel — handy for debugging a strategy's intended exposure vs the tape.
    """
    bars = bars.sort_index()
    index = pd.DatetimeIndex(bars.index)
    overlays = overlays or {}
    subpanels = subpanels or {}

    n_panels = 1 + int(volume) + len(subpanels) + int(positions is not None)
    ratios = [4.0] + [0.9] * int(volume) + [1.4] * len(subpanels) \
        + [0.9] * int(positions is not None)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": ratios, "hspace": 0.05},
    )
    axes = np.atleast_1d(axes)
    ax_price = axes[0]

    draw_candles(ax_price, bars)
    _plot_series_dict(ax_price, overlays, index)
    if trades:
        _plot_trades(ax_price, trades, index)
    if fills:
        _plot_fills(ax_price, fills, index)
    if trades or fills:
        ax_price.legend(loc="upper left", fontsize=8, framealpha=0.6)
    ax_price.set_ylabel("Price")

    panel = 1
    if volume:
        ax_vol = axes[panel]
        x = np.arange(len(index))
        up = bars["close"].to_numpy() >= bars["open"].to_numpy()
        ax_vol.bar(x, bars["volume"].to_numpy(), width=0.7,
                   color=np.where(up, UP_COLOR, DOWN_COLOR), alpha=0.7)
        ax_vol.set_ylabel("Volume")
        panel += 1

    for name, data in subpanels.items():
        ax_sub = axes[panel]
        _plot_series_dict(ax_sub, {name: data}, index)
        if isinstance(data, pd.Series):
            ax_sub.set_ylabel(name, fontsize=8)
        panel += 1

    if positions is not None:
        ax_pos = axes[panel]
        aligned = positions.reindex(index).ffill().fillna(0.0)
        x = np.arange(len(index))
        ax_pos.fill_between(x, aligned.to_numpy(), 0, step="mid",
                            where=aligned.to_numpy() >= 0, color=UP_COLOR,
                            alpha=0.6)
        ax_pos.fill_between(x, aligned.to_numpy(), 0, step="mid",
                            where=aligned.to_numpy() < 0, color=DOWN_COLOR,
                            alpha=0.6)
        ax_pos.axhline(0, color="black", lw=0.6)
        ax_pos.set_ylabel("Position")
        panel += 1

    for ax in axes:
        _style_date_axis(ax, index)
    if title:
        ax_price.set_title(title)
    fig.autofmt_xdate()
    if path is not None:
        _save(fig, path)
    return fig


def plot_equity(equity: pd.Series, positions_history: pd.DataFrame | None = None,
                log_scale: bool = False, title: str = "Equity & exposure",
                path=None, figsize=(12, 7)):
    """Equity curve with drawdown panel and optional per-symbol positions."""
    from ..analytics import performance

    n_panels = 2 + int(positions_history is not None and not positions_history.empty)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4, 1.4][:n_panels], "hspace": 0.08},
    )
    axes = np.atleast_1d(axes)

    axes[0].plot(equity.index, equity.to_numpy(), lw=1.6, color="#1f77b4")
    axes[0].set_ylabel("Equity")
    if log_scale:
        axes[0].set_yscale("log")

    dd = performance.drawdown_series(equity)
    axes[1].fill_between(dd.index, dd.to_numpy(), 0, color=DOWN_COLOR, alpha=0.55)
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(
        FuncFormatter(lambda v, _p: f"{v:.0%}"))

    if n_panels == 3:
        # Use Axes.plot (not DataFrame.plot) so datetime units stay consistent
        # with the equity/drawdown artists under sharex — pandas.plot can
        # retarget the shared x converter and leave the equity line off-screen.
        for col in positions_history.columns:
            series = positions_history[col].dropna()
            if series.empty:
                continue
            axes[2].plot(series.index, series.to_numpy(), lw=0.9, label=str(col))
        axes[2].axhline(0, color="black", lw=0.6)
        axes[2].set_ylabel("Qty held")
        axes[2].legend(loc="upper left", fontsize=8, framealpha=0.6)

    for ax in axes:
        ax.grid(**GRID)
    axes[0].set_title(title)
    fig.autofmt_xdate()
    if path is not None:
        _save(fig, path)
    return fig


def trade_stats(trades: list) -> dict:
    """Summary statistics over round-trip trade dicts (portfolio.trades)."""
    if not trades:
        return {"n_trades": 0}
    pnl = np.array([t["pnl"] for t in trades], dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    return {
        "n_trades": len(pnl),
        "win_rate": float((pnl > 0).mean()),
        "total_pnl": float(pnl.sum()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else np.inf,
        "expectancy": float(pnl.mean()),
    }


def plot_trade_analysis(trades: list, title: str = "Trade analysis", path=None,
                        figsize=(12, 8)):
    """Per-trade PnL, cumulative PnL, PnL histogram, and summary stats."""
    if not trades:
        raise ValueError("no round-trip trades to analyze")
    pnl = np.array([t["pnl"] for t in trades], dtype=float)
    t1 = pd.DatetimeIndex(pd.to_datetime([t["t1"] for t in trades]))
    colors = np.where(pnl >= 0, UP_COLOR, DOWN_COLOR)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title)

    axes[0, 0].bar(np.arange(len(pnl)), pnl, color=colors)
    axes[0, 0].axhline(0, color="black", lw=0.6)
    axes[0, 0].set_title("PnL per round trip")
    axes[0, 0].set_xlabel("Trade #")

    axes[0, 1].plot(t1, np.cumsum(pnl), lw=1.2)
    axes[0, 1].set_title("Cumulative realized PnL")
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    bins = np.histogram_bin_edges(pnl, bins="fd")
    axes[1, 0].hist(wins, bins=bins, color=UP_COLOR, alpha=0.75, label="wins")
    axes[1, 0].hist(losses, bins=bins, color=DOWN_COLOR, alpha=0.75,
                    label="losses")
    axes[1, 0].axvline(0, color="black", lw=0.6)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title("PnL distribution")

    stats = trade_stats(trades)
    pf = stats["profit_factor"]
    text = "\n".join([
        f"trades:        {stats['n_trades']:>8d}",
        f"win rate:      {stats['win_rate']:>8.1%}",
        f"total PnL:     {stats['total_pnl']:>8.2f}",
        f"avg win:       {stats['avg_win']:>8.2f}",
        f"avg loss:      {stats['avg_loss']:>8.2f}",
        f"profit factor: {pf:>8.2f}" if np.isfinite(pf) else
        f"profit factor: {'inf':>8}",
        f"expectancy:    {stats['expectancy']:>8.2f}",
    ])
    axes[1, 1].axis("off")
    axes[1, 1].text(0.05, 0.95, text, transform=axes[1, 1].transAxes, va="top",
                    family="monospace", fontsize=10,
                    bbox={"boxstyle": "round", "alpha": 0.12})
    axes[1, 1].set_title("Summary")

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.grid(**GRID)
    fig.tight_layout()
    if path is not None:
        _save(fig, path)
    return fig


def plot_monthly_returns(equity: pd.Series, title: str = "Monthly returns (%)",
                         path=None, figsize=(11, 4)):
    """Year x month heatmap of percentage returns."""
    rets = equity.resample("ME").last().pct_change()
    monthly = rets.dropna().to_frame("ret")
    monthly["year"] = monthly.index.year
    monthly["month"] = monthly.index.month
    pivot = monthly.pivot(index="year", columns="month", values="ret") * 100.0
    pivot = pivot.reindex(columns=range(1, 13))

    fig, ax = plt.subplots(figsize=figsize)
    values = pivot.to_numpy(dtype=float)
    limit = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    im = ax.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit,
                   aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                        "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center",
                        fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    if path is not None:
        _save(fig, path)
    return fig


def plot_rolling_metrics(equity: pd.Series, window: int = 63,
                         periods: int = 252, path=None, figsize=(12, 8),
                         title: str | None = None):
    """Rolling annualized Sharpe, rolling volatility, and drawdown."""
    from ..analytics import performance

    rets = equity.pct_change()
    roll_sharpe = (rets.rolling(window).mean() / rets.rolling(window).std(ddof=1)
                   * np.sqrt(periods))
    roll_vol = rets.rolling(window).std(ddof=1) * np.sqrt(periods)
    dd = performance.drawdown_series(equity)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                             gridspec_kw={"hspace": 0.08})
    axes[0].plot(equity.index, roll_sharpe.to_numpy(), lw=1.0)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_ylabel(f"Sharpe ({window}b)")

    axes[1].plot(equity.index, roll_vol.to_numpy(), lw=1.0, color="#8e44ad")
    axes[1].set_ylabel(f"Vol ({window}b, ann.)")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0%}"))

    axes[2].fill_between(dd.index, dd.to_numpy(), 0, color=DOWN_COLOR,
                         alpha=0.5)
    axes[2].set_ylabel("Drawdown")
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0%}"))

    for ax in axes:
        ax.grid(**GRID)
    axes[0].set_title(title or f"Rolling metrics (window={window} bars)")
    fig.autofmt_xdate()
    if path is not None:
        _save(fig, path)
    return fig


def plot_path_distribution(paths, percentiles=(5, 25, 50, 75, 95),
                           n_spaghetti: int = 40,
                           title: str = "Monte Carlo path distribution",
                           path=None, figsize=(12, 6)):
    """Fan chart + terminal-value histogram for Monte Carlo ensembles.

    ``paths`` accepts an (n_paths, n_steps) ndarray — as returned by
    ``montecarlo.synthetic.generate_ou_paths``, ``trade_resampling`` results
    (``.paths``), and ``fast_track`` ensembles — or a list of pd.Series.
    """
    if isinstance(paths, (list, tuple)):
        paths = np.column_stack(
            [pd.Series(p).to_numpy(dtype=float) for p in paths]).T
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[0] < 1:
        raise ValueError("paths must be a 2D array (n_paths, n_steps)")

    x = np.arange(paths.shape[1])
    qs = np.percentile(paths, percentiles, axis=0)
    terminal = paths[:, -1]

    fig, (ax_fan, ax_term) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 1]})

    mid = len(percentiles) // 2
    for k in range(mid):
        lo, hi = qs[k], qs[-(k + 1)]
        ax_fan.fill_between(x, lo, hi, alpha=0.25 + 0.15 * k, color="#2c7fb0",
                            lw=0,
                            label=f"P{percentiles[k]}–P{percentiles[-(k + 1)]}")
    ax_fan.plot(x, qs[mid], color="#08306b", lw=1.6,
                label=f"median (P{percentiles[mid]})")

    rng = np.random.default_rng(0)
    sample = rng.choice(paths.shape[0], size=min(n_spaghetti, paths.shape[0]),
                        replace=False)
    ax_fan.plot(x, paths[sample].T, color="grey", alpha=0.15, lw=0.5)
    ax_fan.set_xlabel("Step")
    ax_fan.set_ylabel("Value")
    ax_fan.set_title(title)
    ax_fan.legend(fontsize=8, loc="upper left")
    ax_fan.grid(**GRID)

    ax_term.hist(terminal, bins=40, orientation="horizontal", color="#2c7fb0",
                 alpha=0.7)
    for p, q in zip(percentiles, qs):
        ax_term.axhline(q[-1], ls="--", lw=0.8, alpha=0.6)
        ax_term.annotate(f"P{p}", xy=(0.02, q[-1]), xycoords=("axes fraction", "data"),
                         fontsize=7, va="bottom")
    ax_term.set_title("Terminal values")
    ax_term.grid(**GRID)

    fig.tight_layout()
    if path is not None:
        _save(fig, path)
    return fig


def _save(fig, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
