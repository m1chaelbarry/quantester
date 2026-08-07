"""Scrollable, interactive matplotlib viewer for bars, indicators, and results.

Pure matplotlib — no extra dependencies. Interactivity comes from canvas
event hooks (`scroll_event`, `button_press_event`, `motion_notify_event`,
`key_press_event`), so it works on any interactive backend (Qt/Tk/macOS/
notebook). On headless backends the viewer still renders: use ``save()`` for a
static snapshot and drive the view programmatically via ``zoom()``/``pan()``.

Controls
--------
mouse wheel            zoom the time axis around the cursor
left-drag              pan
left / right arrows    pan one screen step (hold shift for a fine step)
up / down or +/-       zoom in / out around the view center
home / end             jump to the first / last window
r                      reset to the full range
hover                  crosshair with date/OHLCV/indicator readout

This is post-hoc research tooling: it consumes bars DataFrames and stored
portfolio artifacts, never the live DataHandler stream.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from .static import INTERACTIVE_BACKENDS

if matplotlib.get_backend().lower() not in INTERACTIVE_BACKENDS:
    matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import PathCollection, PolyCollection

from .static import (
    DOWN_COLOR,
    GRID,
    UP_COLOR,
    _plot_fills,
    _plot_series_dict,
    _plot_trades,
    _style_date_axis,
    draw_candles,
)

MIN_WINDOW = 8  # bars


class InteractiveChartViewer:
    """Multi-panel scrollable chart; see module docstring for controls."""

    def __init__(self, bars: pd.DataFrame, overlays: dict | None = None,
                 subpanels: dict | None = None,
                 equity: pd.Series | None = None,
                 positions: pd.Series | None = None,
                 trades: list | None = None, fills: list | None = None,
                 volume: bool = True,
                 title: str = "Quantester interactive view",
                 figsize=(13, 9)):
        bars = bars.sort_index()
        self.bars = bars
        self.index = pd.DatetimeIndex(bars.index)
        self.n_bars = len(bars)
        if self.n_bars < 2:
            raise ValueError("need at least 2 bars for an interactive view")
        self._full_xlim = (-0.5, self.n_bars - 0.5)

        overlays = overlays or {}
        subpanels = subpanels or {}
        n_panels = (1 + int(volume) + len(subpanels)
                    + int(positions is not None) + int(equity is not None))
        ratios = ([4.0] + [0.9] * int(volume) + [1.4] * len(subpanels)
                  + [0.9] * int(positions is not None)
                  + [1.6] * int(equity is not None))

        self.fig, axes = plt.subplots(
            n_panels, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": ratios, "hspace": 0.05},
        )
        self.axes = list(np.atleast_1d(axes))
        self.ax_price = self.axes[0]

        draw_candles(self.ax_price, bars)
        _plot_series_dict(self.ax_price, overlays, self.index)
        self._overlay_names = list(overlays.keys())
        if trades:
            _plot_trades(self.ax_price, trades, self.index)
        if fills:
            _plot_fills(self.ax_price, fills, self.index)
        if trades or fills or overlays:
            self.ax_price.legend(loc="upper left", fontsize=8, framealpha=0.6)
        self.ax_price.set_ylabel("Price")

        panel = 1
        if volume:
            ax_vol = self.axes[panel]
            x = np.arange(self.n_bars)
            up = bars["close"].to_numpy() >= bars["open"].to_numpy()
            ax_vol.bar(x, bars["volume"].to_numpy(), width=0.7,
                       color=np.where(up, UP_COLOR, DOWN_COLOR), alpha=0.7)
            ax_vol.set_ylabel("Volume")
            panel += 1

        for name, data in subpanels.items():
            ax_sub = self.axes[panel]
            _plot_series_dict(ax_sub, {name: data}, self.index)
            if isinstance(data, pd.Series):
                ax_sub.set_ylabel(name, fontsize=8)
            panel += 1

        if positions is not None:
            ax_pos = self.axes[panel]
            aligned = positions.reindex(self.index).ffill().fillna(0.0)
            x = np.arange(self.n_bars)
            values = aligned.to_numpy()
            ax_pos.fill_between(x, values, 0, step="mid", where=values >= 0,
                                color=UP_COLOR, alpha=0.6)
            ax_pos.fill_between(x, values, 0, step="mid", where=values < 0,
                                color=DOWN_COLOR, alpha=0.6)
            ax_pos.axhline(0, color="black", lw=0.6)
            ax_pos.set_ylabel("Position")
            panel += 1

        if equity is not None:
            ax_eq = self.axes[panel]
            aligned = equity.reindex(self.index).ffill()
            ax_eq.plot(np.arange(self.n_bars), aligned.to_numpy(), lw=1.1,
                       color="#08306b")
            ax_eq.set_ylabel("Equity")
            panel += 1

        for ax in self.axes:
            _style_date_axis(ax, self.index)
        self.ax_price.set_title(title)

        # Crosshair artists: one vertical line per panel + a readout box.
        self._vlines = [
            ax.axvline(np.nan, color="grey", lw=0.7, ls="--", alpha=0.7)
            for ax in self.axes
        ]
        self._readout = self.ax_price.text(
            0.995, 0.98, "", transform=self.ax_price.transAxes, ha="right",
            va="top", family="monospace", fontsize=8,
            bbox={"boxstyle": "round", "alpha": 0.2})

        self._drag_start = None  # (x, xlim) at button press

        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ------------------------------------------------------------ view state

    @property
    def view_window(self) -> tuple[float, float]:
        return self.ax_price.get_xlim()

    def zoom(self, factor: float, center: float | None = None) -> None:
        """Multiply the visible window width by ``factor`` (<1 zooms in)."""
        lo, hi = self.view_window
        center = (lo + hi) / 2 if center is None else float(center)
        width = (hi - lo) * factor
        full = self._full_xlim[1] - self._full_xlim[0]
        width = min(max(width, MIN_WINDOW), full)
        self._set_window(center - width / 2, center + width / 2)

    def pan(self, fraction: float) -> None:
        """Shift the window by ``fraction`` of its width (positive = right)."""
        lo, hi = self.view_window
        shift = (hi - lo) * fraction
        self._set_window(lo + shift, hi + shift)

    def reset_view(self) -> None:
        self._set_window(*self._full_xlim)

    def _set_window(self, lo: float, hi: float) -> None:
        full_lo, full_hi = self._full_xlim
        width = hi - lo
        if lo < full_lo:
            lo, hi = full_lo, full_lo + width
        if hi > full_hi:
            lo, hi = full_hi - width, full_hi
        self.ax_price.set_xlim(lo, hi)  # shared x: all panels follow
        self._autoscale_y()
        self.fig.canvas.draw_idle()

    def _autoscale_y(self) -> None:
        """Fit each panel's y-range to the visible slice (5% margin)."""
        lo, hi = self.view_window
        i0 = max(int(np.floor(lo)), 0)
        i1 = min(int(np.ceil(hi)) + 1, self.n_bars)
        for ax in self.axes:
            ymin, ymax = np.inf, -np.inf
            if ax is self.ax_price:
                ymin = min(ymin, float(self.bars["low"].to_numpy()[i0:i1].min()))
                ymax = max(ymax, float(self.bars["high"].to_numpy()[i0:i1].max()))
            for line in ax.lines:
                ydata = np.asarray(line.get_ydata(), dtype=float)
                if len(ydata) != self.n_bars:
                    continue  # e.g. axhline / crosshair artists
                visible = ydata[i0:i1]
                visible = visible[np.isfinite(visible)]
                if len(visible):
                    ymin = min(ymin, float(visible.min()))
                    ymax = max(ymax, float(visible.max()))
            for coll in ax.collections:
                if isinstance(coll, PolyCollection):
                    # fill_between shading (position panel); verts in data coords.
                    for p in coll.get_paths():
                        verts = p.vertices
                        mask = (verts[:, 0] >= lo) & (verts[:, 0] <= hi)
                        if mask.any():
                            ymin = min(ymin, float(verts[mask, 1].min()))
                            ymax = max(ymax, float(verts[mask, 1].max()))
                elif isinstance(coll, PathCollection):
                    # scatter markers (fills/trades); offsets hold data coords.
                    offsets = coll.get_offsets()
                    if len(offsets):
                        mask = (offsets[:, 0] >= lo) & (offsets[:, 0] <= hi)
                        if mask.any():
                            ymin = min(ymin, float(offsets[mask, 1].min()))
                            ymax = max(ymax, float(offsets[mask, 1].max()))
            if not np.isfinite(ymin) or not np.isfinite(ymax):
                continue
            margin = (ymax - ymin) * 0.05 or abs(ymax) * 0.05 or 1.0
            ax.set_ylim(ymin - margin, ymax + margin)

    # ----------------------------------------------------------- event hooks

    def _on_scroll(self, event) -> None:
        if event.inaxes not in self.axes or event.xdata is None:
            return
        # step > 0 scrolls up = zoom in.
        self.zoom(0.8 if event.step > 0 else 1.25, center=event.xdata)

    def _on_press(self, event) -> None:
        if event.inaxes in self.axes and event.button == 1 \
                and event.xdata is not None:
            self._drag_start = (event.xdata, self.view_window)

    def _on_release(self, _event) -> None:
        self._drag_start = None

    def _on_motion(self, event) -> None:
        if self._drag_start is not None and event.xdata is not None:
            start_x, (lo, hi) = self._drag_start
            shift = start_x - event.xdata
            self._set_window(lo + shift, hi + shift)
            return
        self._update_crosshair(event)

    def _on_key(self, event) -> None:
        key = event.key or ""
        fine = "shift" in key
        step = 0.05 if fine else 0.25
        if key.endswith("right"):
            self.pan(step)
        elif key.endswith("left"):
            self.pan(-step)
        elif key in ("up", "+", "="):
            self.zoom(0.8)
        elif key in ("down", "-"):
            self.zoom(1.25)
        elif key == "home":
            lo = self._full_xlim[0]
            self._set_window(lo, lo + (self.view_window[1] - self.view_window[0]))
        elif key == "end":
            hi = self._full_xlim[1]
            self._set_window(hi - (self.view_window[1] - self.view_window[0]), hi)
        elif key == "r":
            self.reset_view()

    def _update_crosshair(self, event) -> None:
        if event.inaxes not in self.axes or event.xdata is None:
            for vline in self._vlines:
                vline.set_xdata([np.nan, np.nan])
            self._readout.set_text("")
            self.fig.canvas.draw_idle()
            return
        i = int(round(event.xdata))
        if not 0 <= i < self.n_bars:
            return
        for vline in self._vlines:
            vline.set_xdata([i, i])
        bar = self.bars.iloc[i]
        date = pd.Timestamp(self.index[i]).strftime("%Y-%m-%d")
        self._readout.set_text(
            f"{date}\n"
            f"O {bar['open']:.2f}  H {bar['high']:.2f}\n"
            f"L {bar['low']:.2f}  C {bar['close']:.2f}\n"
            f"V {bar['volume']:.0f}"
        )
        self.fig.canvas.draw_idle()

    # ---------------------------------------------------------------- output

    def show(self) -> bool:
        """Block on the interactive window; False when headless (Agg)."""
        if matplotlib.get_backend().lower() not in INTERACTIVE_BACKENDS:
            print("Headless backend (Agg): interactivity unavailable. "
                  "Use viewer.save(path) for a snapshot, or run with an "
                  "interactive backend (e.g. `matplotlib.use('QtAgg')` or a "
                  "notebook with `%matplotlib widget`).")
            return False
        plt.show()
        return True

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=110)
        return path


def interactive_view(bars: pd.DataFrame, overlays: dict | None = None,
                     subpanels: dict | None = None,
                     equity: pd.Series | None = None,
                     positions: pd.Series | None = None,
                     trades: list | None = None, fills: list | None = None,
                     volume: bool = True,
                     title: str = "Quantester interactive view",
                     ) -> InteractiveChartViewer:
    """Build a scrollable chart over bars + indicators + backtest artifacts.

    Example
    -------
    >>> from quantester.visualization import indicators, interactive_view
    >>> viewer = interactive_view(
    ...     bars,
    ...     overlays={"SMA(10)": indicators.sma(bars["close"], 10)},
    ...     subpanels={"RSI(14)": indicators.rsi(bars["close"])},
    ...     equity=portfolio.equity_curve,
    ...     trades=portfolio.trades, fills=portfolio.fills,
    ... )
    >>> viewer.show()          # interactive backend: scroll/drag/keys
    >>> viewer.save("view.png")  # headless: static snapshot
    """
    return InteractiveChartViewer(
        bars, overlays=overlays, subpanels=subpanels, equity=equity,
        positions=positions, trades=trades, fills=fills, volume=volume,
        title=title,
    )
