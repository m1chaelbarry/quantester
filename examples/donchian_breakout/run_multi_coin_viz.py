"""Multi-coin Donchian portfolio dashboard.

Renders a single composition figure covering:
  1. Normalized equity — portfolio vs BTC-only sleeve vs BTC buy-and-hold
  2. Per-coin strategy equity overlay
  3. Underwater (drawdown) comparison
  4. Strategy-return correlation heatmap
  5. Concurrent names active over time
  6. Spectral risk shares (PC1…PC5)
  7. Calendar-year returns for the book
  8. Per-coin contribution bar (standalone Sharpe / CAGR)

Run from the repo root:
  python examples/donchian_breakout/run_multi_coin_viz.py
  python examples/donchian_breakout/run_multi_coin_viz.py \\
      --universe BTC/USD,ETH/USD,XRP/USD --risk-budget 0.02
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_DIR = REPO_ROOT / "examples" / "data"
OUTPUT_DIR = HERE / "output"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from quantester.analytics.performance import annualized_sharpe, max_drawdown
from quantester.portfolio.risk import spectral_risk_attribution

from run_multi_coin import (
    DEFAULT_UNIVERSE,
    INITIAL_CAPITAL,
    PERIODS,
    load_universe,
    run_bh,
    run_portfolio,
    run_single,
    summarize,
)

# Calm research palette (no purple/glow defaults).
INK = "#1a1f24"
PAPER = "#f7f4ef"
GRID = "#d9d2c5"
ACCENT = "#0f6b5c"       # portfolio
BTC_ONLY = "#c45c26"     # BTC sleeve
BH = "#5b6b7a"           # buy & hold
COIN_COLORS = {
    "BTC/USD": "#c45c26",
    "ETH/USD": "#2f5d8c",
    "LTC/USD": "#6b7c85",
    "XRP/USD": "#1f8a70",
    "BCH/USD": "#8b5e3c",
}
CORR_CMAP = LinearSegmentedColormap.from_list(
    "sand_teal", ["#f0ebe3", "#9bb8b0", "#0f6b5c"],
)


def _style(ax, title: str | None = None):
    ax.set_facecolor(PAPER)
    ax.tick_params(colors=INK, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.85)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)


def _norm_equity(eq: pd.Series) -> pd.Series:
    return eq / float(eq.iloc[0])


def _drawdown(eq: pd.Series) -> pd.Series:
    peak = eq.cummax()
    return eq / peak - 1.0


def _active_count(portfolio, symbols, index) -> pd.Series:
    ph = portfolio.positions_history
    if ph.empty:
        return pd.Series(0.0, index=index)
    cols = [s for s in symbols if s in ph.columns]
    if not cols:
        return pd.Series(0.0, index=index)
    active = (ph[cols].abs() > 1e-12).astype(float).sum(axis=1)
    return active.reindex(index).ffill().fillna(0.0)


def build_dashboard(
    book,
    singles: dict,
    bh_btc,
    frames: dict,
    risk_budget: float,
    risk_per_name: float,
    path: Path,
):
    symbols = list(frames.keys())
    book_eq = book.equity_curve
    btc_eq = singles["BTC/USD"].equity_curve if "BTC/USD" in singles else None
    bh_eq = bh_btc.equity_curve if bh_btc is not None else None

    rets = pd.DataFrame(
        {s: p.equity_curve.pct_change() for s, p in singles.items()}
    ).dropna(how="all")
    corr = rets.corr()
    attr = spectral_risk_attribution(rets.fillna(0.0))
    active = _active_count(book, symbols, book_eq.index)

    # Calendar-year book returns
    yr = book_eq.pct_change().groupby(book_eq.index.year).apply(
        lambda s: float((1.0 + s.fillna(0.0)).prod() - 1.0)
    )

    fig = plt.figure(figsize=(14.5, 11.5), facecolor=PAPER)
    gs = GridSpec(
        3, 3, figure=fig,
        height_ratios=[1.15, 1.0, 0.95],
        hspace=0.38, wspace=0.28,
        left=0.07, right=0.98, top=0.90, bottom=0.06,
    )

    # --- Title block ---
    fig.suptitle(
        "Daily long-only Donchian — multi-coin portfolio",
        fontsize=16, color=INK, x=0.07, ha="left", y=0.97,
    )
    book_stats = summarize(book_eq, book)
    subtitle = (
        f"{' · '.join(s.split('/')[0] for s in symbols)}   |   "
        f"risk budget {risk_budget:.0%} ({risk_per_name:.2%}/name)   |   "
        f"Sharpe {book_stats['sharpe']:+.2f}   "
        f"CAGR {book_stats['cagr']:+.1%}   "
        f"MaxDD {book_stats['max_dd']:.1%}   "
        f"{book_eq.index[0].date()} → {book_eq.index[-1].date()}"
    )
    fig.text(0.07, 0.935, subtitle, fontsize=9, color="#5a6570", ha="left")

    # 1) Normalized equity
    ax1 = fig.add_subplot(gs[0, :2])
    _style(ax1, "Normalized equity")
    ax1.plot(book_eq.index, _norm_equity(book_eq), color=ACCENT, lw=1.8,
             label="Portfolio")
    if btc_eq is not None:
        ax1.plot(btc_eq.index, _norm_equity(btc_eq), color=BTC_ONLY, lw=1.2,
                 alpha=0.9, label="BTC sleeve only")
    if bh_eq is not None:
        ax1.plot(bh_eq.index, _norm_equity(bh_eq), color=BH, lw=1.0,
                 alpha=0.75, label="BTC buy & hold")
    ax1.axhline(1.0, color=GRID, lw=0.8)
    ax1.legend(loc="upper left", fontsize=8, frameon=False)
    ax1.set_ylabel("Growth of $1")

    # 2) Spectral risk
    ax2 = fig.add_subplot(gs[0, 2])
    _style(ax2, "Spectral risk share")
    shares = attr["risk_share"].to_numpy()
    labels = attr.index.tolist()
    colors = [ACCENT] + [GRID] * (len(shares) - 1)
    ax2.barh(labels[::-1], shares[::-1], color=colors[::-1], height=0.65)
    ax2.set_xlim(0, 1.0)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.text(
        0.98, 0.05, f"PC1 = {shares[0]:.0%}",
        transform=ax2.transAxes, ha="right", va="bottom",
        fontsize=9, color=INK,
    )

    # 3) Per-coin overlay
    ax3 = fig.add_subplot(gs[1, :2])
    _style(ax3, "Per-coin strategy equity (standalone 2% risk)")
    for symbol, port in singles.items():
        eq = port.equity_curve
        ax3.plot(
            eq.index, _norm_equity(eq), lw=1.15,
            color=COIN_COLORS.get(symbol, INK),
            label=symbol.split("/")[0],
        )
    ax3.axhline(1.0, color=GRID, lw=0.8)
    ax3.legend(loc="upper left", fontsize=8, ncol=min(5, len(singles)),
               frameon=False)
    ax3.set_ylabel("Growth of $1")

    # 4) Correlation heatmap
    ax4 = fig.add_subplot(gs[1, 2])
    _style(ax4, "Strategy-return correlation")
    labels_short = [s.split("/")[0] for s in corr.columns]
    im = ax4.imshow(corr.to_numpy(), cmap=CORR_CMAP, vmin=0.0, vmax=1.0,
                    aspect="equal")
    ax4.set_xticks(range(len(labels_short)), labels_short, fontsize=8)
    ax4.set_yticks(range(len(labels_short)), labels_short, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax4.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                     fontsize=7, color=INK)
    cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    mean_corr = float(
        corr.where(~np.eye(len(corr), dtype=bool)).stack().mean()
    )
    ax4.text(
        0.0, -0.18, f"mean pairwise = {mean_corr:.2f}",
        transform=ax4.transAxes, fontsize=8, color="#5a6570",
    )

    # 5) Drawdowns
    ax5 = fig.add_subplot(gs[2, 0])
    _style(ax5, "Drawdown")
    dd_book = _drawdown(book_eq)
    ax5.fill_between(dd_book.index, dd_book.to_numpy(), 0,
                     color=ACCENT, alpha=0.45, label="Portfolio")
    if btc_eq is not None:
        dd_btc = _drawdown(btc_eq)
        ax5.plot(dd_btc.index, dd_btc.to_numpy(), color=BTC_ONLY, lw=1.0,
                 label="BTC sleeve")
    if bh_eq is not None:
        dd_bh = _drawdown(bh_eq)
        ax5.plot(dd_bh.index, dd_bh.to_numpy(), color=BH, lw=0.9, alpha=0.8,
                 label="BTC B&H")
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax5.legend(loc="lower left", fontsize=7, frameon=False)

    # 6) Concurrent exposure
    ax6 = fig.add_subplot(gs[2, 1])
    _style(ax6, "Names active")
    ax6.fill_between(active.index, active.to_numpy(), 0,
                     color=ACCENT, alpha=0.5, step="mid")
    ax6.plot(active.index, active.to_numpy(), color=ACCENT, lw=0.8,
             drawstyle="steps-mid")
    ax6.set_ylim(0, max(len(symbols), 1) + 0.3)
    ax6.set_ylabel("# long")
    ax6.text(
        0.98, 0.92,
        f"flat {(active == 0).mean():.0%} · "
        f"2+ {(active >= 2).mean():.0%} · "
        f"mean {active.mean():.2f}",
        transform=ax6.transAxes, ha="right", va="top", fontsize=8,
        color="#5a6570",
    )

    # 7) Calendar years + per-coin bars (split panel conceptually in one ax)
    ax7 = fig.add_subplot(gs[2, 2])
    _style(ax7, "Calendar-year book return")
    years = yr.index.astype(int).to_list()
    vals = yr.to_numpy()
    colors = [ACCENT if v >= 0 else "#a33b2b" for v in vals]
    ax7.bar(range(len(years)), vals, color=colors, width=0.75)
    ax7.set_xticks(range(len(years)), [str(y) for y in years],
                   rotation=45, ha="right", fontsize=7)
    ax7.axhline(0, color=INK, lw=0.6)
    ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=PAPER)
    plt.close(fig)
    return path


def build_coin_cards(singles: dict, path: Path):
    """Secondary figure: per-coin Sharpe / CAGR / MaxDD cards."""
    rows = []
    for symbol, port in singles.items():
        s = summarize(port.equity_curve, port)
        rows.append({
            "symbol": symbol.split("/")[0],
            "sharpe": s["sharpe"],
            "cagr": s["cagr"],
            "max_dd": s["max_dd"],
            "trades": s["trades"],
        })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), facecolor=PAPER)
    fig.suptitle("Standalone long-only Donchian — per-coin diagnostics",
                 fontsize=12, color=INK, x=0.06, ha="left")

    for ax, col, title, fmt in [
        (axes[0], "sharpe", "Sharpe", lambda v: f"{v:+.2f}"),
        (axes[1], "cagr", "CAGR", lambda v: f"{v:+.1%}"),
        (axes[2], "max_dd", "Max DD", lambda v: f"{v:.1%}"),
    ]:
        ax.set_facecolor(PAPER)
        colors = [
            COIN_COLORS.get(f"{r['symbol']}/USD", INK) for _, r in df.iterrows()
        ]
        ax.barh(df["symbol"], df[col], color=colors, height=0.62)
        ax.axvline(0, color=INK, lw=0.6)
        for i, v in enumerate(df[col]):
            ax.text(v, i, f"  {fmt(v)}", va="center", fontsize=8, color=INK)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.tick_params(colors=INK, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(True, axis="x", color=GRID, lw=0.6)

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])
    fig.savefig(path, dpi=140, facecolor=PAPER)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default=",".join(DEFAULT_UNIVERSE))
    parser.add_argument("--risk-budget", type=float, default=0.02)
    args = parser.parse_args()
    symbols = tuple(s.strip() for s in args.universe.split(",") if s.strip())
    risk_per_name = args.risk_budget / max(len(symbols), 1)

    print("=" * 72)
    print("Multi-coin Donchian — visualization dashboard")
    print("=" * 72)
    frames = load_universe(symbols)
    print(f"Universe: {', '.join(symbols)}")
    print(f"Window:   {min(df.index.min() for df in frames.values()).date()} → "
          f"{max(df.index.max() for df in frames.values()).date()}")

    print("Running standalone sleeves ...")
    singles = {
        s: run_single(s, df, risk=0.02) for s, df in frames.items()
    }
    print("Running combined book ...")
    book = run_portfolio(frames, risk_per_name=risk_per_name)
    bh_btc = run_bh("BTC/USD", frames["BTC/USD"]) if "BTC/USD" in frames else None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dash = OUTPUT_DIR / "multi_coin_dashboard.png"
    cards = OUTPUT_DIR / "multi_coin_per_name.png"
    build_dashboard(
        book, singles, bh_btc, frames,
        risk_budget=args.risk_budget, risk_per_name=risk_per_name, path=dash,
    )
    build_coin_cards(singles, cards)

    s = summarize(book.equity_curve, book)
    print(f"Portfolio: Sharpe {s['sharpe']:+.3f}  CAGR {s['cagr']:+.1%}  "
          f"MaxDD {s['max_dd']:.1%}  trades={s['trades']}")
    print(f"Dashboard: {dash}")
    print(f"Per-coin:  {cards}")


if __name__ == "__main__":
    main()
