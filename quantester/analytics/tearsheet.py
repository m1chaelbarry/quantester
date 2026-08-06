"""Tearsheet generation (Report 1 section 2.5): equity, drawdown, returns, stats."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import performance


def generate_tearsheet(equity: pd.Series, path, title: str = "Quantester Tearsheet",
                       extra_stats: dict | None = None) -> dict:
    """Render the tearsheet PNG and return the summary statistics dict."""
    stats = performance.summarize(equity)
    if extra_stats:
        stats.update(extra_stats)

    dd = performance.drawdown_series(equity)
    rets = performance.log_returns(equity)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False,
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    fig.suptitle(title)

    axes[0].plot(equity.index, equity.values, lw=1.2)
    axes[0].set_ylabel("Equity")
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(dd.index, dd.values, 0, alpha=0.5)
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.3)

    axes[2].hist(rets.values, bins=50, alpha=0.7)
    axes[2].set_ylabel("Frequency")
    axes[2].set_xlabel("Log return")
    axes[2].grid(alpha=0.3)

    text = "\n".join(
        [
            f"Total return:      {stats['total_return']:>10.2%}",
            f"Sharpe (ann.):     {stats['sharpe']:>10.3f}",
            f"Max drawdown:      {stats['max_drawdown']:>10.2%}",
            f"MDD duration:      {stats['max_drawdown_duration_days']:>7d} d",
            f"Calmar:            {stats['calmar']:>10.3f}",
        ]
        + [f"{k}: {v}" for k, v in (extra_stats or {}).items()]
    )
    axes[0].text(0.01, 0.98, text, transform=axes[0].transAxes, va="top",
                 family="monospace", fontsize=8,
                 bbox={"boxstyle": "round", "alpha": 0.15})

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return stats
