"""Ernest Chan's historical truncation test: temporal-leakage diagnostic.

Protocol (Chan, "Quantitative Trading", 2nd ed., backtesting chapter; the
exact test is not covered by the user's quant-literature notebook, so it is
implemented from Chan's canonical description and Report 1 section 4.1):

- Run A: execute the backtest on the FULL dataset and save the position
  ledger (per-symbol portfolio weights at each close) as ``positions_A.csv``.
- Run B: truncate the last ``n_truncated`` trading days from the data, re-run
  the identical program, save ``positions_B.csv``.
- Comparison: drop the final ``n_truncated`` rows of A so it matches B's
  date-index length; the two ledgers must agree within a floating-point
  absolute tolerance of 1e-9
  (``abs(position_full[t] - position_truncated[t]) <= atol``). Any divergence
  is strong evidence the pipeline consumed future data (look-ahead leakage)
  and raises ``ValueError`` pinpointing the first divergence. This is a
  diagnostic, not a formal mathematical proof that all look-ahead is
  impossible.

Truncation is performed against the master (outer-join union) calendar:
removing the last N master timestamps from every leg keeps Run B's calendar
an exact prefix of Run A's, even when availability masks punch holes in one
leg's own calendar.

Everything runs offline: callers pass pre-downloaded daily-bar data as
``{symbol: DataFrame}`` or ``{symbol: csv_path}`` (Yahoo-style
``datetime,open,high,low,close,volume`` schema). No external API requests are
made; the module's only RNG use is the seeded mock-data demo under ``__main__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from ..data.csv_handler import HistoricCSVDataHandler
from ..engine import BacktestEngine
from ..execution.costs import CostModel
from ..execution.simulator import SimulatedExecutionHandler
from ..portfolio.portfolio import HedgeRatioSizer, PercentEquitySizer, PortfolioManager
from ..strategy.base import Strategy
from ..strategy.pairs_trading import PairsTradingStrategy

DataSource = dict[str, Union[pd.DataFrame, str, Path]]

#: 5 bps per trade per leg, expressed as half of a 10 bps quoted spread: the
#: deterministic CostModel charges spread_pct / 2 * price per share on every
#: fill, i.e. exactly 0.0005 * notional per leg per trade (phi_t component).
FIVE_BPS_COST_MODEL = CostModel(
    fixed_commission=0.0,
    per_share_commission=0.0,
    spread_pct=0.0010,
    slippage_vol_coef=0.0,
    impact_coef=0.0,
)

#: Frictionless twin used to isolate cost impact on performance metrics.
ZERO_COST_MODEL = CostModel(
    fixed_commission=0.0,
    per_share_commission=0.0,
    spread_pct=0.0,
    slippage_vol_coef=0.0,
    impact_coef=0.0,
)


@dataclass
class EngineRunArtifacts:
    """Assembled engine artefacts of one causal backtest run.

    Named distinctly from any trader-facing result type so the two are not
    confused in imports or docs.
    """

    portfolio: PortfolioManager
    strategy: Strategy
    handler: HistoricCSVDataHandler


# Backward-compatible alias (prefer EngineRunArtifacts in new code).
BacktestResult = EngineRunArtifacts


@dataclass
class TruncationTestResult:
    """Outcome of a passing truncation verification (failure raises)."""

    passed: bool
    n_truncated: int
    rows_compared: int
    positions_a_path: Path
    positions_b_path: Path

    def __str__(self) -> str:
        return (
            f"Truncation test [PASS]: {self.rows_compared} overlapping rows "
            f"identical after truncating the last {self.n_truncated} bars "
            f"(positions_A={self.positions_a_path}, "
            f"positions_B={self.positions_b_path})."
        )


def run_pairs_backtest(
    data: DataSource,
    *,
    cost_model: Optional[CostModel] = None,
    initial_capital: float = 100_000.0,
    leg_fraction: float = 0.5,
    strategy_factory: Optional[Callable] = None,
    **strategy_kwargs,
) -> EngineRunArtifacts:
    """Assemble and run one fresh, fully causal pairs backtest.

    Every run builds NEW handler/strategy/portfolio/execution instances so no
    state can leak between Run A and Run B. ``leg_fraction`` is the fraction
    of equity targeted on the Y leg (X is sized q_X = -beta q_Y).
    """
    handler = HistoricCSVDataHandler(data)
    if strategy_factory is not None:
        strategy = strategy_factory(handler)
    else:
        strategy = PairsTradingStrategy(handler, **strategy_kwargs)
    portfolio = PortfolioManager(
        handler,
        initial_capital,
        sizer=HedgeRatioSizer(leg_fraction),
    )
    execution = SimulatedExecutionHandler(cost_model or ZERO_COST_MODEL)
    BacktestEngine(handler, strategy, portfolio, execution).run_backtest()
    return EngineRunArtifacts(portfolio=portfolio, strategy=strategy, handler=handler)


def _as_frame(source: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source
    return pd.read_csv(Path(source), parse_dates=["datetime"], index_col="datetime")


def position_weights(result: EngineRunArtifacts, data: DataSource) -> pd.DataFrame:
    """Position ledger as target portfolio weights per symbol per close.

    w_{i,t} = qty_{i,t} * close_{i,t} / equity_t, with the last available
    close carried forward for timestamps where a leg is untradeable
    (availability masks never erase timestamps).
    """
    positions = result.portfolio.positions_history
    if positions.empty:
        raise ValueError("Backtest produced no position ledger to compare.")
    equity = result.portfolio.equity_curve
    closes = pd.DataFrame(
        {symbol: _as_frame(data[symbol])["close"] for symbol in positions.columns}
    ).reindex(positions.index).ffill()
    weights = positions.mul(closes).div(equity, axis=0).fillna(0.0)
    weights.index.name = "datetime"
    return weights


def truncate_on_master_calendar(data: DataSource, n_truncated: int) -> dict:
    """Remove the last ``n_truncated`` master-calendar timestamps from every leg."""
    frames = {symbol: _as_frame(df) for symbol, df in data.items()}
    master = sorted(set().union(*[set(df.index) for df in frames.values()]))
    if not 1 <= n_truncated < len(master):
        raise ValueError(
            f"n_truncated={n_truncated} must lie in [1, {len(master) - 1}] "
            f"for a master calendar of {len(master)} timestamps."
        )
    cutoff = master[-n_truncated]  # first excluded timestamp
    return {symbol: df.loc[df.index < cutoff] for symbol, df in frames.items()}


def _read_positions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")


def run_pairs_truncation_test(
    data: DataSource,
    output_dir: Union[str, Path],
    *,
    n_truncated: int = 50,
    atol: float = 1e-9,
    cost_model: Optional[CostModel] = None,
    initial_capital: float = 100_000.0,
    leg_fraction: float = 0.5,
    strategy_factory: Optional[Callable] = None,
    **strategy_kwargs,
) -> TruncationTestResult:
    """Execute Chan's truncation loop and assert leakage-free positions.

    Raises
    ------
    ValueError
        If the overlapping ledgers diverge beyond ``atol`` (look-ahead
        leakage). The message pinpoints the exact timestamp and symbol of the
        first divergence and reports the total number of divergent cells.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Run A: full history.
    result_a = run_pairs_backtest(
        data, cost_model=cost_model, initial_capital=initial_capital,
        leg_fraction=leg_fraction, strategy_factory=strategy_factory,
        **strategy_kwargs,
    )
    weights_a = position_weights(result_a, data)
    path_a = output / "positions_A.csv"
    weights_a.to_csv(path_a)

    # Run B: identical program on data truncated by the last N master bars.
    truncated = truncate_on_master_calendar(data, n_truncated)
    result_b = run_pairs_backtest(
        truncated, cost_model=cost_model, initial_capital=initial_capital,
        leg_fraction=leg_fraction, strategy_factory=strategy_factory,
        **strategy_kwargs,
    )
    weights_b = position_weights(result_b, truncated)
    path_b = output / "positions_B.csv"
    weights_b.to_csv(path_b)

    # Compare the persisted ledgers (CSV round-trip, as Chan specifies files).
    a = _read_positions(path_a)
    b = _read_positions(path_b)
    a_trunc = a.iloc[:-n_truncated]
    if len(a_trunc) != len(b) or not a_trunc.index.equals(b.index):
        raise ValueError(
            "Truncation test failed structurally: after dropping the final "
            f"{n_truncated} rows of positions_A, the date index does not match "
            f"positions_B ({len(a_trunc)} vs {len(b)} rows). The master "
            "calendar itself is not a deterministic prefix — the data pipeline "
            "is consuming future information."
        )
    a_trunc, b = a_trunc.align(b, join="inner", axis=1)

    diff = (a_trunc.to_numpy(dtype=float) - b.to_numpy(dtype=float))
    bad = np.abs(diff) > atol
    if bad.any():
        rows, cols = np.where(bad)
        order = np.lexsort((cols, rows))  # chronological, then symbol order
        r, c = rows[order[0]], cols[order[0]]
        raise ValueError(
            "Look-ahead leakage detected: position ledgers diverge at "
            f"{a_trunc.index[r]} for symbol '{a_trunc.columns[c]}' "
            f"(full={a_trunc.iloc[r, c]!r}, truncated={b.iloc[r, c]!r}, "
            f"|diff|={abs(diff[r, c]):.3e} > atol={atol:.1e}). "
            f"{int(bad.sum())} divergent cell(s) across "
            f"{len(np.unique(rows))} timestamp(s). The strategy, portfolio, "
            "or data pipeline is consuming information from after the "
            "simulated time t."
        )
    return TruncationTestResult(
        passed=True,
        n_truncated=n_truncated,
        rows_compared=len(b),
        positions_a_path=path_a,
        positions_b_path=path_b,
    )


if __name__ == "__main__":
    # Offline self-check on seeded mock GLD/GDX daily bars (no API requests).
    from ..analytics.performance import summarize
    from ..utils.synthetic import make_cointegrated_pair

    mock = make_cointegrated_pair(n_bars=750, seed=7)
    for label, costs in [("gross (no costs)", ZERO_COST_MODEL),
                         ("net (5 bps/leg)", FIVE_BPS_COST_MODEL)]:
        res = run_pairs_backtest(mock, cost_model=costs)
        stats = summarize(res.portfolio.equity_curve)
        print(
            f"{label:>18}: sharpe={stats['sharpe']:+.3f}  "
            f"mdd={stats['max_drawdown']:+.2%}  calmar={stats['calmar']:+.3f}"
        )
    outcome = run_pairs_truncation_test(mock, Path("validation_output"),
                                        n_truncated=50)
    print(outcome)
