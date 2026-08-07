# Analytics & Tearsheets

Package: `quantester/analytics`

Analytics run **after** the backtest, on the equity curve, fills, and trades
the portfolio recorded. Nothing in this package is wired into the event loop.

## Performance statistics

`quantester/analytics/performance.py`. All functions take the equity curve
(`portfolio.equity_curve`, a `pd.Series` indexed by timestamp).

| Function | Returns |
| --- | --- |
| `log_returns(equity)` | Daily log returns. |
| `annualized_sharpe(equity, risk_free_daily=0.0)` | `(μ_daily − Rf)/σ_daily × √252`. |
| `max_drawdown(equity)` | Dict: `max_drawdown` (worst peak-to-trough, negative), `duration` (calendar days to recover the high-watermark), `peak`, `trough`. |
| `drawdown_series(equity)` | Underwater series `equity/HWM − 1` (for plotting). |
| `calmar_ratio(equity)` | Annualized return / \|max drawdown\|. |
| `summarize(equity, risk_free_daily=0.0)` | One dict with total return, Sharpe, MDD, MDD duration, Calmar — the standard headline block. |

### Carver cost drag

```python
from quantester.analytics.performance import carver_cost_drag_sr, speed_limit_warning

drag = carver_cost_drag_sr(annual_turnover=4.0, standardized_cost_sr=0.01)
warning = speed_limit_warning(drag)   # None or a warning string
```

Cost drag in Sharpe units = annual round-trip turnover × standardized
instrument cost (in SR). Carver's **"speed limit" of 0.08 SR/yr** is surfaced
as a warning: above it, turnover is consuming the edge faster than a
realistic allocator can sustain.

## Tearsheets

`quantester/analytics/tearsheet.py`.

```python
from quantester.analytics.tearsheet import generate_tearsheet

stats = generate_tearsheet(
    portfolio.equity_curve,
    "output/tearsheet.png",          # parent dirs created automatically
    title="My strategy",
    extra_stats={"DSR": "0.97"},     # optional lines added to the stats box
)
```

Renders a PNG — equity curve, underwater (drawdown) plot, log-return
histogram, and a monospace stats box — and returns the summary dict.
Matplotlib runs headless (`Agg`), so this works on servers and CI.

## Trials Registry

`quantester/analytics/trials_registry.py` — a SQLite-backed log of every
backtest/optimization trial. It exists because the Deflated Sharpe Ratio
mathematically requires the *actual* number of trials N and the cross-trial
variance of Sharpes — values you cannot reconstruct honestly after the fact.

```python
from quantester.analytics.trials_registry import (
    TrialsRegistry, auto_register_from_equity,
)

registry = TrialsRegistry("trials.db")          # or ":memory:"
registry.register_experiment(
    strategy_id="ma_cross",
    params={"fast": 10, "slow": 40},
    sharpe=1.23,
    data_source="yfinance",
    universe=["AAPL"],
    cost_model={"spread_bps": 5},
    random_seed=7,
    code_version="0.1.0",
)
# Or derive moments from an equity curve automatically:
auto_register_from_equity(registry, equity, strategy_id="ma_cross",
                          params={"fast": 10, "slow": 40})
registry.n_trials()          # N for DSR — the true registered trial count
registry.experiment_ids()    # unique experiment hashes
registry.sharpe_variance()   # sigma^2 across trials
registry.best_trial()        # dict with params, sharpe, moments
registry.close()
```

Return helpers in `quantester/analytics/returns.py` keep **simple returns**,
**log returns**, **P&L**, and **equity/wealth** mathematically distinct
(`wealth_from_simple_returns`, `wealth_from_log_returns`, `simple_to_log`,
`log_to_simple`).

**Parallel-safe write path:** SQLite rejects concurrent writers. During
parallel optimization, workers append records with
`TrialsRegistry.write_jsonl_record(path, record)`; afterwards a single thread
bulk-loads them with `registry.import_jsonl(path)`.

## Deflated Sharpe Ratio (DSR / PSR)

`quantester/analytics/dsr.py` — Bailey & López de Prado. The primary defense
against **selection bias**: deflates the observed Sharpe by the expected
maximum Sharpe you would have gotten from N skill-less trials.

```python
from quantester.analytics.dsr import (
    probabilistic_sharpe_ratio, deflated_sharpe_ratio, dsr_from_registry,
)

psr = probabilistic_sharpe_ratio(sr_hat=1.2, sr_benchmark=0.0, n_obs=500,
                                 skew=0.1, kurtosis=3.2)
dsr = dsr_from_registry(registry, sr_hat=best["sharpe"], n_obs=best["n_obs"],
                        skew=best["skew"], kurtosis=best["kurt"])
```

- `expected_max_sharpe(n_trials, trial_variance)` — the E[max SR] null
  benchmark via the Euler–Mascheroni quantile blend.
- `deflated_sharpe_ratio(...)` — PSR against that benchmark.
- `dsr_from_registry(...)` — pulls N and σ²_SR straight from the registry so
  the inputs always reflect what you actually tried. **Never hand-feed N.**

Interpretation: DSR is a probability — the chance the true Sharpe exceeds the
best-of-N-nulls benchmark. Want it ≥ 0.95 before taking a result seriously.
