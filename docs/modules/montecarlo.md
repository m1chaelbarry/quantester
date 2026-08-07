# Monte Carlo Suite

Package: `quantester/montecarlo`

A single realized backtest path is one draw from a distribution. This package
maps that distribution: what performance looks like when history is
resampled, permuted, or regenerated. Everything here is vectorized — heavy
loops never re-run the event engine.

All functions take `seed=` and use local `numpy.random.Generator`s:
reproducible by construction.

## Fast-track backtesting

`quantester/montecarlo/fast_track.py` — the vectorized execution bypass that
makes 10,000-rep studies tractable.

```python
from quantester.montecarlo.fast_track import fast_backtest
from quantester.execution.costs import CostModel

target = strategy.vectorized_signals({"AAA": df})["AAA"]  # {-1, 0, +1} per close
result = fast_backtest(df, target, CostModel(),
                       initial_capital=100_000.0, units=100.0)
result.equity, result.sharpe, result.total_return
```

**Parity contract** (asserted by `tests/test_montecarlo.py`): targets decided
at close T execute at open T+1; fills use the *same* `CostModel` adverse
adjustments as the event engine; `cash_t = cash_{t-1} − dQ_t·fill −
commission`; `equity_t = cash_t + Q_t·close_t`. If you change fill semantics,
keep that test green or Monte Carlo silently diverges from the engine.

## Trade-level resampling

`quantester/montecarlo/trade_resampling.py`.

```python
from quantester.montecarlo.trade_resampling import (
    empirical_resample, ehlers_randomized_equity,
)

hat = empirical_resample(returns, horizon=260, n_sims=10_000, seed=7)
hat.quantiles()                      # {0.05: ..., 0.5: ..., 0.95: ...} terminal returns
hat.paths                            # (n_sims, horizon+1) equity paths, start at 1.0
```

- `empirical_resample(returns, horizon, n_sims, seed, block_length=None)` —
  "hat" resampling: draw historical net returns **with replacement** to build
  synthetic paths (e.g. 260-day years), preserving the exact empirical
  distribution. Pass `block_length=L` for a stationary block bootstrap that
  preserves serial correlation within blocks.
- `ehlers_randomized_equity(win_rate, profit_factor, avg_loss, n_trades,
  n_sims=10_000, e0=1.0, seed=None)` — Ehlers' parametric randomization: the
  system is stripped to win-rate p and profit factor PF; each trade draws
  u ~ U(0,1), wins `|avg_loss| × PF` if u ≤ p, else loses `|avg_loss|`.
  Returns `(n_sims, n_trades+1)` equity paths.

## MCPT — Monte Carlo Permutation Testing

`quantester/montecarlo/permutation.py`. Shuffling log price changes destroys
the chronological patterns a strategy exploits while preserving the data's
exact statistical moments. A real edge must beat the permuted distribution.

```python
from quantester.montecarlo.permutation import permutation_test

def optimizer(data) -> float:        # RETRAIN from scratch on each path
    ...
    return best_sharpe               # higher is better

result = permutation_test(close, optimizer, n_reps=1000, seed=7)
result.p_value                       # Masters' exact p: count starts at 1
result.significant                   # p < 0.05
```

- **Crucial:** the optimizer is retrained on every permuted path — same
  parameter sweep, same costs — so the permutation distribution includes the
  same selection effects as the original run.
- **Run the [autocorrelation gate](#autocorrelation-gate) first:** serial
  correlation invalidates iid resampling.

### Permutation protocols

| Function | What it shuffles |
| --- | --- |
| `permute_log_changes(prices, rng, offset=0)` | Log changes after `offset`; rebuilds from the base price. The default single-market protocol. |
| `multi_market_permutation(prices_df, rng, offset=0)` | **Protocol I:** identical shuffle indices across all markets — preserves cross-sectional correlations (use for multi-asset strategies). |
| `intra_inter_bar_permutation(ohlc, rng)` | **Protocol II:** intra-bar records (H/O, L/O, C/O) shuffled *jointly*, inter-bar gaps (O_t/C_{t−1}) shuffled independently; reconstructs physically valid OHLC bars. Use when the strategy reads intra-bar geometry. |

### Masters' Trend/Bias/Skill partition

```python
from quantester.montecarlo.permutation import trend_bias_skill

partition = trend_bias_skill(r_orig=R, b_orig=B, r_perm=R̄, b_perm=B̄)
# {"trend", "training_bias", "unbiased_return", "skill"}
```

`Bias = R_perm − B_perm`; `Skill = (R_orig − Bias) − B_orig`; trend is the
buy-and-hold benchmark `B_orig`. Benchmarks are recomputed on permuted paths,
not assumed zero — this separates genuine skill from "the market went up".

## Drawdown bounds (double bootstrap)

`quantester/montecarlo/drawdown.py`. Drawdown is path-dependent: a
single-loop bootstrap of the OOS returns **underestimates** catastrophic
drawdown by over 10×, because the OOS sample is itself a volatile draw from
the parent population. Masters' nested double bootstrap ("bound on a bound")
corrects for both sequencing risk and sampling error.

```python
from quantester.montecarlo.drawdown import (
    double_bootstrap_dd_bound, single_loop_dd_quantile,
)

bound = double_bootstrap_dd_bound(returns, horizon=None,
                                  dd_conf=0.95, bound_conf=0.70,
                                  n_outer=10_000, n_inner=1_000, seed=7)
bound.bound                 # the conservative DD bound
bound.outer_distribution    # for plotting
```

- Inner loop: resample to horizon H → DD distribution → take the `dd_conf`
  quantile. Outer loop: resample the population itself → take the
  `bound_conf` quantile of the inner quantiles.
- `single_loop_dd_quantile` is provided as the anti-conservative benchmark to
  demonstrate the difference.
- Recommended: `dd_conf=0.95, bound_conf=0.70, n_outer=10_000, n_inner=1_000`.

## Synthetic paths (Ornstein-Uhlenbeck) & OTR sweeps

`quantester/montecarlo/synthetic.py`.

```python
from quantester.montecarlo.synthetic import (
    estimate_ou_params, generate_ou_paths, otr_sweep,
)

ou = estimate_ou_params(close)                      # OLS of dP on P
paths = generate_ou_paths(ou, p0=close.iloc[-1], n_steps=120,
                          n_paths=100_000, seed=7)  # (n_paths, n_steps+1)
grid = otr_sweep(paths, stop_losses=[0.05, 0.10],
                 take_profits=[0.10, 0.20])
```

`dP = θ(μ − P)dt + σdW` with `{θ, μ, σ}` estimated from history by OLS (a
non-mean-reverting fit falls back to θ=0, a random walk around the sample
mean). Sweeping stop-loss/take-profit grids over a large path ensemble yields
**Optimal Trading Rules calibrated over the whole stochastic space** rather
than the one realized path — a far more robust way to choose exits than
fitting them to history.

`correlated_gaussian_returns(n_assets, n_obs, cov=None,
common_shock_scale=0, idio_shock_scale=0, ...)` generates multi-asset return
matrices with a Cholesky correlation structure plus injected common and
idiosyncratic shocks (fat tails, regime shifts) for stress-testing
allocators.

## Stationary block bootstrap (OHLCV)

`quantester/montecarlo/synthetic.py` — the Monte Carlo vehicle for
**path-dependent strategies with no closed-form vectorized twin** (e.g. the
tranche pullback ladder): instead of a fast-track, the full event engine
re-runs on each synthetic path.

```python
from quantester.montecarlo.synthetic import bootstrap_ohlcv

frame = bootstrap_ohlcv(df, mean_block=20, seed=42)
```

Politis-Romano stationary bootstrap over bars: block lengths are geometric
(mean `mean_block`), and each bar contributes its close-to-close return,
open gap, intra-bar wick fractions and volume **jointly**, so
`high ≥ max(open, close)` and `low ≤ min(open, close)` hold by construction.
Within-block serial correlation and volatility clustering survive; the
long-run regime ordering is scrambled — that scrambling is the null
hypothesis ("BTC-like short-run structure, shuffled regimes"), not a claim
that history will repeat. Seeded via `numpy.random.Generator`.

Protocol endorsed by the notebook cross-reference (Masters' stationary /
tapered block bootstrap; de Prado's sequential bootstrap as the alternative);
implemented from the canonical Politis-Romano (1994) source. See
`examples/run_parameter_study_ccxt.py` for the full harness
(autocorrelation gate → bootstrap paths → per-path engine re-runs →
no-edge probability and same-path buy-and-hold benchmark).

## Autocorrelation gate

`quantester/montecarlo/diagnostics.py` — **run this before any resampling.**

```python
from quantester.montecarlo.diagnostics import autocorrelation_gate

report = autocorrelation_gate(returns, alpha=0.05, lags=10)
report.serial_correlation     # True -> iid resampling is INVALID
report.recommended_method     # "iid_resampling" or "block_bootstrap_or_ou_paths"
report.runs_p, report.ljung_box_p
```

Combines the Wald-Wolfowitz runs test and the Ljung-Box Q test. If serial
correlation exists, simple shuffling artificially smooths simulated paths and
dangerously underestimates downside risk (Kaufman's autocorrelation trap) —
route to `empirical_resample(..., block_length=L)` or OU paths instead.
