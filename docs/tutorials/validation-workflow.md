# Validation Workflow: Trusting a Backtest

A backtest that makes money has told you almost nothing. This page is the
mandatory checklist Quantester runs before a strategy is trusted — each gate
catches a different way backtests lie. The gates are ordered; run them in
order, because later gates assume earlier ones passed.

| # | Gate | Catches | Pass criterion | Module |
| --- | --- | --- | --- | --- |
| 1 | Truncation test | Look-ahead leakage in the pipeline | `passed == True` | `validation/truncation.py` |
| 2 | CPCV (ML only) | Label overlap leaking train ↔ test | OOS Sharpe distribution across paths | `validation/cpcv.py` |
| 3 | PBO | Parameter selection by luck | `pbo < 0.10` | `validation/pbo.py` |
| 4 | DSR | Selection bias across everything you tried | DSR ≥ 0.95 | `analytics/dsr.py` + registry |
| 5 | MCPT | "Edge" that exists in any random path | `p < 0.05` | `montecarlo/permutation.py` |
| — | Autocorrelation gate | Invalid iid resampling assumptions | run **before** 5 | `montecarlo/diagnostics.py` |

## Gate 1 — Truncation test (always, first)

Chop the last N bars and re-run the identical program. Overlapping positions
must match bit-for-bit; a mismatch means future data leaked into past
decisions and **every other result is void until fixed**.

```python
from quantester.validation.truncation import run_truncation_test
result = run_truncation_test(run_fn, n_truncated=20)
assert result.passed, result.mismatches
```

## Gate 2 — Purged/embargoed CPCV (whenever a model is fitted)

If your strategy fits *anything* (a meta-labeling classifier, a regression on
features), plain k-fold is invalid: labels that overlap in time leak between
train and test. Use purged, embargoed CV — and for path-distributed results,
combinatorial CPCV:

```python
from quantester.validation.cpcv import CombinatorialPurgedKFold

cpcv = CombinatorialPurgedKFold(n_groups=6, k_test=2, t1=t1, pct_embargo=0.01)
sharpes = []
for train_idx, test_idx in cpcv.split(X):
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    sharpes.append(evaluate_oos(model, X, y, test_idx))
# judge the DISTRIBUTION of len(sharpes) == cpcv.n_splits scores,
# reconstructable into cpcv.n_paths distinct backtest paths
```

## Gate 3 — PBO after any parameter sweep

You swept lookbacks `(10, 20, 40)` and kept the best. Was it skill or the
luckiest column? Feed all trials' synchronous PnL into CSCV:

```python
from quantester.validation.pbo import pbo_cscv
result = pbo_cscv(pnl_matrix, n_blocks=16)   # pnl_matrix: T x N trials
assert result.passes_gate                     # PBO < 0.10
```

## Gate 4 — Registry-driven DSR

Every trial you ran belongs in the registry — including the failures. DSR
then deflates your champion's Sharpe by the expected best-of-N null:

```python
from quantester.analytics.trials_registry import TrialsRegistry
from quantester.analytics.dsr import dsr_from_registry

registry = TrialsRegistry("trials.db")
for params in grid:
    ...                                     # run backtest
    registry.log_trial(params=params, sharpe=stats["sharpe"],
                       skew=..., kurt=..., n_obs=len(rets), run_id="sweep")
best = registry.best_trial()
dsr = dsr_from_registry(registry, sr_hat=best["sharpe"], n_obs=best["n_obs"],
                        skew=best["skew"], kurtosis=best["kurt"])
assert dsr >= 0.95
```

Hand-feeding `n_trials` from memory defeats the purpose — the registry *is*
the honest N.

## Gate 5 — MCPT (via the fast-track)

```python
from quantester.montecarlo.diagnostics import autocorrelation_gate
from quantester.montecarlo.permutation import permutation_test

gate = autocorrelation_gate(oos_returns)
# if gate.serial_correlation: use block bootstrap / OU paths instead of MCPT

result = permutation_test(close, optimizer, n_reps=1000, seed=7)
assert result.significant                   # p < 0.05
```

The optimizer must retrain *from scratch* on each permuted path (same sweep,
same costs) and run on the vectorized fast-track — 10,000 event-loop re-runs
are intractable. This requires your strategy to implement
`vectorized_signals` (see the
[tutorial](../tutorials/creating-a-strategy.md#step-3--write-the-strategy-class)).

## If a gate fails

| Gate | Meaning | What to do |
| --- | --- | --- |
| Truncation | Leak in pipeline | Find where raw data bypasses the DataHandler; fix; re-run everything. |
| CPCV | Model doesn't generalize | Simpler features/model, more data, longer embargo. |
| PBO | Parameters chosen by luck | Widen the economic rationale per parameter; prefer plateaus over peaks; fewer trials. |
| DSR | Too many trials for the observed edge | Report honestly; combine with economic priors; stop iterating on the same data. |
| MCPT | No pattern beyond chance | The market segment may be efficient at your horizon; new hypothesis needed. |
| Autocorrelation | Resampling invalid | Use `empirical_resample(..., block_length=L)` or OU synthetic paths. |

A strategy that passes all five gates has earned the right to a paper-trading
account — and not before.
