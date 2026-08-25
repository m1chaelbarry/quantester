# Validation: Truncation, CPCV, PBO, Gates

Package: `quantester/validation`

Three orthogonal defenses, ordered from "always run" to "run after any
optimization", plus a research-governance gate layer. The full workflow with
acceptance gates is in the
[Validation Workflow tutorial](../tutorials/validation-workflow.md).

## Truncation diagnostic — temporal leakage check

`quantester/validation/truncation.py` (Ernest Chan's check).

Run the full backtest (positions file A), truncate the last N bars, re-run
the *identical* program (file B), and compare overlapping positions with an
**absolute-difference** tolerance. This is a strong temporal-leakage
diagnostic — not a formal proof that all look-ahead is impossible.

```python
from quantester.validation.truncation import run_truncation_test

def run(truncate_last=None):
    data = {"AAA": df.iloc[:-truncate_last] if truncate_last else df}
    handler = HistoricCSVDataHandler(data)
    ...
    engine.run_backtest()
    return portfolio.positions_history

result = run_truncation_test(run, n_truncated=20)
print(result)        # Truncation diagnostic [PASS]: compared ... 0 mismatch(es).
result.passed        # bool — the gate
```

`TruncationResult` carries `passed`, `rows_compared`, and up to 20 example
`mismatches` (timestamp, symbol, full vs truncated value, abs_diff).

## Research gates

`quantester/validation/gates.py` aggregates PASS / WARN / FAIL /
NOT_APPLICABLE outcomes. A run may only claim `VALIDATED` when every
mandatory gate is PASS or NOT_APPLICABLE — mandatory FAIL blocks validation
and warnings remain visible.

```python
from quantester.validation import evaluate_gates, build_validation_report

gates = evaluate_gates(
    data_audit_status="PASS",
    truncation_passed=True,
    pbo_passed=True,
    pbo_value=0.04,
    dsr_value=0.97,
    untouched_oos_passed=True,
    monte_carlo_passed=True,
    accounting_invariant_passed=True,
    execution_assumptions_documented=True,
    execution_stress_passed=True,
    cpcv_passed=True,
)
report = build_validation_report(gates, trial_count=48, code_version="0.1.0")
assert report.validated
```

## Purged cross-validation (for ML strategies)

`quantester/validation/cpcv.py` (AFML ch. 7/12). Financial labels overlap in
time (a 10-bar label starting Monday shares information with one starting
Tuesday), so plain k-fold leaks. Purging drops training samples whose
**label interval** overlaps the test interval; the embargo drops training
samples just *after* the test set.

### `PurgedKFold(n_splits=3, t1=None, embargo_bars=None, lookback=None, lookahead=None, pct_embargo=None)`

sklearn-style splitter. `t1` is a Series of label end-times aligned with
`X.index` (e.g. the vertical barrier from triple-barrier labeling). A train
sample `[t_i0, t_i1]` is purged against test `[t_j0, t_j1]` when it starts
inside, ends inside, or envelops the test interval.

Embargo length is an **integer bar window** after the test label end (ruling
D8). Priority: `embargo_bars` → `min(lookback, lookahead) − 1` (a single
horizon uses that value minus 1) → explicit `pct_embargo` floored to bars
(de Prado ~0.01T research override) → **0 bars**. The old silent
`pct_embargo=0.01` default is gone.

```python
for train_idx, test_idx in PurgedKFold(5, t1=t1, lookahead=10).split(X):
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    score(X.iloc[test_idx], y.iloc[test_idx])
```

### `CombinatorialPurgedKFold(n_groups=6, k_test=2, t1=None, embargo_bars=None, lookback=None, lookahead=None, pct_embargo=None)`

CPCV: partition T observations into N groups (no shuffling), then test on
every combination of k groups — `C(N, N−k)` splits spanning
`φ[N,k] = C(N−1, k−1)` unique backtest paths (`.n_splits`, `.n_paths`; the
binomial identity is exact — do not compute `(k/N)·C(N, N−k)` in floats).
Use the per-path Sharpe distribution — not a single number — for honest
model selection. Embargo knobs are identical to `PurgedKFold`.

## PBO — Probability of Backtest Overfitting

`quantester/validation/pbo.py` (Bailey–López de Prado CSCV, the
rank-logit algorithm).

After any parameter sweep: collect the N trials' synchronous PnL series into
a `T × N` matrix, partition row-wise into S blocks, and for each of the
`C(S, S/2)` train/test block combinations ask: *the trial that was best
in-sample — where does it rank out-of-sample?* PBO is the fraction of
combinations where that rank is below median (logit < 0).

```python
from quantester.validation.pbo import pbo_cscv, PBO_GATE   # PBO_GATE = 0.10

result = pbo_cscv(pnl_dataframe, n_blocks=16)   # n_blocks must be even
result.pbo            # probability of backtest overfitting
result.passes_gate    # pbo < 0.10 — hard gate before paper trading
result.logits         # per-combination logits (plot their distribution)
```

A high PBO means your "best" parameters won in-sample by luck and degrade
out-of-sample as often as not. Default performance per block is per-trial
Sharpe; pass `performance_fn=` to use another metric.
