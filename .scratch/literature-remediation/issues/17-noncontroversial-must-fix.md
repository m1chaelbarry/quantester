# Which non-controversial critical defects are must-fix on the spec?

Type: grilling
Status: resolved
Part of: [Literature remediation decision map](../map.md)
Blocked by: 01, 02

## Question

Which synthesis §1 items that are **not** parked behind a §4 conflict are must-fix on the implementation spec (still decisions, not patches)?

Candidates the synthesis already calls uncontested or unambiguous:

- `validation/truncation.py` empty-overlap `n_truncate` NameError (`n_truncated` is the parameter).
- Forbid `bfill` on trading features / `as_daily_reindex`.
- Hedge-ratio pairs sizer (legs currently `strength=1.0`).
- Itô \(-0.5\sigma^2\) in *synthetic* GBM only.
- UTC-midnight drawdown breaker.
- Procyclical MTM sizers.
- Triple-barrier close-only vs high/low path.
- Seal or warn on `source_ohlcv` from `calculate_signals`.
- CPCV `n_paths` integer truncation.
- Pandas hot-path / \(O(TN)\) (severity-only conflict — include or park).

AFML imbalance estimator: include only if [What is the AFML imbalance-bar threshold estimator versus current code?](02-afml-imbalance-estimator.md) confirms a real mismatch. Masters skill partition: include only if [What does Masters Assessing specify for the Trend/Bias/Skill partition?](01-masters-skill-partition.md) shows the notebook-verified formula is wrong.

Do **not** pull in Sharpe representation, 252 vs 256, delay-0, `gap_stress`, Ledoit–Wolf vs LSM, or Fisher-on-all-indicators.

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.

## Answer

**Already merged on `main` (do not redo):** truncation `n_truncated` (#24); forbid `bfill` on `as_daily_reindex` (#26); seal `source_ohlcv` during `calculate_signals` (#25); hedge-ratio pairs sizer, Itô \(-\frac12\sigma^2\) in `make_synthetic_ohlcv` only, CPCV `n_paths` binomial identity, imbalance EWMA of **per-bar** mean signed flow, optional triple-barrier high/low + same-bar both-hit → stop, opt-in resting `STOP_ORDER` (#27).

**AFML imbalance:** research confirmed a real EWMA-units mismatch; the bar-level estimator in #27 is the product fix. Do not reopen tick-flow concatenation.

**Masters Skill partition:** KEEP TTMTS `Skill = R_{\mathrm{unbiased}} - B_{\mathrm{orig}}` (notebook **D6**). The *Assessing* Ability line is a documented discrepancy, not a must-fix. See [What does Masters Assessing specify for the Trend/Bias/Skill partition?](01-masters-skill-partition.md).

**Still must-fix (now that D1–D12 are ruled), as task tickets:**

- Procyclical MTM sizers → [Size live sizers on cash, not MTM equity](26-cash-base-sizers.md) (D10)
- UTC-midnight drawdown breaker → [Session-close drawdown breaker](27-session-close-dd-breaker.md) (D11)
- Triple-barrier **default** high/low when OHLC exists → [Default triple-barrier to high/low](28-triple-barrier-default-hl.md) (D12)

**Park:** pandas hot-path / \(O(TN)\).

Sharpe representation, measured \(N_T\), delay-0, `gap_stress`, dual-track Sharpe, CPCV embargo length, and unadjusted+CA are no longer “non-controversial leftovers” — they have D-rulings and their own task tickets 19–25.

## Comments

- 2026-08-19 notebook plus merged PRs #24–#27.
