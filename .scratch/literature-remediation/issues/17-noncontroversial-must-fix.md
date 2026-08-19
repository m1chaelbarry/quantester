# Which non-controversial critical defects are must-fix on the spec?

Type: grilling
Status: open
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
