# Literature remediation — locked implementation spec

This is the destination artifact for [What is the ordered implementation spec after the rulings?](issues/18-ordered-implementation-spec.md). A later effort implements the **task** tickets; this file does not patch the engine.

Provenance: Gemini Notebook rulings D1–D12 (2026-08-19) plus research tickets 01–04. Canonical dump: [`3rd Cross Reference Synthesis.md`](../../3rd%20Cross%20Reference%20Synthesis.md) §5, rewritten under those rulings.

## Do not reopen

- ETF-trick \(c_t\) stays external to \(K_t\) (book as a negative dividend at strategy level, never embed in the spread).
- MCPT Protocol I: the same shuffle index across assets.
- Delay-1 **market entries** after the close remain the default live-replicable path.
- `fill_price` already embeds spread/slippage/impact; do not double-deduct `slippage_cost`.
- Stops fill at the next available price after gap-through, never a guaranteed stop print.
- All RNG via `numpy.random.Generator(seed)`.
- Queue-only lifecycle; Temporal Firewall two-phase bars.

## Already on `main` (do not redo)

PRs #24–#27: truncation `n_truncated`; forbid `bfill` on `as_daily_reindex`; seal `source_ohlcv` during `calculate_signals`; `HedgeRatioSizer`; Itô \(-\frac12\sigma^2\) in `make_synthetic_ohlcv` only; CPCV `n_paths = C(n_{\mathrm{groups}}-1, k_{\mathrm{test}}-1)`; imbalance EWMA of per-bar mean signed flow; optional triple-barrier high/low + same-bar both-hit → stop; opt-in resting `STOP_ORDER` (Donchian/tranche delay-1 entry). Delay-0 `stop_only` at close remains forbidden.

## KEEP (no task)

| ID | Call |
| --- | --- |
| D5 | Adjacent live sizers. Ledoit–Wolf stays the covariance library. No LSM-on-every-fill. |
| D6 | TTMTS `Skill = R_{\mathrm{unbiased}} - B_{\mathrm{orig}}`. *Assessing* Ability is a documented discrepancy. |
| — | Cash yield `/365`. Visualization rolling Sharpe already simple. Both stop families (opt-in resting STOP shipped). |

MCPT still shuffles **log** price changes. Tearsheet Sharpe is **simple**. That split is documented, not a convert-on-the-boundary of one function.

## First wave — implement in this order

| Order | Ticket | D | Notes |
| --- | --- | --- | --- |
| 1 | [Switch tearsheet Sharpe to simple returns](issues/19-simple-tearsheet-sharpe.md) | D1 | Unblocks 23. |
| 2 | [Annualize with measured periods-per-year](issues/20-measured-periods-per-year.md) | D2 | Unblocks 23. Cash `/365` KEEP. |
| 3 | [Fast-track Sharpe must call annualized_sharpe](issues/23-fast-track-sharpe-parity.md) | D7 | **Blocked by 19 and 20.** Calling pre-D1 `annualized_sharpe` would move fast-track to log. |
| * | [Default Vince gap_stress to 1.0](issues/21-vince-gap-stress-default.md) | D3 | Independent. |
| * | [Forbid delay-0 fills by default](issues/22-forbid-delay-0-default.md) | D4 | Independent. Keep firewall behind opt-in. |
| * | [Integer B/F-bar CPCV embargo](issues/24-cpcv-embargo-bars.md) | D8 | Independent. Weakest row: Assessing ch. 1 not in the notebook. |
| * | [Unadjusted Yahoo prices plus dividend cash](issues/25-unadjusted-dividend-cash.md) | D9 | Independent. PIT universe deferred. |
| * | [Size live sizers on cash, not MTM equity](issues/26-cash-base-sizers.md) | D10 | Independent. Vol-target stays library (D5). |
| * | [Session-close drawdown breaker](issues/27-session-close-dd-breaker.md) | D11 | Independent. Parameterize `day_roll_time`. |
| * | [Default triple-barrier to high/low](issues/28-triple-barrier-default-hl.md) | D12 | Independent. Close-only is opt-out. |

Rows marked `*` may proceed in parallel with 1–3 once claimed.

## Parked specialty (not this spec’s build list)

DC/HMM; TRMI parser; Vince LSM extras; Brunton LQR/SINDYc; Ehlers cycle stack; Harris TCA; Hilpisch HDF5/sockets; MI/mRMR; ensembles/GRNN; Ruggiero equity-curve feedback; TBB / automatic \(b_{\mathrm{SB}}\); walk-forward / NTEST–EXTRA; Clenow \(\mathrm{slope}\times R^2\) as a new strategy; Carver FDM/inertia beyond cash-base sizing; Sortino / round-trip PF extras; pandas hot-path rewrite; Fisher-on-all-indicators; PIT constituent file / delist / halt.

## Tensions recorded, not buried

- **D1 then D7:** implement simple tearsheet Sharpe before fast-track calls `annualized_sharpe`.
- **D8:** Assessing ch. 1 is the weakest authority; implement integer bars and mark the module “not covered by the notebook.”
- **D6 vs D1:** MCPT log shuffle vs simple tearsheet — keep both, document.
- **D10:** notebook said “smoothed cash / vol-target”; ticket 26 picks cash (+ optional EWMA) as the live base and leaves vol-target adjacent.
- **D9 / D11:** first-wave CA is dividend cash + split quantity; session calendar is `day_roll_time` + tz, not a full exchange holiday file.

## Verification status after implementation

Each patched module docstring must mark formulas **notebook-verified** vs **not covered by the notebook — implemented from \<canonical source\>**. D8 embargo length is the latter.
