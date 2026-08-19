# Literature remediation decision map

## Destination

Locked: [`spec.md`](spec.md). Grilling and research tickets on this map are **resolved**. Remaining open work is **task** tickets 19–28 for a later implementation effort — do not re-litigate D1–D12 while claiming those tasks.

## Notes

- Domain: Quantester event-driven backtester; literature remediation after the third cross-reference dump.
- Canonical merge: [`3rd Cross Reference Synthesis.md`](../../3rd%20Cross%20Reference%20Synthesis.md). Raw per-book audits: [`3rd Cross Reference.md`](../../3rd%20Cross%20Reference.md). Engine survey: [`SYSTEM_ARCHITECTURE_QUANTITATIVE_METHODOLOGY_BLUEPRINT.md`](../../SYSTEM_ARCHITECTURE_QUANTITATIVE_METHODOLOGY_BLUEPRINT.md).
- Invariants: [`.cursor/rules/quantester-core.mdc`](../../.cursor/rules/quantester-core.mdc), [`.cursor/rules/quant-literature-notebook.mdc`](../../.cursor/rules/quant-literature-notebook.mdc). Glossary: [`CONTEXT.md`](../../CONTEXT.md). Tracker: [`docs/agents/issue-tracker.md`](../../docs/agents/issue-tracker.md).
- Notebook rulings D1–D12 (2026-08-19) are the product calls. Do not reopen §3 alignments unless a conflict ticket names them: ETF-trick cash stays external to `K_t`; MCPT Protocol I uses the same shuffle index across assets; delay-1 **entries** after the close remain the default live-replicable path.
- A book's 🔴 on a missing specialty method is not automatically an engine bug.
- Refer to tickets by **title wrapping the path**, never by a bare number.

## Decisions so far

- [What does Masters Assessing specify for the Trend/Bias/Skill partition?](issues/01-masters-skill-partition.md) — *Assessing* Ability subtracts \(B_{\mathrm{perm}}\); repo Skill subtracts \(B_{\mathrm{orig}}\) and matches TTMTS. **Product KEEP TTMTS** (D6).
- [What is the AFML imbalance-bar threshold estimator versus current code?](issues/02-afml-imbalance-estimator.md) — EWMA units were ticks not bars; bar-level estimator shipped in #27.
- [What does Carver prescribe for 256 business days versus measured frequency?](issues/03-carver-256.md) — 256 is a daily convenience so \(\sqrt{P}=16\), not a universal or measured \(N_T\).
- [What bootstrap methods remain missing after the existing stationary bootstrap?](issues/04-stationary-bootstrap-gap.md) — SB already exists; TBB and automatic \(b_{\mathrm{SB}}\) stay parked.
- [Which Sharpe representation is canonical for tearsheet, MCPT, and DSR?](issues/05-canonical-sharpe.md) — **simple** tearsheet / DSR ingest; viz rolling KEEP simple; Masters log = MCPT exception (D1).
- [What is the canonical periods-per-year and cash day-count policy?](issues/06-periods-per-year.md) — measured \(N_T\) for metrics; cash yield KEEP `/365` (D2).
- [Are delay-1 market entries and resting intra-bar stops orthogonal policies?](issues/07-delay-entries-vs-stops.md) — yes; resting STOP opt-in already shipped.
- [Keep delay-0 as a firewall feature, or require minimum latency?](issues/08-delay-0-policy.md) — forbid delay-0 **by default**; keep the firewall behind opt-in (D4).
- [Keep Vince gap_stress, use realized gap-through W, or drop the 1.5×?](issues/09-vince-gap-stress.md) — raw BiggestLoss; `gap_stress` default 1.0 (D3).
- [Live allocation: Vince LSM, AFML covariance library, or both with an explicit live-sizer policy?](issues/10-live-allocation-philosophy.md) — KEEP adjacent sizers; Ledoit–Wolf stays library (D5).
- [Must the engine require stops, forbid them, or support both families?](issues/11-stops-required-vs-forbidden.md) — both families; Donchian/tranche opt-in shipped.
- [Is CPCV overlap geometry enough, or must embargo length be lookback/lookahead bars?](issues/12-cpcv-embargo-length.md) — keep geometry; integer B/F bars (D8; Assessing ch. 1 not in notebook).
- [Adjusted total-return prices versus unadjusted plus corporate-action cash ledger?](issues/13-adjusted-vs-ca-cash.md) — unadjusted Yahoo + dividend cash (D9).
- [Dual-track: keep two engines, but must fast-track Sharpe match the tearsheet function?](issues/14-dual-track-sharpe-parity.md) — keep two engines; fast-track must call `annualized_sharpe` **after** D1/D2 (D7).
- [Which architectural gaps are first-wave core versus parked specialty toolkits?](issues/15-first-wave-gaps.md) — first wave = D1–D12 tasks; DC/HMM/TRMI/LSM-on-every-fill parked.
- [Is point-in-time universe plus delist and CA in the first-wave spec?](issues/16-pit-first-wave.md) — CA/dividend cash in; PIT universe/delist deferred.
- [Which non-controversial critical defects are must-fix on the spec?](issues/17-noncontroversial-must-fix.md) — #24–#27 shipped; leftovers are D10/D11/D12 tasks; pandas hot-path parked.
- [What is the ordered implementation spec after the rulings?](issues/18-ordered-implementation-spec.md) — [`spec.md`](spec.md); tasks 19–28.

## Not yet specified

- PIT vendor and constituent-file schema (deferred; not first wave).
- Full exchange holiday calendar for the drawdown breaker (ticket 27 parameterizes `day_roll_time` + tz as the substitute).
- Walk-forward / NTEST–EXTRA product shape (parked).
- Hilpisch broker sockets / HDF5 live storage (parked).
- Whether `PercentEquitySizer` is renamed after ticket 26 (API-stable `base="cash"` is the spec).

## Out of scope

- Re-synthesizing the third cross-reference.
- Reopening §3 aligned items except where a conflict ticket explicitly names them.
- Implementing specialty toolkits listed as parked in [`spec.md`](spec.md).
- Migrating this map onto GitHub Issues in this session (agents here cannot create issues).
- Claiming task tickets 19–28 on this planning pass — they stay `Status: open` for the implementation effort.
