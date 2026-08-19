# Literature remediation decision map

## Destination

A locked decision spec — not the engine patches themselves — that a later implementation effort can follow without re-litigating [`3rd Cross Reference Synthesis.md`](../../3rd%20Cross%20Reference%20Synthesis.md): how each §4 literature conflict is ruled; which §1 critical findings are in-scope engine defects versus specialty-scope; which §2 architectural gaps belong on the first implementation wave and which stay parked.

## Notes

- Domain: Quantester event-driven backtester; literature remediation after the third cross-reference dump.
- This map **plans**; it does not implement. Tickets resolve decisions. Implementation is a later effort.
- Canonical merge: [`3rd Cross Reference Synthesis.md`](../../3rd%20Cross%20Reference%20Synthesis.md). Raw per-book audits: [`3rd Cross Reference.md`](../../3rd%20Cross%20Reference.md). Engine survey: [`SYSTEM_ARCHITECTURE_QUANTITATIVE_METHODOLOGY_BLUEPRINT.md`](../../SYSTEM_ARCHITECTURE_QUANTITATIVE_METHODOLOGY_BLUEPRINT.md).
- Invariants: [`.cursor/rules/quantester-core.mdc`](../../.cursor/rules/quantester-core.mdc), [`.cursor/rules/quant-literature-notebook.mdc`](../../.cursor/rules/quant-literature-notebook.mdc). Glossary: [`CONTEXT.md`](../../CONTEXT.md). Tracker: [`docs/agents/issue-tracker.md`](../../docs/agents/issue-tracker.md).
- Every session: `/grilling` and `/domain-modeling`. Research tickets: `/research` against primary sources (code, books, notebook), not the synthesis alone.
- Do not reopen §3 alignments unless a conflict ticket names them: ETF-trick cash stays external to `K_t`; MCPT Protocol I uses the same shuffle index across assets; delay-1 **entries** after the close remain the default live-replicable path.
- A book's 🔴 on a missing specialty method is not automatically an engine bug.
- Refer to tickets by **title wrapping the path**, never by a bare number.

## Decisions so far

- [What does Masters Assessing specify for the Trend/Bias/Skill partition?](issues/01-masters-skill-partition.md) — *Assessing* Ability subtracts \(B_{\mathrm{perm}}\); repo Skill subtracts \(B_{\mathrm{orig}}\) and matches TTMTS, not *Assessing*.
- [What is the AFML imbalance-bar threshold estimator versus current code?](issues/02-afml-imbalance-estimator.md) — product-shape after warmup, but EWMA of tick flows vs bar-level signed size; leftover flush and constant warmup are extra.
- [What does Carver prescribe for 256 business days versus measured frequency?](issues/03-carver-256.md) — 256 is a daily convenience so \(\sqrt{P}=16\), not a universal or measured \(N_T\).
- [What bootstrap methods remain missing after the existing stationary bootstrap?](issues/04-stationary-bootstrap-gap.md) — SB already exists; remaining are TBB and automatic \(b_{\mathrm{SB}}\).

## Not yet specified

- PIT vendor and constituent-file schema (only if [Is point-in-time universe plus delist and CA in the first-wave spec?](issues/16-pit-first-wave.md) says yes).
- Whether a follow-on map carries execution or a separate implementation PR starts once this map has no open tickets.
- Notebook re-verification batch of remaining formulas after the conflict rulings.
- Walk-forward / NTEST–EXTRA product shape (depends on [Which architectural gaps are first-wave core versus parked specialty toolkits?](issues/15-first-wave-gaps.md)).
- `periods_per_year` API sketch (depends on [What is the canonical periods-per-year and cash day-count policy?](issues/06-periods-per-year.md)).
- Hilpisch broker sockets / HDF5 live storage — likely specialty, not yet a ticket.
- `source_ohlcv` permission model details (depends on [Which non-controversial critical defects are must-fix on the spec?](issues/17-noncontroversial-must-fix.md)).

## Out of scope

- Implementing or patching engine, analytics, or strategy code on this map.
- Re-synthesizing the third cross-reference.
- Reopening §3 aligned items (ETF-trick cash, MCPT Protocol I, delay-1 entries, CSCV/PBO, cash yield on positive cash, limit gap improvement, MOC all-or-nothing, drawdown double bootstrap) except where a conflict ticket explicitly names them.
- Migrating this map onto GitHub Issues in this session (agents here cannot create issues).
