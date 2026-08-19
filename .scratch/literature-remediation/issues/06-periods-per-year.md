# What is the canonical periods-per-year and cash day-count policy?

Type: grilling
Status: resolved
Part of: [Literature remediation decision map](../map.md)
Blocked by: 03

## Question

What is the product policy for annualization and cash day-count once [What does Carver prescribe for 256 business days versus measured frequency?](03-carver-256.md) is a fact?

Options the synthesis isolates (§4.2):

- Carver **256** / \(\sqrt{256}=16\) as the default.
- Measured \(N_T\) (Chan hourly NYSE 1638, crypto 8760, etc.).
- Inferred from median \(\Delta t\) of the index (US daily equities ≈ 252, **not** 256).
- Explicit `periods_per_year` (and a separate cash day-count, today 365 simple) per instrument calendar, with a documented default.

Uncontested: stop silently applying `TRADING_DAYS = 252` to non-daily indexes. The default is a product decision, not a theorem.

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.

## Answer

**Metrics** annualize with the series’ measured \(N_T\) (Chan *Quantitative Trading* ch. 3: hourly NYSE \(252\times 6.5\), crypto 8760, etc.). Stop applying `TRADING_DAYS = 252` to non-daily indexes. Carver 256 is a daily convenience so \(\sqrt{P}=16\), not the product default — already a fact in [What does Carver prescribe for 256 business days versus measured frequency?](03-carver-256.md).

**Cash yield KEEP `/365`** simple day-count. Do not fold cash accrual into the metrics scalar.

Implement: [Annualize with measured periods-per-year](20-measured-periods-per-year.md).

## Comments

- 2026-08-19 notebook ruling **D2** (Chan QT ch. 3). Cash-yield `/365` is KEEP, not a Chan claim.
