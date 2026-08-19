# What is the canonical periods-per-year and cash day-count policy?

Type: grilling
Status: open
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
