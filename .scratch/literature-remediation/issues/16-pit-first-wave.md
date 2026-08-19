# Is point-in-time universe plus delist and CA in the first-wave spec?

Type: grilling
Status: resolved
Part of: [Literature remediation decision map](../map.md)
Blocked by: 13

## Question

After [Adjusted total-return prices versus unadjusted plus corporate-action cash ledger?](13-adjusted-vs-ca-cash.md): is a point-in-time constituent file, delist/halt residual, and `CorporateActionEvent` cash booking **in the first-wave spec**, or deferred until a dataset exists?

Synthesis §1.4 (Chan Example 3.3 survivorship) is a real bias, but ingestion schema and vendor are still fog. This ticket only rules **in or out of the spec**, not the vendor.

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.

## Answer

**CA / dividend cash is in the first wave.** Unadjusted OHLC + dividend cash credits + split quantity adjustments — [Unadjusted Yahoo prices plus dividend cash](25-unadjusted-dividend-cash.md).

**PIT constituent file, delist, and halt residuals are deferred** until a dataset and vendor schema exist. Survivorship bias is real (Chan Example 3.3); it is not implementable without fog-clearing on the ingest contract. Do not invent a fake PIT universe.

## Comments

- 2026-08-19 notebook ruling **D9** split: cash CA in; PIT universe out.
