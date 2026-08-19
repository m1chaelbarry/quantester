# Adjusted total-return prices versus unadjusted plus corporate-action cash ledger?

Type: grilling
Status: open
Part of: [Literature remediation decision map](../map.md)

## Question

What is the research-data policy for splits and dividends: `auto_adjust=True` total-return-like OHLC with no cash events (today’s YFinance path), or `auto_adjust=False` plus a corporate-action / cash-dividend ledger — or both as documented modes?

Synthesis §4.12: Clenow / Peterson / Kaufman want total-return ranking vs cash booking. The synthesis says Yahoo is not “wrong”; the fix is policy. Survivorship / PIT / delist residuals are [Is point-in-time universe plus delist and CA in the first-wave spec?](16-pit-first-wave.md) and must not be collapsed into this ticket.

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.
