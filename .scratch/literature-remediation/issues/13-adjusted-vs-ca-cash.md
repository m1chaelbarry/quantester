# Adjusted total-return prices versus unadjusted plus corporate-action cash ledger?

Type: grilling
Status: resolved
Part of: [Literature remediation decision map](../map.md)

## Question

What is the research-data policy for splits and dividends: `auto_adjust=True` total-return-like OHLC with no cash events (today’s YFinance path), or `auto_adjust=False` plus a corporate-action / cash-dividend ledger — or both as documented modes?

Synthesis §4.12: Clenow / Peterson / Kaufman want total-return ranking vs cash booking. The synthesis says Yahoo is not “wrong”; the fix is policy. Survivorship / PIT / delist residuals are [Is point-in-time universe plus delist and CA in the first-wave spec?](16-pit-first-wave.md) and must not be collapsed into this ticket.

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.

## Answer

**Unadjusted ledger.** Yahoo `auto_adjust=False` by default. Book dividends as **cash credits** on the portfolio (Peterson *Developing & Testing* ch. 11), not as total-return OHLC. Splits adjust **quantity**, not the price series used for fills. A documented `auto_adjust=True` total-return mode may remain for ranking research; it is not the default backtest path.

Point-in-time universe membership and delist residuals stay **out** of the first wave — [Is point-in-time universe plus delist in the first-wave spec?](16-pit-first-wave.md).

Implement: [Unadjusted Yahoo prices plus dividend cash](25-unadjusted-dividend-cash.md).

## Comments

- 2026-08-19 notebook ruling **D9** (Peterson ch. 11). CA/dividend stream is first-wave; PIT vendor is not.
