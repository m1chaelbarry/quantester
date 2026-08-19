# What does Carver prescribe for 256 business days versus measured frequency?

Type: research
Status: open
Part of: [Literature remediation decision map](../map.md)

## Question

What does Robert Carver, *Systematic Trading*, actually prescribe for annualizing volatility and Sharpe — the **256** business-day / \(\sqrt{256}=16\) convention — and does he present it as a universal constant, a UK/US equity calendar convenience, or a default that should yield to the instrument’s actual periods per year?

Chan and others in the synthesis want measured \(N_T\) (e.g. hourly NYSE \(252\times 6.5=1638\)). Median \(\Delta t\) on US daily equities is ~252, not 256. Cash yield in this repo uses 365-day simple interest.

Resolve the **Carver fact** from the book (or an honest “not available”), not the product default. That default is [What is the canonical periods-per-year and cash day-count policy?](06-periods-per-year.md).

Write findings to [`../research/03-carver-256.md`](../research/03-carver-256.md).
