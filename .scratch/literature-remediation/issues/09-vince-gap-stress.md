# Keep Vince gap_stress, use realized gap-through W, or drop the 1.5×?

Type: grilling
Status: open
Part of: [Literature remediation decision map](../map.md)

## Question

For Vince \(f^*\) / BiggestLoss \(W\): keep the repo `gap_stress` multiplier (default 1.5), replace \(W\) with **realized** gap-through PnL from the stop ledger, or drop the multiplier and use raw historical BiggestLoss as Vince p. 18?

Camps from synthesis §4.5:

- Vince: \(W\) = raw BiggestLoss; 1.5× is silent de-lever and distorts \(f^*\).
- Quantester invariant: perfect stops unbound \(f^*\); stops gap through, so historical min loss can be too small.

A documented conservative \(f\) input is allowed; a magic 1.5 without a ledger basis is the disputed part.

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.
