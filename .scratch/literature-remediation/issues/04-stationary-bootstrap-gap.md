# What bootstrap methods remain missing after the existing stationary bootstrap?

Type: research
Status: open
Part of: [Literature remediation decision map](../map.md)

## Question

Audits in *Assessing…* and *SSML* claim stationary / tapered block bootstrap and Politis–White optimal \(b\) are absent. The synthesis flags a possible false negative: `quantester/montecarlo/synthetic.py::bootstrap_ohlcv` already implements a Politis–Romano-style stationary bootstrap, and there is a block-bootstrap path after an autocorrelation gate.

From **code as primary** plus the Politis–Romano / Politis–White / Paparoditis–Politis papers (not the audit dump): what is already present, and what remains (TBB, automatic \(b_{\mathrm{SB}}\), shape-preserving OHLC caveats, mean_block heuristic vs optimal \(b\))?

Do not recommend implementing a second stationary bootstrap.

Write findings to [`../research/04-stationary-bootstrap-gap.md`](../research/04-stationary-bootstrap-gap.md).
