# Which architectural gaps are first-wave core versus parked specialty toolkits?

Type: grilling
Status: open
Part of: [Literature remediation decision map](../map.md)
Blocked by: 04, 10

## Question

From synthesis §2, which architectural gaps belong on the **first implementation wave** (core engine / research loop) versus **parked specialty toolkits** (out of this spec’s build list, not necessarily forever out of the product)?

Use [What bootstrap methods remain missing after the existing stationary bootstrap?](04-stationary-bootstrap-gap.md) so stationary bootstrap is not re-implemented from a false-negative audit. Use [Live allocation: Vince LSM, AFML covariance library, or both with an explicit live-sizer policy?](10-live-allocation-philosophy.md) so “wire Kelly to every fill” is not assumed.

Candidates to sort (not pre-sliced into tickets): Carver vol-target / FDM / inertia; walk-forward / NTEST–EXTRA; Ehlers cycle stack; Clenow \(\mathrm{slope}\times R^2\); Harris TCA; Hilpisch HDF5/sockets; Sortino / round-trip PF; cross-section fractile; MI/mRMR; ensembles/GRNN; DC/HMM; TRMI parser; Vince LSM extras; Brunton LQR/SINDYc; Ruggiero equity-curve feedback.

A book’s 🔴 on a missing specialty method does not auto-promote it to first wave.

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.
