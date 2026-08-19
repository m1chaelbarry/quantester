# Which architectural gaps are first-wave core versus parked specialty toolkits?

Type: grilling
Status: resolved
Part of: [Literature remediation decision map](../map.md)
Blocked by: 04, 10

## Question

From synthesis §2, which architectural gaps belong on the **first implementation wave** (core engine / research loop) versus **parked specialty toolkits** (out of this spec’s build list, not necessarily forever out of the product)?

Use [What bootstrap methods remain missing after the existing stationary bootstrap?](04-stationary-bootstrap-gap.md) so stationary bootstrap is not re-implemented from a false-negative audit. Use [Live allocation: Vince LSM, AFML covariance library, or both with an explicit live-sizer policy?](10-live-allocation-philosophy.md) so “wire Kelly to every fill” is not assumed.

Candidates to sort (not pre-sliced into tickets): Carver vol-target / FDM / inertia; walk-forward / NTEST–EXTRA; Ehlers cycle stack; Clenow \(\mathrm{slope}\times R^2\); Harris TCA; Hilpisch HDF5/sockets; Sortino / round-trip PF; cross-section fractile; MI/mRMR; ensembles/GRNN; DC/HMM; TRMI parser; Vince LSM extras; Brunton LQR/SINDYc; Ruggiero equity-curve feedback.

A book’s 🔴 on a missing specialty method does not auto-promote it to first wave.

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.

## Answer

**First-wave core** is the notebook D1–D12 product calls (task tickets 19–28), plus the already-merged uncontroversial engine defects (truncation `n_truncated`, forbid `bfill`, seal `source_ohlcv`, hedge-ratio sizer, Itô GBM, CPCV `n_paths`, imbalance bar-level EWMA, optional TB high/low, resting STOP opt-in).

**Parked specialty toolkits** (out of this spec’s build list): DC/HMM; TRMI parser; Vince LSM extras / couple \(f^*\) to every fill; Brunton LQR/SINDYc; Ehlers cycle stack; Harris TCA; Hilpisch HDF5/sockets; MI/mRMR; ensembles/GRNN; Ruggiero equity-curve feedback; TBB and automatic \(b_{\mathrm{SB}}\) ([What bootstrap methods remain missing after the existing stationary bootstrap?](04-stationary-bootstrap-gap.md)); walk-forward / NTEST–EXTRA product shape; Clenow \(\mathrm{slope}\times R^2\) as a new strategy; Carver FDM/inertia beyond cash-base sizing; Sortino / round-trip PF extras; pandas hot-path rewrite; Fisher-on-all-indicators.

Carver vol-target as a **library** stays adjacent (D5). First-wave vol-related work is only [Size live sizers on cash, not MTM equity](26-cash-base-sizers.md), not a full FDM stack.

## Comments

- 2026-08-19 notebook: first wave = D1–D12 tasks; specialty 🔴s do not auto-promote.
