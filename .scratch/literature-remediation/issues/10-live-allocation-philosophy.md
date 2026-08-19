# Live allocation: Vince LSM, AFML covariance library, or both with an explicit live-sizer policy?

Type: grilling
Status: open
Part of: [Literature remediation decision map](../map.md)

## Question

What is the live allocation philosophy relative to the research libraries (Kelly, Vince \(f^*\), Ledoit–Wolf / HRP, vol-parity)?

Camps from synthesis §4.7:

- Vince *Leverage Space* / Ruggiero: discard \(\Sigma\), use joint scenarios; couple \(f^*\) to every fill.
- AFML / Chan / Vince *MoMM* (as a gap, not a bug): the library may stay adjacent; the live sizer is a separate policy. Default today is `PercentEquitySizer(0.5)` / one-call `0.9`.
- AFML / Brunton: Ledoit–Wolf spectral risk is correct for ill-conditioned \(\Sigma\).

Do not rip out Ledoit–Wolf because Vince LSM rejects MPT. Which gaps belong in the first wave is [Which architectural gaps are first-wave core versus parked specialty toolkits?](15-first-wave-gaps.md).

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.
