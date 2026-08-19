# Live allocation: Vince LSM, AFML covariance library, or both with an explicit live-sizer policy?

Type: grilling
Status: resolved
Part of: [Literature remediation decision map](../map.md)

## Question

What is the live allocation philosophy relative to the research libraries (Kelly, Vince \(f^*\), Ledoit–Wolf / HRP, vol-parity)?

Camps from synthesis §4.7:

- Vince *Leverage Space* / Ruggiero: discard \(\Sigma\), use joint scenarios; couple \(f^*\) to every fill.
- AFML / Chan / Vince *MoMM* (as a gap, not a bug): the library may stay adjacent; the live sizer is a separate policy. Default today is `PercentEquitySizer(0.5)` / one-call `0.9`.
- AFML / Brunton: Ledoit–Wolf spectral risk is correct for ill-conditioned \(\Sigma\).

Do not rip out Ledoit–Wolf because Vince LSM rejects MPT. Which gaps belong in the first wave is [Which architectural gaps are first-wave core versus parked specialty toolkits?](15-first-wave-gaps.md).

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.

## Answer

**KEEP adjacent live sizers.** Vince LSM / couple \(f^*\) to every fill is **not** first-wave core. Ledoit–Wolf stays the covariance **library** (spectral risk), not a competing live-allocation religion. Default remains a percent sizer; the first-wave defect is **MTM equity as the sizing base** (procyclical), ruled in [Size live sizers on cash, not MTM equity](26-cash-base-sizers.md) (D10), not “wire Kelly/LSM into `PortfolioManager`.”

Park LSM-on-every-fill, HRP-as-default, and Brunton LQR with the specialty toolkits in [Which architectural gaps are first-wave core versus parked specialty toolkits?](15-first-wave-gaps.md).

## Comments

- 2026-08-19 notebook ruling **D5** (AFML ch. 10/16). KEEP; no dedicated implementation ticket beyond D10’s cash-base change.
