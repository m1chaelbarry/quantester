# Keep delay-0 as a firewall feature, or require minimum latency?

Type: grilling
Status: open
Part of: [Literature remediation decision map](../map.md)

## Question

Keep `delay=0` (fill at bar \(T\)’s open under the intra-bar guard: data strictly before the fill timestamp) as an explicit Temporal Firewall feature, or require a minimum 1-bar / millisecond latency because same-print fills are unphysical?

Camps from synthesis §4.6:

- Keep: Quantester firewall; AFML-style overnight mean-reversion at the open.
- Delete or gate: Harris bilateral-search latency; Peterson NFP/TRMI timing.

This is a simulation-fidelity switch, not a coding error. It is orthogonal to delay-1 entries vs resting stops.

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.
