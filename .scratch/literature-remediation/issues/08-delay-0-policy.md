# Keep delay-0 as a firewall feature, or require minimum latency?

Type: grilling
Status: resolved
Part of: [Literature remediation decision map](../map.md)

## Question

Keep `delay=0` (fill at bar \(T\)’s open under the intra-bar guard: data strictly before the fill timestamp) as an explicit Temporal Firewall feature, or require a minimum 1-bar / millisecond latency because same-print fills are unphysical?

Camps from synthesis §4.6:

- Keep: Quantester firewall; AFML-style overnight mean-reversion at the open.
- Delete or gate: Harris bilateral-search latency; Peterson NFP/TRMI timing.

This is a simulation-fidelity switch, not a coding error. It is orthogonal to delay-1 entries vs resting stops.

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.

## Answer

**Forbid delay-0 by default** (`delay >= 1`). Same-print fills are unphysical under Harris bilateral-search latency (ch. 10/22). Keep the Temporal Firewall **code path** (intra-bar guard, open-phase data strictly before the fill timestamp) behind an explicit engine/backtest opt-in such as `allow_same_print_fills=True` for authorized research (overnight mean-reversion at the open). `Strategy.delay` default stays 1.

Do not delete the firewall. Do not treat delay-0 as a live-replicable default.

Implement: [Forbid delay-0 fills by default](22-forbid-delay-0-default.md).

## Comments

- 2026-08-19 notebook ruling **D4** (Harris ch. 10/22). Orthogonal to delay-1 entries vs resting stops.
