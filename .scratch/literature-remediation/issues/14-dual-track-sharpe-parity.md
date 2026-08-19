# Dual-track: keep two engines, but must fast-track Sharpe match the tearsheet function?

Type: grilling
Status: open
Part of: [Literature remediation decision map](../map.md)
Blocked by: 05

## Question

After [Which Sharpe representation is canonical for tearsheet, MCPT, and DSR?](05-canonical-sharpe.md): keep the dual architecture (event-loop vs vectorized fast-track with a documented subset contract), and must fast-track Sharpe call the **same function** as the tearsheet on the same equity series (parity mode)?

Synthesis §4.9: Hilpisch rates documenting the split 🟢; AFML / Chan / Carver / Masters rate metric divergence 🔴. Uncontested suggestion: two engines, one Sharpe function, convert explicitly if MCPT needs a different representation.

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.
