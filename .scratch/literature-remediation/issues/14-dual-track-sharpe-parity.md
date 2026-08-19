# Dual-track: keep two engines, but must fast-track Sharpe match the tearsheet function?

Type: grilling
Status: resolved
Part of: [Literature remediation decision map](../map.md)
Blocked by: 05

## Question

After [Which Sharpe representation is canonical for tearsheet, MCPT, and DSR?](05-canonical-sharpe.md): keep the dual architecture (event-loop vs vectorized fast-track with a documented subset contract), and must fast-track Sharpe call the **same function** as the tearsheet on the same equity series (parity mode)?

Synthesis §4.9: Hilpisch rates documenting the split 🟢; AFML / Chan / Carver / Masters rate metric divergence 🔴. Uncontested suggestion: two engines, one Sharpe function, convert explicitly if MCPT needs a different representation.

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.

## Answer

**Keep two engines** (event-loop vs vectorized fast-track with the documented subset contract). Fast-track Sharpe **must call** `annualized_sharpe` on the same equity series (Hilpisch ch. 12). No private `pct_change` × \(\sqrt{252}\) in `FastResult.sharpe`.

**Order:** implement [Switch tearsheet Sharpe to simple returns](19-simple-tearsheet-sharpe.md) and [Annualize with measured periods-per-year](20-measured-periods-per-year.md) **first**. Calling today’s log `annualized_sharpe` from fast-track would switch fast-track **to log** and undo D1.

Implement: [Fast-track Sharpe must call annualized_sharpe](23-fast-track-sharpe-parity.md).

## Comments

- 2026-08-19 notebook ruling **D7** (Hilpisch ch. 12). Tension with D1 is sequencing, not a formula fight.
