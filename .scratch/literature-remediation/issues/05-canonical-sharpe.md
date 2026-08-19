# Which Sharpe representation is canonical for tearsheet, MCPT, and DSR?

Type: grilling
Status: open
Part of: [Literature remediation decision map](../map.md)

## Question

Which return representation is canonical for (a) the event-loop tearsheet, (b) the MCPT / fast-track objective, and (c) the DSR registry — **simple** \(r_t\), **log** \(\ell_t=\log(E_t/E_{t-1})\), or an explicit convert-on-the-boundary split?

Camps from synthesis §4.1 (do not implement until this ticket closes):

- Simple / TWRR / cost drag: AFML audit; Masters *SSML* Passage 269; Carver *Systematic Trading* (drag linear in simple SR).
- Log / symmetry / IID resampling: Masters *Assessing* pp. 32, 149–150; *Testing and Tuning* ch. 1.

Uncontested: `analytics/performance.py`, `montecarlo/fast_track.py`, and `visualization/static.py` must not silently mix representations. Dual-track honesty is [Dual-track: keep two engines, but must fast-track Sharpe match the tearsheet function?](14-dual-track-sharpe-parity.md).

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.
