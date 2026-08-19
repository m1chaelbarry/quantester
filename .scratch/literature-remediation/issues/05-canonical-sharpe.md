# Which Sharpe representation is canonical for tearsheet, MCPT, and DSR?

Type: grilling
Status: resolved
Part of: [Literature remediation decision map](../map.md)

## Question

Which return representation is canonical for (a) the event-loop tearsheet, (b) the MCPT / fast-track objective, and (c) the DSR registry — **simple** \(r_t\), **log** \(\ell_t=\log(E_t/E_{t-1})\), or an explicit convert-on-the-boundary split?

Camps from synthesis §4.1 (do not implement until this ticket closes):

- Simple / TWRR / cost drag: AFML audit; Masters *SSML* Passage 269; Carver *Systematic Trading* (drag linear in simple SR).
- Log / symmetry / IID resampling: Masters *Assessing* pp. 32, 149–150; *Testing and Tuning* ch. 1.

Uncontested: `analytics/performance.py`, `montecarlo/fast_track.py`, and `visualization/static.py` must not silently mix representations. Dual-track honesty is [Dual-track: keep two engines, but must fast-track Sharpe match the tearsheet function?](14-dual-track-sharpe-parity.md).

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.

## Answer

**Simple** \(r_t = E_t/E_{t-1}-1\) is canonical for the event-loop tearsheet, so Carver cost drag stays linear in Sharpe units (*Systematic Trading* ch. 12/15). Fast-track already uses simple — keep that representation. DSR / `auto_register_from_equity` must ingest that same simple tearsheet SR (today they take log moments from `annualized_sharpe` / `log_returns`). Visualization rolling Sharpe stays simple (`pct_change`, `ddof=1`).

Masters log returns remain a **documented exception** for stat-arb / HFT permutation tests (MCPT still shuffles log price changes). They are not the engine tearsheet default. Do not convert-on-the-boundary inside `annualized_sharpe`; MCPT and cash/tearsheet stay two documented representations.

Implement: [Switch tearsheet Sharpe to simple returns](19-simple-tearsheet-sharpe.md). Dual-track function parity is [Fast-track Sharpe must call annualized_sharpe](23-fast-track-sharpe-parity.md) and is blocked on this change — calling today’s log `annualized_sharpe` from fast-track would regress fast-track onto log.

## Comments

- 2026-08-19 notebook ruling **D1** (Carver *Systematic Trading* ch. 12/15). Product call, not a re-read of Masters *Assessing*.
