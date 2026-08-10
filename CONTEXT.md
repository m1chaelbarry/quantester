# Quantester

Institutional-grade event-driven backtesting. Domain language for research
scripts and engine work that must stay consistent with the four-event
lifecycle and temporal firewall.

## Language

### Research goals

**Quantester-economic parity**:
A recreation succeeds when the studied rule improves *net* performance under
Quantester’s ledger, fills, and cost models versus a stated baseline on the
same data window — not when headline ARC/Sharpe match an external paper’s
tables.
_Avoid_: paper-number fidelity, table-matching, replicate their Sharpe

**Flagship configuration**:
The single primary experiment for a paper recreation: one model family, one
position mode, one cost-aware setting, one feature nest, and the paper’s
walk-forward fold design — before ablations or rival architectures.
_Avoid_: full paper clone, H1–H6 suite

### Trading rule

**Cost-aware filter**:
A position-update gate: change to the forecast-implied target only when
forecast magnitude exceeds λ times proportional cost times turnover;
otherwise hold the current position.
_Avoid_: Kakushadze haircut (sizing-layer edge shrink), meta-labeling

**Forecast strategy**:
A Quantester `Strategy` whose primary signal is a precomputed or online
one-step return forecast mapped through sign and optional cost-aware filter.
_Avoid_: meta-labeling as primary, secondary model

### Packaging

**Separate research script**:
Runnable code under `examples/` that imports `quantester` but is not part of
the installable `quantester` package.
_Avoid_: core module, package feature
