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

### Features and selection

**OHLCV+TA nest**:
Feature set built from open/high/low/close/volume plus technical indicators and
lags — without EGARCH volatility-model features.
_Avoid_: OHLCV+TA+EGARCH (v1), full paper feature tiers

**Optuna fold search**:
Per walk-forward fold, an automated hyperparameter search (Optuna) over
XGBoost settings, scored on the validation window (paper: ~50 trials/fold).
_Avoid_: fixed defaults only, one global fit for all years

**Validation selector**:
Rule that picks which Optuna trial becomes the fold’s deployed model using
validation-only scores — loss-best (lowest forecast error), IC-best (highest
rank correlation of forecast vs realised return), or IR**-best (best
validation trading score with costs).
_Avoid_: peeking at test, picking by test Sharpe

### Execution assumptions (this recreation)

**Ledger-native ~10 bps friction**:
A Quantester cost model tuned so all-in trading friction is roughly ten basis
points of notional intent — not the paper’s flat spreadsheet `c·|Δpos|` formula.
_Avoid_: paper flat proportional TC, ignoring the fill ledger

**Delay-1 fill**:
Signal on bar T’s close; earliest fill at bar T+1’s open under the temporal
firewall.
_Avoid_: close-to-close paper return identity, delay-0 open fill for this study

**Availability mask**:
A missing bar makes the symbol untradeable at that timestamp; prices are not
forward-filled into the DataHandler.
_Avoid_: paper flat synthetic bars in the tradeable series

### Scoreboard

**Parity pass criteria**:
Cost-aware rule beats naive sign on consolidated out-of-sample net Sharpe,
beats buy-and-hold on the same window, truncation test passes, and
cost-stress remains non-disastrous.
_Avoid_: must hit SR>1 or ARC≈65% under Quantester accounting

### Packaging

**Separate research script**:
Runnable code under `examples/` that imports `quantester` but is not part of
the installable `quantester` package.
_Avoid_: core module, package feature
