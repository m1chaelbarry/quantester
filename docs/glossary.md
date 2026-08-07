# Glossary

Terminology used across Quantester's docs, code, and reports.

## Engine mechanics

- **Event-driven architecture** — components communicate only by posting
  events (`Market → Signal → Order → Fill`) on a shared queue; no direct
  calls between modules.
- **Bar phases (`open` / `close`)** — every bar is processed twice. At the
  open phase only prior bars plus the current open print are visible; the
  full bar appears at the close phase.
- **State-Based Temporal Firewall** — the enforced guarantee that decisions
  use only data available at their simulated time. Implemented by the two
  phases plus `earliest_fill_time` on orders — not by a hardcoded T+1 rule.
- **`delay`** — bars a strategy waits before its orders may execute. `1` =
  signal at close T, fill at open T+1. `0` = signal and fill at bar T's open
  under the intra-bar guard.
- **`earliest_fill_time`** — timestamp stamped on each order by the portfolio
  (`signal.timestamp + delay` on the master calendar); the execution ledger
  cannot fill before it.
- **Pending-order ledger** — the execution handler's parking lot for orders
  whose time has not come; retried at every open phase.
- **Availability mask** — when a symbol has no bar at a master-calendar
  timestamp it is served as `None` (untradeable). The timestamp is never
  deleted from other symbols.
- **Master calendar** — the outer-join union of all symbols' timestamps.
- **Vectorized twin** — a strategy's closed-form, full-history target-position
  function (`vectorized_signals`), numerically identical to its event form;
  required for Monte Carlo fast-track validation.
- **Fast-track parity** — the tested contract that the vectorized backtest
  and the event engine produce identical equity curves under identical
  sizing and costs.

## Costs & execution

- **cₜ (proportional costs)** — commissions/fees: `fixed + per_share × qty`.
  Charged to cash separately.
- **φₜ (implementation shortfall)** — spread crossing + volatility slippage +
  market impact, embedded in `fill_price`; recorded for analytics, never
  double-charged.
- **Half-spread** — `price × spread_pct / 2`; the price of taking liquidity.
- **Kaufman slippage** — slippage proportional to the bar's range
  `(high − low)/close`.
- **Kyle's lambda** — market impact `dp = λ·dx`; here λ is an Amihud-style
  illiquidity coefficient rising with volatility and falling with volume.
- **Gap-through** — when the open jumps past a stop price; the stop fills at
  the open (next available price), never at the guaranteed stop.

## Portfolio & sizing

- **Sizer** — callable mapping `(signal, portfolio, ref_price)` to a target
  quantity; where "how much" is decided.
- **Kelly fraction** — `f* = p − q/b` (binary) or `f = μ/σ²` (Gaussian);
  growth-optimal bet fraction.
- **Volatility parity** — weights `w_i ∝ 1/σ_i`; equal risk contribution.
- **Optimal-f (Vince)** — the fraction maximizing the Terminal Wealth
  Relative `TWR(f) = Π (1 + f·(−Trade_i/WorstLoss))`; Quantester stresses
  WorstLoss below the nominal stop (`gap_stress`) for gap risk.
- **Spectral risk attribution** — decomposing portfolio variance onto
  principal components of a Ledoit-Wolf-stabilized covariance:
  `R_n = β_n²Λ_nn/σ²`.
- **Margin monitor** — liquidates a fraction of all positions when
  `gross_exposure/equity` exceeds `max_leverage`.

## Validation & statistics

- **Sharpe ratio (annualized)** — `(μ_daily − Rf)/σ_daily × √252` on log
  returns.
- **Max drawdown / Calmar** — worst peak-to-trough loss; annualized return
  divided by it.
- **Carver speed limit** — cost drag above ~0.08 SR/yr means turnover is
  consuming the edge.
- **Truncation test** — Chan's leak detector: full vs truncated runs must
  produce identical overlapping positions.
- **Purged k-fold / embargo** — CV that drops training samples whose label
  interval overlaps the test interval (and a post-test buffer), because
  financial labels overlap in time.
- **CPCV** — Combinatorial Purged CV: N groups, all `C(N, N−k)` test
  combinations, `φ[N,k] = (k/N)·C(N, N−k)` backtest paths.
- **PBO** — Probability of Backtest Overfitting (Bailey–de Prado CSCV): how
  often the in-sample-best trial ranks below median out-of-sample. Gate:
  `< 0.10`.
- **PSR / DSR** — Probabilistic / Deflated Sharpe Ratio: probability the true
  Sharpe exceeds a benchmark / the expected best of N null trials.
- **Trials Registry** — the SQLite log of every optimization trial; the
  honest N and σ²_SR that DSR requires.
- **MCPT** — Monte Carlo Permutation Testing: retrain on shuffled log-change
  paths; the original must beat ≥ 95% of permutations (Masters' p-value:
  count starts at 1).
- **Trend/Bias/Skill partition** — Masters' decomposition:
  `Bias = R_perm − B_perm`; `Skill = (R_orig − Bias) − B_orig`;
  `Trend = B_orig` (the benchmark return).
- **Double bootstrap DD bound** — nested resampling giving a conservative
  "bound on a bound" for maximum drawdown (inner: sequencing; outer:
  sampling error of the dataset itself).
- **Autocorrelation gate** — runs test + Ljung-Box; if significant, iid
  resampling is invalid (Kaufman's autocorrelation trap) — use block
  bootstrap or OU paths.
- **Ornstein-Uhlenbeck (O-U) process** — mean-reverting synthetic paths
  `dP = θ(μ − P)dt + σdW`; OTR sweeps calibrate stop/take-profit grids over
  the ensemble instead of one realized path.

## Data

- **Dollar bars** — sample a bar per fixed amount of dollars traded.
- **Tick/volume/dollar imbalance bars (TIB/VIB/DIB)** — sample when the
  signed trade-flow imbalance exceeds its EWMA-expected value.
- **ETF trick** — de Prado's recurrence pricing a multi-product basket as a
  $1 total-return index `K_t`, with rebalancing costs `c_t` kept strictly
  external (a negative dividend).
- **Meta-labeling / triple barrier** — a primary model decides side; a
  secondary classifier trained on triple-barrier outcomes (TP / SL / vertical
  barrier) predicts P(correct) and scales size.
