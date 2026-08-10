# Paper analysis: Bysika & Ślepaczuk (2026) vs Quantester

**Paper:** Andrei Bysika, Robert Ślepaczuk — *Machine Learning-Based Bitcoin
Trading Under Transaction Costs: Evidence From Walk-Forward Forecasting*
(arXiv:2606.00060v1, 19 May 2026).

**Scope:** Map the paper’s claims, methods, and gaps onto Quantester’s event
engine, cost stack, validation gates, and crypto data path. This is a
codebase-context analysis, not a full replication plan.

---

## 1. One-line verdict

The paper’s central thesis — **hourly ML forecasts only become economically
useful when the trading rule itself is cost-aware** — matches Quantester’s
design philosophy (Carver cost drag, Kakushadze edge haircut, retail cost
stress). Quantester is **stricter on look-ahead, missing bars, and
anti-overfitting** than the paper’s vectorized WFO backtest, but **lacks** the
paper’s explicit `|forecast| > λ·cost` execution filter, rolling WFO scheduler,
primary return-forecast ML strategies, EGARCH features, turnover metric, and
paired circular block-bootstrap on return differentials.

---

## 2. What the paper does

| Dimension | Paper choice |
| --- | --- |
| Asset / freq | BTC/USDT USD-M futures, hourly OHLCV (Binance), ~70k bars, 2018–2026 |
| Target | One-step log return \(r_{t+1}=\ln(P_{t+1}/P_t)\) |
| Models | XGBoost, LSTM, iTransformer (one-step regression) |
| Features | OHLCV → +TA → +EGARCH (nested sets; fold-local selection) |
| CV | Non-anchored rolling WFO: 12m train / 3m val / 3m test, advance 3m → **27 folds** |
| Baseline rule | Sign → long-only \(\{0,1\}\) or long-short \(\{-1,1\}\) |
| Cost | Flat proportional \(c=10\) bps per unit turnover |
| Key mechanism | Cost-aware filter: trade only if \(\lvert\hat r_{t+1}\rvert > \lambda\, c\, \lvert pos^*_t - pos_{t-1}\rvert\) (main: \(\lambda=2\)) |
| Inference | Paired **circular** block-bootstrap on return differentials + Holm; fold diagnostics |

**Headline empirical claim:** Naive sign strategies die after 10 bps costs
because of turnover. With the cost-aware filter, selected long-only XGBoost
configs show ARC > 65% and SR > 1, but bootstrap tests often fail to establish
formal dominance vs B&H / rival architectures after multiple-testing correction.
Fold performance is highly regime-dependent (16/27 profitable folds). Validation
loss correlates weakly with OOS trading metrics — reinforcing the
prediction-to-PnL gap.

---

## 3. Alignment with Quantester (strong)

### 3.1 Prediction ≠ economic PnL

Paper §1–2: MSE/MAE/directional accuracy do not measure net trading value.
Quantester encodes the same principle:

- Execution costs live in `quantester/execution/costs.py` and are applied in the
  fill path (`SimulatedExecutionHandler`), not as a post-hoc spreadsheet haircut.
- `carver_cost_drag_sr` / `speed_limit_warning` (`analytics/performance.py`)
  flag when turnover consumes edge (~0.08 SR/yr Carver limit).
- `run_cost_stress` in `validation/gates.py` and retail scenarios
  (`retail_cost_scenario("BASE"|"CONSERVATIVE"|"STRESS")`) force net-of-cost
  evaluation before research gates pass.
- Appendix E of the paper (validation loss ↛ OOS IR/Sharpe) is exactly why
  Quantester’s research checklist prefers **PBO / DSR / MCPT / untouched OOS**
  over “best validation loss wins.”

### 3.2 Cost-aware edge haircut (conceptual cousin)

Paper eq. (5) zeroes *trades* when \(|\hat r|\) is below a cost multiple.
Quantester’s closest primitive is Kakushadze effective returns in
`portfolio/sizing.py`:

```text
E_eff = sign(E) * max(|E| - τ, 0)
```

Same economic idea (do not act on edges smaller than linear costs), but applied
**before weight optimization**, not as a **position-hysteresis execution
filter**. The paper’s λ-gate keeps the previous position; Kakushadze shrinks
the expected-return input to the sizer. Different layer of the stack.

### 3.3 Crypto hourly path exists

`CCXTDataHandler` (`data/ccxt_handler.py`), `load_crypto` in `simple.py`, and
examples (`donchian_breakout/run_ccxt.py`, `tranche_pullback/run_ccxt.py`)
already support BTC/USDT-style OHLCV research under the temporal firewall.
Replicating the paper’s *data source* is feasible; replicating its *evaluation
protocol* is not a one-liner.

### 3.4 Technical indicators

Paper TA families (RSI, ROC, MACD, ATR, BB, VWAP, OBV, MFI, …) overlap
substantially with `quantester/indicators/` (`rsi`, `macd`, `atr`,
`bollinger_bands`, `sma`/`ema`, `rolling_volatility`, …). Quantester does not
ship the paper’s fold-local Spearman group-selection of 10 TA features from a
94-candidate pool.

### 3.5 Anti-overfitting toolkit is *stricter* than the paper

| Concern | Paper | Quantester |
| --- | --- | --- |
| Temporal CV | Rolling WFO (27 folds) | `CombinatorialPurgedKFold` + embargo (`validation/cpcv.py`) |
| Selection bias | Optuna on val; selectors Loss / IC / IR\*\* | CSCV **PBO** gate `< 0.10`, Trials Registry + **DSR** |
| Skill vs noise | Circular block-bootstrap on Δreturns | MCPT (`montecarlo/permutation.py`), Masters partition, autocorrelation gate → block/O-U |
| Look-ahead audit | Causal feature construction claims | Truncation test (`validation/truncation.py`), `check_lookahead`, phase-aware `get_latest_bars` |
| Cost stress | Flat 10 bps + sensitivity §6.2 | Microstructure costs + `run_cost_stress` |

A Quantester reading of the paper’s flagship equity curve would ask: what is
PBO over the λ / feature / selector grid? What is DSR after counting all
Optuna trials × folds × architectures? The paper’s descriptive ARC/SR figures
would likely be **deflated** under Quantester’s gates — consistent with the
paper’s own fragile bootstrap results.

---

## 4. Conflicts and design tensions

### 4.1 Missing bars: forward-fill vs availability mask

**Paper §3.2:** reconstruct a complete hourly index; missing bars → carry
forward previous close, volume = 0 (flat synthetic bar).

**Quantester invariant** (`data/streaming.py`, core rules): outer-join timestamp
union; missing bars mark the asset **untradeable**, never fabricate prices.

Implication: a faithful Quantester replication must **not** copy the paper’s
gap fill. Treat gaps as no-trade; report coverage. The paper’s flat bars can
mechanically smooth returns near outages and feed spurious TA values.

### 4.2 Execution realism

| | Paper | Quantester |
| --- | --- | --- |
| Fills | Position × next-hour close-to-close return minus \(c\cdot|\Delta pos|\) | Event queue: `MarketEvent → Signal → Order → Fill` with `earliest_fill_time` |
| Costs | Single proportional \(c\) | Commission + half-spread + Kaufman slip + Kyle impact (embedded in `fill_price`) |
| Delay | Implicit: signal at \(t\), earn \(r_{t+1}\) | Explicit `delay` / open-phase intra-bar guard |

The paper’s net-return formula (eq. 4) is a **vectorized accounting identity**,
not an order-book or bar-open execution ledger. Quantester’s delay-0/1 firewall
is the correct place to map “forecast at close \(t\), hold over \([t,t+1]\)” —
typically **delay=1** (signal at close \(T\), fill at open \(T+1\)) or a
documented close-to-close research mode with truncation proof.

### 4.3 Walk-forward vs CPCV/PBO

Paper contribution is a **27-fold rolling WFO**. Quantester’s production
example (`examples/production_research/run.py`) has a walk-forward *stage* for
expanding re-selection, but there is **no library-level WFO fold scheduler**
comparable to the paper’s 12/3/3 protocol.

For ML primary models with label overlap, Quantester’s preferred path remains
**purged CPCV + embargo**, not rolling WFO alone (de Prado AFML; see
`validation/cpcv.py`). WFO is still useful for regime diagnostics (the paper’s
Appendix D fold table is valuable); it should complement, not replace, PBO/DSR.

### 4.4 Primary forecast ML vs meta-labeling

Paper: the ML model **is** the primary signal (regression → sign → position).

Quantester: `MetaLabelingStrategy` (`strategy/meta_labeling.py`) is a
**secondary** probability sizer over a primary rule (triple-barrier labels).
There is no first-class XGBoost/LSTM/iTransformer return-forecast strategy
module.

To reproduce the paper inside Quantester you would add a primary
`ForecastStrategy` (or offline forecast series → `SignalEvent` emitter), then
optionally wrap with meta-labeling — not the other way around.

### 4.5 Inference API gap

Paper: paired **circular** block-bootstrap on strategy return differentials,
Holm-adjusted.

Quantester today:

- Politis–Romano **stationary** block bootstrap on OHLCV
  (`montecarlo/synthetic.py`)
- Fixed-length block trade resampling (`montecarlo/trade_resampling.py`)
- MCPT / Masters p-values

Missing: circular wrapping, and a dedicated **paired differential** test
object for strategy A − strategy B (or strategy − B&H) after costs.

---

## 5. Capability matrix

| Paper theme | Quantester status | Where |
| --- | --- | --- |
| Cost-aware filter \(\lvert\hat r\rvert > \lambda c \lvert\Delta pos\rvert\) | **Gap** (Kakushadze is sizing-only cousin) | Would live in `strategy/` or portfolio target filter |
| Flat ~10 bps TC | **Partial** — configurable; retail spread presets include 10 bps | `execution/costs.py` |
| Sign long-only / long-short | **Yes** (rule-based) | `examples/production_research/strategy.py`, Donchian `long_only` |
| Primary return-forecast ML | **Gap** | Only meta-labeling scaffold |
| Rolling 27-fold WFO API | **Gap** (example stage only) | `examples/production_research/run.py` |
| TA features | **Partial** | `indicators/` |
| EGARCH features | **Gap** | — |
| Sharpe / returns | **Yes** | `analytics/performance.py` |
| Turnover metric | **Gap** (Carver drag needs caller-supplied turnover) | — |
| Circular block-bootstrap on Δreturns | **Gap** | Related: stationary/fixed-block MC |
| CPCV / PBO / DSR / truncation | **Stronger than paper** | `validation/`, `analytics/dsr.py` |
| CCXT BTC data | **Yes** | `data/ccxt_handler.py` |
| Missing-bar policy | **Stricter (correct) divergence** | Availability masks, not ffill |

---

## 6. What Quantester should take from the paper

Highest-value, low-architecture-risk ideas:

1. **Cost-aware execution filter** as a reusable strategy/portfolio primitive  
   Implement eq. (5)–(7): given a target position from any primary signal
   (forecast, rule, meta-label), update only when
   \(\lvert edge\rvert > \lambda \cdot c \cdot \lvert\Delta pos\rvert\).
   This is the paper’s main economic contribution and plugs the gap between
   Kakushadze (sizing) and Carver (ex-post drag).

2. **Turnover analytics**  
   First-class annual turnover + wire into `carver_cost_drag_sr` automatically
   from the fill ledger — the paper shows turnover is the channel that kills
   hourly strategies.

3. **Rolling WFO helper** (research utility, not a replacement for CPCV)  
   Emit fold masks (train/val/test), support retrain-on-train+val, and
   fold-level tearsheets like Appendix D. Useful for crypto regime diagnostics.

4. **Paired circular block-bootstrap on return differentials**  
   Complements MCPT when comparing two net-of-cost equity curves (filter on vs
   off; model A vs B; strategy vs B&H).

Lower priority / optional:

- EGARCH feature helper (fold-fit, freeze params, recurse) — niche vs
  `rolling_volatility` / ATR.
- Primary ML forecast adapters (XGBoost tabular) behind the `Strategy` interface
  with mandatory `vectorized_signals` for fast-track parity.

---

## 7. How a Quantester-native “replication” should differ from the paper

If someone ports this study into Quantester, do **not** copy the paper’s
backtest literally. Prefer:

1. **Data:** `CCXTDataHandler` hourly BTC/USDT; **no** close-ffill of gaps;
   audit with `audit_ohlcv_frame`.
2. **Timing:** declare `delay` explicitly; prove no look-ahead with truncation
   test.
3. **Costs:** start from `retail_cost_scenario("CONSERVATIVE")` (~10 bps spread
   stress) or a proportional commission model; still run cost-stress gates —
   do not stop at a single flat \(c\).
4. **Filter:** implement λ-gate as strategy logic; log trades suppressed.
5. **Validation:** WFO for regime tables **plus** CPCV/PBO over the research
   grid, Trials Registry → DSR, MCPT on the chosen rule, autocorrelation gate
   before IID assumptions.
6. **Inference:** report paired block-bootstrap ΔSR **and** Quantester gates;
   treat ARC > 65% as a point estimate under scrutiny, not a deploy signal.
7. **Accounting:** fills through the event ledger so `fill_price` embeds costs
   once (no double-counting `slippage_cost` in cash).

Under that protocol, the paper’s qualitative lesson (cost-aware execution
rescues weak hourly forecasts more than architecture shopping) is the result
worth stress-testing — not the headline ARC figure.

---

## 8. Bottom line for the codebase

- **Philosophical fit: high.** The paper is empirical evidence for Quantester’s
  “costs and turnover first” stance on short-horizon crypto.
- **Methodological fit: mixed.** Quantester already exceeds the paper on
  look-ahead firewalls and selection-bias gates; the paper exceeds Quantester on
  an explicit cost-threshold **execution** rule and a clean WFO/ML forecast
  experiment design.
- **Do not import:** missing-bar forward-fill; flat-only costs without stress;
  trusting validation loss or single ARC without PBO/DSR/MCPT.
- **Worth implementing:** λ-cost execution filter, turnover metric, optional
  WFO fold utility, paired circular block-bootstrap for strategy differentials.

---

## Sources

- Paper PDF: uploaded `2606.00060v1` (arXiv:2606.00060v1 [q-fin.TR]).
- Quantester modules cited above (paths relative to repo root).
- Workspace invariants: `.cursor/rules/quantester-core.mdc`,
  `docs/architecture.md`, `docs/modules/execution.md`,
  `docs/modules/analytics.md`, `docs/modules/montecarlo.md`,
  `examples/production_research/run.py`.
