# Quantester — Codebase Context & Backtesting Guide

> **Purpose of this document.** This is a single-file briefing on the
> **Quantester** repository, written to be loaded into a notebook as context.
> It describes what the application is, how it is built, every module's public
> API (with exact signatures), and — most importantly — **all the ways to
> backtest a trading strategy with it**, end to end, with runnable code.
>
> Quantester is an institutional-grade, **event-driven quantitative
> backtesting engine** in pure Python (≥ 3.12). Its design is governed by a
> set of non-negotiable invariants that make the most common backtest cheats
> (look-ahead bias, perfect fills, erased gaps, selection bias) structurally
> impossible rather than merely discouraged. Core formulas are verified
> against specialist quant literature (López de Prado *AFML*, Bailey, Masters,
> Vince, Carver, Kaufman, Chan, Ehlers); items the literature does not cover
> are flagged in module docstrings with their canonical source.

---

## 1. What the app is

Five decoupled modules communicate **only** through a strict four-event
lifecycle on a shared synchronous queue:

```
MarketEvent  →  SignalEvent  →  OrderEvent  →  FillEvent
 (new bars)     (strategy)      (portfolio)    (execution)
```

| Module | Package | Role |
| --- | --- | --- |
| Data Handler | `quantester/data` | Point-in-time market data stream; enforces what is *visible* at each moment (the look-ahead firewall). |
| Strategy | `quantester/strategy` | Consumes market events, emits directional signals (`LONG` / `SHORT` / `EXIT`). Never decides *how much* or *at what price*. |
| Portfolio Manager | `quantester/portfolio` | Turns signals into sized orders; keeps the cash/holdings ledger; risk overlays (margin monitor, daily drawdown breaker); sizing engines (Kelly / vol-parity / Vince optimal-f). |
| Execution Simulator | `quantester/execution` | Pending-order ledger enforcing the temporal firewall; fills at real bar prices with cost models (half-spread, Kaufman slippage, Kyle lambda impact, commissions). |
| Performance Analytics | `quantester/analytics` | Offline (never in the event loop): Sharpe/MDD/Calmar, tearsheets, Carver cost drag, trials registry, Deflated Sharpe Ratio. |

Supporting packages:

- `quantester/validation` — truncation leak test, Purged/Embargoed K-Fold &
  CPCV, CSCV Probability of Backtest Overfitting (PBO).
- `quantester/montecarlo` — vectorized fast-track engine, trade resampling,
  MCPT permutation testing (Masters), double-bootstrap drawdown bounds,
  Ornstein-Uhlenbeck synthetic paths + OTR sweeps, stationary block bootstrap
  of OHLCV, autocorrelation diagnostics gate.
- `quantester/utils` — de Prado's ETF trick, seeded synthetic OHLCV generator.
- `quantester/visualization` — static charts + interactive viewer (display
  tooling only, never inside the event loop).

## 2. Installation & test suite

```bash
pip install -e .[dev]          # engine + pytest
pip install "quantester[data]" # optional: yfinance + ccxt + akshare + requests
pytest                          # full suite, finishes in seconds
```

Runtime dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`,
`matplotlib`. Optional extras: `yfinance` (Yahoo Finance), `ccxt>=4` (100+
crypto exchanges), `akshare` (China/US portals), `requests` (Stooq/FMP/macro
REST). Macro overlays live in `quantester.macro` (World Bank, NBP, GUS).
Tests live in `tests/` (one file per module) and are
configured in `pyproject.toml` (`pythonpath=["."]`, `testpaths=["tests"]`).

## 3. Core architecture invariants

These five rules define the engine; every backtest inherits them.

### 3.1 Queue-only communication

Components never call each other directly. `BacktestEngine.run_backtest()`
(`quantester/engine.py`) is an outer loop over the data stream with an inner
loop that drains the event queue. Every bar is processed **twice** — an
`open` phase, then a `close` phase:

```
prime_data()
while data remains:
    timestamp, bars = data_handler.advance()
    # OPEN phase:  execution handler retries parked orders first,
    #              then delay=0 strategies calculate signals
    # CLOSE phase: portfolio marks equity to market (+ margin monitor),
    #              then delay>=1 strategies calculate signals
```

Event routing inside the drain:

| Event | Routed to |
| --- | --- |
| `MarketEvent` (open) | `ExecutionHandler.on_market`, then delay-0 strategies |
| `MarketEvent` (close) | `Portfolio.update_portfolio_valuation`, then delay-≥1 strategies |
| `SignalEvent` | `Portfolio.update_from_signal` → emits `OrderEvent` |
| `OrderEvent` | `ExecutionHandler.execute_order` → emits `FillEvent` (or parks it) |
| `FillEvent` | `Portfolio.update_from_fill` → updates the ledger |

Events posted during a drain are processed before the phase ends, so a full
`Market → Signal → Order → Fill` cascade completes within one phase when the
order is immediately eligible.

### 3.2 State-Based Temporal Firewall (look-ahead safety is enforced)

Not a hardcoded T+1 — a mechanism:

- **Two phases per bar.** During the open phase the DataHandler exposes all
  bars *strictly before* the current bar plus the current bar's **open print
  only** (`get_current_open`). The full bar becomes visible at the close
  phase. A strategy can never peek at the current close while trading at the
  current open.
- **`delay` per strategy.** `delay=1` (default): signal computed at close of
  bar T fills at open of bar T+1. `delay=0`: signal computed at bar T's open
  fills at bar T's open under the intra-bar guard (data strictly before the
  fill timestamp).
- **`earliest_fill_time` per order.** The portfolio stamps each order with
  `timestamp_at_offset(signal.timestamp, delay)`; the execution handler's
  pending-order ledger parks any order whose time has not arrived and retries
  it on every subsequent open phase. An order can never fill early.

### 3.3 Availability masks (no silent history rewrites)

Multi-symbol data aligns on the **outer-join union** of all timestamps. A
symbol with no bar at a timestamp is served as `None` in `event.bars` —
**untradeable, never erased**. Dropping incomplete bars would delete exactly
the high-stress/illiquid periods a backtest must survive (selection bias).
Strategies must guard: `if event.bars.get(symbol) is None: return`.

### 3.4 Ledger accounting

- **`fill_price` is all-in** — it already embeds half-spread + volatility
  slippage + market impact. Cash is charged `qty * fill_price + commission`.
- **`slippage_cost` (φₜ, implementation shortfall) is informational** —
  recorded on the `FillEvent` for cost analytics, **never** deducted again.
  Commissions (cₜ, proportional costs) are the only separately charged cost.
- Completed round-trips land in `portfolio.trades` with entry/exit prices and
  realized PnL; open lots are tracked at volume-weighted average price.
- Stop orders fill at the **next available price after gap-through**, never
  the guaranteed stop price (perfect stops silently unbound optimal-f).

### 3.5 Seeded randomness only

All RNG uses local `numpy.random.Generator(seed)` — no global `np.random`.
Every Monte Carlo function takes `seed=`; everything is reproducible.

## 4. Repository layout

```
quantester/
├── engine.py              # BacktestEngine — the synchronous event loop
├── events.py              # MarketEvent / SignalEvent / OrderEvent / FillEvent + constants
├── data/
│   ├── base.py            # DataHandler abstract interface (the firewall contract)
│   ├── streaming.py       # StreamingDataHandler — shared streaming engine + firewall
│   ├── csv_handler.py     # HistoricCSVDataHandler (CSV files or DataFrames)
│   ├── yfinance_handler.py# YFinanceDataHandler (Yahoo Finance, optional extra)
│   ├── ccxt_handler.py    # CCXTDataHandler (crypto exchanges, optional extra)
│   ├── stooq_handler.py   # StooqDataHandler (CSV download, API key)
│   ├── fmp_handler.py     # FMPDataHandler (stable EOD, API key)
│   ├── akshare_handler.py # AKShareDataHandler (CN/US daily)
│   └── bars.py            # dollar bars, tick/dollar/volume imbalance bars (AFML ch. 2)
├── macro/                 # World Bank / NBP / GUS overlays + as_daily_reindex
├── strategy/
│   ├── base.py            # Strategy ABC (delay, calculate_signals, vectorized twin)
│   ├── examples.py        # BuyAndHoldStrategy, MovingAverageCrossStrategy
│   ├── pairs_trading.py   # PairsTradingStrategy (rolling-OLS GLD/GDX z-score, Chan)
│   ├── tranche_pullback.py# TranchePullbackStrategy (BTC dip-buying limit ladder)
│   └── meta_labeling.py   # triple_barrier_labels + MetaLabelingStrategy (AFML ch. 3)
├── portfolio/
│   ├── portfolio.py       # PortfolioManager, PercentEquitySizer, FixedUnitSizer
│   ├── sizing.py          # kelly_fraction, kelly_gaussian, volatility_parity_weights,
│   │                      #   hpr, twr, optimal_f, kakushadze_effective_returns
│   └── risk.py            # MarginMonitor, DailyDrawdownBreaker,
│                          #   stabilized_covariance, spectral_risk_attribution
├── execution/
│   ├── base.py            # ExecutionHandler interface (on_market, execute_order)
│   ├── simulator.py       # SimulatedExecutionHandler (pending-order ledger)
│   └── costs.py           # CostModel, ConservativeFrictionCostModel
├── analytics/
│   ├── performance.py     # log_returns, annualized_sharpe, max_drawdown,
│   │                      #   drawdown_series, calmar_ratio, carver_cost_drag_sr,
│   │                      #   speed_limit_warning, summarize
│   ├── tearsheet.py       # generate_tearsheet (PNG + stats dict)
│   ├── trials_registry.py # TrialsRegistry (SQLite log of every optimization trial)
│   └── dsr.py             # expected_max_sharpe, probabilistic_sharpe_ratio,
│                          #   deflated_sharpe_ratio, dsr_from_registry
├── validation/
│   ├── truncation.py      # run_truncation_test (Chan's leak detector)
│   ├── cpcv.py            # PurgedKFold, CombinatorialPurgedKFold (AFML ch. 7/12)
│   └── pbo.py             # pbo_cscv (Bailey–de Prado CSCV rank-logit), PBO_GATE=0.10
├── montecarlo/
│   ├── fast_track.py      # fast_backtest — vectorized engine with proven parity
│   ├── trade_resampling.py# empirical_resample (hat/block bootstrap), ehlers_randomized_equity
│   ├── permutation.py     # permute_log_changes, multi_market_permutation,
│   │                      #   intra_inter_bar_permutation, masters_p_value,
│   │                      #   permutation_test, trend_bias_skill
│   ├── drawdown.py        # double_bootstrap_dd_bound, single_loop_dd_quantile
│   ├── synthetic.py       # estimate_ou_params, generate_ou_paths, otr_sweep,
│   │                      #   correlated_gaussian_returns, bootstrap_ohlcv
│   └── diagnostics.py     # runs_test, ljung_box, autocorrelation_gate
├── utils/
│   ├── etf_trick.py       # ETFTrick (de Prado total-return index K_t, costs external)
│   └── synthetic.py       # make_synthetic_ohlcv, write_csvs (seeded GBM data)
└── visualization/
    ├── static.py          # plot_candles, plot_equity, plot_trade_analysis,
    │                      #   plot_monthly_returns, plot_rolling_metrics,
    │                      #   plot_path_distribution, trade_stats
    ├── interactive.py     # interactive_view / InteractiveChartViewer (scroll/zoom/pan)
    └── indicators.py      # sma, ema, rsi, macd, bollinger_bands, atr, rolling_volatility
examples/                  # 9 runnable end-to-end scripts (see §7)
tests/                     # pytest suite mirroring the modules
docs/                      # human docs (architecture, tutorials, module references)
```

Top-level research reports (`Technical Raport*.md`, `Cross-Reference.md`,
`2nd Cross Reference.md`, `Monte Carlo.md`) are the design specifications the
engine was built and audited against.

---

## 5. The canonical backtest recipe (everything plugs into this)

Every backtest — synthetic or real data, any strategy — is the same six-step
wiring:

```python
from quantester.data.csv_handler import HistoricCSVDataHandler
from quantester.engine import BacktestEngine
from quantester.execution.costs import CostModel
from quantester.execution.simulator import SimulatedExecutionHandler
from quantester.portfolio.portfolio import PercentEquitySizer, PortfolioManager
from quantester.strategy.examples import MovingAverageCrossStrategy

# 1. DATA: {symbol: DataFrame} or {symbol: csv_path}; schema
#    datetime index + open/high/low/close/volume columns.
handler = HistoricCSVDataHandler({"AAPL": "data/AAPL.csv"})

# 2. STRATEGY: direction only; declare delay (1 = T+1, safest default).
strategy = MovingAverageCrossStrategy(handler, "AAPL", fast=10, slow=40,
                                      direction="both")

# 3. PORTFOLIO: sizing + risk overlays + ledger.
portfolio = PortfolioManager(handler, initial_capital=100_000.0,
                             sizer=PercentEquitySizer(0.9))

# 4. EXECUTION: deterministic cost model; fills at real bar prices.
execution = SimulatedExecutionHandler(CostModel())

# 5. RUN.
engine = BacktestEngine(handler, strategy, portfolio, execution)
engine.run_backtest()          # returns the portfolio

# 6. READ RESULTS (post-run, offline).
from quantester.analytics.performance import summarize
print(summarize(portfolio.equity_curve))
# {'total_return': ..., 'sharpe': ..., 'max_drawdown': ...,
#  'max_drawdown_duration_days': ..., 'calmar': ...}
```

Post-run portfolio attributes:

| Attribute | Type | Contents |
| --- | --- | --- |
| `equity_curve` | `pd.Series` | Mark-to-market equity per bar (cash + holdings). |
| `positions_history` | `pd.DataFrame` | Position per symbol per bar. |
| `fills` | `list[FillEvent]` | Every fill: `fill_price` (all-in), `commission`, `slippage_cost`, `reference_price`. |
| `trades` | `list[dict]` | Round-trips: `symbol, t0, t1, qty, entry_price, exit_price, pnl`. |
| `cash`, `positions` | — | Final ledger state. |
| `equity`, `gross_exposure` | properties | Live account value; Σ\|qty·price\|. |

`BacktestEngine` also accepts a **list of strategies** (multi-strategy books).

---

## 6. Ways to backtest a strategy

### 6.1 Data sources (bar feeds + macro overlays, one firewall)

All **bar** feeds converge on `StreamingDataHandler`, so the temporal firewall and
availability-mask semantics are identical regardless of source.

**A. Seeded synthetic GBM data** (no downloads; deterministic):

```python
from quantester.utils.synthetic import make_synthetic_ohlcv, write_csvs

df = make_synthetic_ohlcv("AAA", n_bars=750, s0=100.0,
                          mu=0.10, sigma=0.22,      # annualized drift / vol
                          start="2020-01-01", seed=1,
                          missing_every=None)        # k -> drop every k-th bar
paths = write_csvs({"AAA": df}, "examples/data")    # schema-correct CSVs
```

`missing_every=k` simulates illiquid gaps — exactly what the availability
mask is built for.

For pairs-trading work, `make_cointegrated_pair(n_bars=750, beta=1.4,
spread_phi=0.95, spread_sigma=0.02, ..., seed=7, gdx_missing_every=None)`
returns seeded cointegrated GLD/GDX-like OHLCV frames: ln GDX follows GBM and
ln GLD = α + β·ln GDX + e, where e is a stationary AR(1) (discrete
Ornstein-Uhlenbeck) spread — cointegrated by construction with a **known**
hedge ratio, ideal for testing `PairsTradingStrategy`.

**B. Local CSVs** — header `datetime,open,high,low,close,volume`, one per
symbol, or pre-loaded DataFrames indexed by datetime:

```python
handler = HistoricCSVDataHandler({
    "AAPL": "data/AAPL.csv",
    "MSFT": msft_dataframe,
})
```

**C. Yahoo Finance** (`pip install "quantester[yfinance]"`):

```python
from quantester.data import YFinanceDataHandler
handler = YFinanceDataHandler(["AAPL", "MSFT"], start="2022-01-01",
                              end="2025-01-01", interval="1d",
                              auto_adjust=True)
```

**D. Crypto exchanges via ccxt** (`pip install "quantester[ccxt]"`):

```python
from quantester.data import CCXTDataHandler
handler = CCXTDataHandler(["BTC/USD", "ETH/USD"], exchange="coinbase",
                          timeframe="1d", start="2023-01-01", end="2025-01-01")
# kwargs: exchange="binance"|..., timeframe="1m"|"1h"|"1d"|..., limit=1000,
#         drop_incomplete=True (drops the still-forming last candle)
```

**E. Stooq / FMP / AKShare** (`pip install "quantester[data]"`):

```python
from quantester.data import StooqDataHandler, FMPDataHandler, AKShareDataHandler
# export QUANTESTER_STOOQ_API_KEY=... / QUANTESTER_FMP_API_KEY=...
stooq = StooqDataHandler("aapl.us", start="2022-01-01", end="2025-01-01")
fmp = FMPDataHandler("AAPL", start="2022-01-01", end="2025-01-01")
ak = AKShareDataHandler("000001", market="cn", start="2022-01-01", end="2025-01-01")
```

**F. Macro overlays** (`quantester.macro` — not bar feeds):

```python
from quantester.macro import load_world_bank, load_nbp_fx, as_daily_reindex
fx = load_nbp_fx("USD", start="2023-01-01", end="2024-12-31")
aligned = as_daily_reindex(handler.source_ohlcv("AAPL").index, fx)
```

**G. Tick data → information-driven bars** (`quantester/data/bars.py`, AFML
ch. 2). Input ticks: datetime-indexed DataFrame with `price, volume`; output
is OHLCV ready for `HistoricCSVDataHandler`:

```python
from quantester.data.bars import (dollar_bars, tick_imbalance_bars,
                                  dollar_imbalance_bars, volume_imbalance_bars)
bars = dollar_bars(ticks, threshold=1_000_000)        # one bar per $1M traded
bars = tick_imbalance_bars(ticks, span=10, warmup=3)  # EWMA tick-rule imbalance
```

Imbalance bars sample more when trade flow is one-sided (informed bursts),
producing return series closer to iid-normal than clock-time bars.

### 6.2 Using a bundled strategy

**`BuyAndHoldStrategy(data_handler)`** — long every symbol once, at the first
tradeable bar. Benchmark and test fixture.

**`MovingAverageCrossStrategy(data_handler, symbol, fast=10, slow=30,
direction="both", delay=1)`** — SMA crossover; `direction` is `"both"`,
`"long"`, or `"short"`. Emits only on crosses; its event form and vectorized
twin share the pure function `crossover_positions`, so Monte Carlo parity
holds by construction.

**`PairsTradingStrategy(data_handler, leg_y="GLD", leg_x="GDX",
ols_window=252, zscore_window=20, entry_z=2.0, exit_z=0.5, delay=1,
min_train_obs=None)`** — Chan's rolling-hedge-ratio mean reversion. Spread
`z_t = ln(P_y) − β_t·ln(P_x) − α_t` from a rolling OLS fit (scikit-learn
`LinearRegression`), z-scored over a rolling window; enter beyond
`±entry_z`, exit when `|s| ≤ exit_z`. Both legs signal in the same event
cycle; gaps in either leg pause the strategy (no fabricated spreads). Exposes
`history_` diagnostics (per-bar α, β, z, s, state) and a vectorized twin.

**`TranchePullbackStrategy(data_handler, symbol, regime_window=200,
peak_window=20, atr_window=14, atr_spacing=1.5, tranche_fractions=(0.25,
0.35, 0.40), exit_window=5, stop_atr_mult=5.0, reanchor_every=1,
cooldown_bars=0)`** — volatility-spaced dip-buying limit ladder (built for
BTC):

- Regime gate: arms only while `close > SMA(regime_window)`.
- While flat, three resting `LIMIT` buys at `peak − k·atr_spacing·ATR`
  (peak = rolling `peak_window` close high; Wilder ATR) are re-anchored every
  `reanchor_every` bars. The **first fill freezes** the levels until the
  position fully closes.
- Sizing rides on `SignalEvent.strength` against `limit_price=T_k`, so wire
  `PortfolioManager(sizer=PercentEquitySizer(1.0))` for the exact
  `q_k = equity·f_k/T_k` mapping.
- Exits: mean reversion at `close ≥ SMA(exit_window)` (next open); hard stop
  at `peak − stop_atr_mult·ATR` per Kaufman's close-execution rule — the
  intra-bar low triggers a **market-on-close** exit.
- `delay=1`, **no vectorized twin** (latched path-dependent state machine) —
  validate with the block-bootstrap harness (§6.7), not the fast-track.
- On intraday data set `reanchor_every=bars_per_day` and
  `cooldown_bars=bars_per_day − 1` (daily cadence, per-bar resolution).

Typical wiring (conservative friction + circuit breaker):

```python
from quantester.strategy.tranche_pullback import TranchePullbackStrategy
from quantester.portfolio.risk import DailyDrawdownBreaker
from quantester.execution.costs import ConservativeFrictionCostModel

portfolio = PortfolioManager(handler, 25_000.0, sizer=PercentEquitySizer(1.0),
                             drawdown_breaker=DailyDrawdownBreaker(0.045))
engine = BacktestEngine(handler, TranchePullbackStrategy(handler, "BTC/USD"),
                        portfolio,
                        SimulatedExecutionHandler(ConservativeFrictionCostModel(
                            spread_pct=0.0002, fee_rate=0.0004)))
```

### 6.3 Writing a custom strategy (the authoring contract)

Subclass `Strategy` (`quantester/strategy/base.py`). A strategy decides
**direction only** — never size (the portfolio's sizer does that), never
price (the execution simulator does that).

Three rules keep every strategy honest:

1. **Only look through the firewall** — read data exclusively via
   `data_handler.get_latest_bars(symbol, n)` / `get_current_open(symbol)` and
   `event.bars`. Never touch a raw DataFrame.
2. **Only speak in events** — `events_queue.put(SignalEvent(...))`. Never
   call the portfolio or execution handler directly.
3. **Emit only on target changes** — carry position state on the instance;
   re-emitting the same target every bar spams redundant orders.

Complete example (from `examples/run_custom_strategy.py`):

```python
from quantester.events import EXIT, LONG, SignalEvent
from quantester.strategy.base import Strategy

class MomentumStrategy(Strategy):
    """Long when lookback-bar close-to-close momentum is positive; flat else."""

    def __init__(self, data_handler, symbol: str, lookback: int = 20):
        self.data_handler = data_handler   # firewall-respecting data window
        self.symbol = symbol
        self.lookback = lookback
        self.delay = 1                     # signal at close T -> fill open T+1
        self._position = 0.0

    def calculate_signals(self, event, events_queue):
        if event.bars.get(self.symbol) is None:
            return                                   # untradeable right now
        bars = self.data_handler.get_latest_bars(self.symbol, self.lookback + 1)
        if len(bars) < self.lookback + 1:
            return                                   # warmup: not enough history
        momentum = bars["close"].iloc[-1] / bars["close"].iloc[0] - 1.0
        if momentum > 0 and self._position <= 0:
            events_queue.put(SignalEvent(event.timestamp, self.symbol,
                                         LONG, strength=1.0, delay=self.delay))
            self._position = 1.0
        elif momentum <= 0 and self._position > 0:
            events_queue.put(SignalEvent(event.timestamp, self.symbol,
                                         EXIT, strength=1.0, delay=self.delay))
            self._position = 0.0

    # Optional but required for Monte Carlo fast-track validation:
    def vectorized_signals(self, data: dict):
        close = data[self.symbol]["close"]
        momentum = close / close.shift(self.lookback) - 1.0
        return {self.symbol: (momentum > 0).astype(float)}
```

Notes on the interface:

- `delay`: `1` (default) = act on close-phase events, fill at next open;
  `0` = act on open-phase events, fill at the current open under the
  intra-bar guard. `matches_phase(phase)` does the routing; override only if
  you know why.
- `SignalEvent` fields: `symbol`, `signal_type` (`LONG`/`SHORT`/`EXIT`),
  `strength` (conviction multiplier in `(0, 1]`; scales size only — direction
  comes from `signal_type`), `delay`, `fill_at` (`"open"` default; `"close"`
  requests a market-on-close fill at the current bar's close — close-phase
  `delay>=1` strategies only), `limit_price` (size at this price and rest a
  LIMIT order), `cancel_orders` (purge the symbol's resting book first — set
  on exits by strategies that rest orders).
- **Vectorized twin** (`vectorized_signals(data) -> {symbol: target_series}`):
  full-history target positions, numerically identical to the event form
  (share one pure signal function, as `crossover_positions` does). Required
  for MCPT/fast-track Monte Carlo; a parity test proves the two forms
  produce the same equity curve. Raises `NotImplementedError` by default.

### 6.4 Sizing ("how much")

A sizer is any callable `(signal, portfolio, ref_price) -> target_qty`
(signed target position). Wire via `PortfolioManager(sizer=...)`.

| Sizer | Target |
| --- | --- |
| `PercentEquitySizer(pct=0.5)` | `± pct × equity × strength / ref_price` (0 on EXIT). Compounds with account size. Default. |
| `FixedUnitSizer(units=100.0)` | `± units × strength` shares. Used for fast-track parity checks. |

Standalone sizing engines (`quantester/portfolio/sizing.py`) for calibrating
fractions *outside* the event loop:

| Function | Formula / purpose |
| --- | --- |
| `kelly_fraction(win_rate, win_loss_ratio)` | Binary Kelly: `f* = p − q/b`. |
| `kelly_gaussian(mean, variance)` | Continuous Kelly: `f = μ/σ²` (0 if σ² ≤ 0). |
| `volatility_parity_weights(cov)` | `w_i ∝ 1/σ_i`, normalized — equal risk contribution. |
| `hpr(trades, f, worst_loss)` | Vince's Holding Period Return: `1 + f·(−Trade_i/WorstLoss)`. |
| `twr(trades, f, worst_loss)` | Terminal Wealth Relative `Π HPR` (0 if any HPR ≤ 0 — ruin). |
| `optimal_f(trades, worst_loss=None, gap_stress=1.5, f_max=1.0)` | `argmax TWR(f)`; `worst_loss` defaults to historical worst × `gap_stress` — stressed below the nominal stop because stops don't guarantee fills through gaps. |
| `kakushadze_effective_returns(expected, linear_costs)` | `sign(E)·max(|E|−τ, 0)` — zero edges smaller than linear costs **before** weight optimization. |

### 6.5 Execution & cost models

`SimulatedExecutionHandler(cost_model)` keeps the pending-order ledger and
fills per order type:

| Order type | Fill rule |
| --- | --- |
| `MARKET` | Current bar's **open** + adverse cost adjustment. |
| `STOP` | Next available price after the stop is touched, including gap-through (a buy stop gapped at the open fills at the open). Untriggered stops stay pending. |
| `LIMIT` | Rests until touched: buy fills when the bar's low reaches the level, at `min(open, limit)` (gap-through earns improvement, never worse). |
| `MOC` | The bar's **close print**, on its `earliest_fill_time` bar only; never parks (a missed close auction expires the order). |
| `CANCEL` | Synchronously purges the symbol's resting limit/stop orders (parked market orders are never purged). |

**`CostModel`** — five deterministic knobs (determinism is a hard
requirement: the event engine and the Monte Carlo fast-track share these
functions, and a parity test proves they agree):

```python
@dataclass
class CostModel:
    fixed_commission: float = 1.0        # currency per order
    per_share_commission: float = 0.005  # currency per share
    spread_pct: float = 0.0005           # full bid-ask spread, fraction of price
    slippage_vol_coef: float = 0.1       # Kaufman coefficient on bar range %
    impact_coef: float = 0.1             # Kyle lambda scale (Amihud-style)
```

- `commission(quantity, price=None)` → cₜ, charged separately to cash.
- `half_spread(price)` → `price × spread_pct / 2`.
- `kaufman_slippage(price, high, low)` → ∝ the bar's range.
- `kyle_lambda(price, qty, volume, high, low)` → impact rises with
  volatility, falls with depth.
- `adverse_adjustment(price, qty, bar)` → sum of the three, embedded in
  `fill_price` (recorded as φₜ). The adjustment is always adverse (buys pay
  up, sells receive less) — optimistic fills are the most common way
  backtests lie.

**`ConservativeFrictionCostModel(spread_pct, fee_rate,
friction_multiplier=2.0)`** — stressed exchange friction:
`C_trade = 2 × (S_bid-ask/2 + μ_fee)` — the full spread as adverse adjustment
plus doubled notional fee on **every** fill (resting limits included;
taker-grade friction on maker fills is deliberately pessimistic). Matches
Carver's round-trip form.

### 6.6 Risk overlays & account features

```python
PortfolioManager(handler, initial_capital=100_000.0,
                 sizer=PercentEquitySizer(0.9),
                 margin_monitor=MarginMonitor(max_leverage=2.0,
                                              liquidation_fraction=0.5),
                 drawdown_breaker=DailyDrawdownBreaker(max_intraday_dd=0.045),
                 cash_yield_rate=0.02, idle_cash_fraction=0.5)
```

- **`MarginMonitor`** — at close-phase valuation, if
  `gross_exposure / equity > max_leverage`, liquidates `liquidation_fraction`
  of **every** position at the next open.
- **`DailyDrawdownBreaker`** — account-level circuit breaker against the
  daily opening balance. On a breach: cancels all resting orders,
  market-liquidates everything at the next open (retried until filled), and
  suspends all signal flow until the next trading-day rollover. `0.045` gives
  a 0.5% cushion under the 5% daily-loss limit common in prop evaluations.
- **Idle-cash yield** (Kaufman/Carver-verified accounting): positive cash
  accrues `cash_yield_rate × idle_cash_fraction` annualized, compounded by
  elapsed calendar time; `cash_yield_rate=0.0` (default) disables it;
  borrowed cash accrues nothing.
- **`stabilized_covariance(returns)`** — Ledoit-Wolf-shrunk covariance
  (always use before eigendecomposition).
- **`spectral_risk_attribution(returns, weights=None)`** — decomposes
  portfolio variance onto principal components:
  `R_n = β_n²·Λ_nn / σ²`. If PC1 carries 90% of your risk, you have one bet,
  not N.

### 6.7 Monte Carlo validation (`quantester/montecarlo`)

A single realized backtest is one draw from a distribution; this package maps
the distribution. Everything is vectorized and seeded. **Run the
autocorrelation gate first.**

```python
from quantester.montecarlo.diagnostics import autocorrelation_gate
report = autocorrelation_gate(returns, alpha=0.05, lags=10)
report.serial_correlation      # True -> iid resampling is INVALID
report.recommended_method      # "iid_resampling" or "block_bootstrap_or_ou_paths"
```

**Fast-track backtesting** — the vectorized bypass that makes 10,000-rep
studies tractable (parity-tested against the event engine):

```python
from quantester.montecarlo.fast_track import fast_backtest
target = strategy.vectorized_signals({"AAA": df})["AAA"]   # {-1, 0, +1}
result = fast_backtest(df, target, CostModel(),
                       initial_capital=100_000.0, units=100.0)
result.equity, result.sharpe, result.total_return
```

Parity contract: targets decided at close T execute at open T+1; same
`CostModel.adverse_adjustment`; `equity_t = cash_t + Q_t·close_t`.

**Trade-level resampling** (`trade_resampling.py`):

```python
from quantester.montecarlo.trade_resampling import (empirical_resample,
                                                    ehlers_randomized_equity)
hat = empirical_resample(returns, horizon=260, n_sims=10_000, seed=7,
                         block_length=None)   # L -> stationary block bootstrap
hat.quantiles()                                # terminal-return quantiles
hat.paths                                      # (n_sims, horizon+1) paths
eh = ehlers_randomized_equity(win_rate, profit_factor, avg_loss, n_trades,
                              n_sims=10_000, e0=1.0, seed=None)
```

**MCPT permutation testing** (`permutation.py`, Masters) — shuffling log
price changes destroys the chronological patterns a strategy exploits while
preserving statistical moments; a real edge must beat the permuted
distribution. The optimizer is **retrained from scratch on every permuted
path** (same sweep, same costs), so the null includes the same selection
effects:

```python
from quantester.montecarlo.permutation import (permutation_test,
                                               permute_log_changes,
                                               multi_market_permutation,
                                               intra_inter_bar_permutation,
                                               masters_p_value,
                                               trend_bias_skill)
result = permutation_test(close, optimizer, n_reps=1000, seed=7)
result.p_value        # Masters' exact p (count starts at 1)
result.significant    # p < 0.05
```

Permutation protocols: `permute_log_changes` (single market, default);
`multi_market_permutation` (Protocol I — identical shuffle across markets,
preserves cross-sectional correlation); `intra_inter_bar_permutation`
(Protocol II — intra-bar H/O, L/O, C/O shuffled jointly, inter-bar gaps
independently; reconstructs physically valid OHLC).
`trend_bias_skill(r_orig, b_orig, r_perm, b_perm)` partitions return into
`trend` (benchmark), `training_bias = R_perm − B_perm`, and
`skill = (R_orig − Bias) − B_orig`.

**Drawdown bounds** (`drawdown.py`) — single-loop bootstrap underestimates
catastrophic drawdown by >10×; Masters' nested double bootstrap corrects
sequencing risk and sampling error:

```python
from quantester.montecarlo.drawdown import double_bootstrap_dd_bound
bound = double_bootstrap_dd_bound(returns, horizon=None, dd_conf=0.95,
                                  bound_conf=0.70, n_outer=10_000,
                                  n_inner=1_000, seed=7)
bound.bound                 # conservative max-drawdown bound
```

**O-U synthetic paths + OTR sweeps** (`synthetic.py`):

```python
from quantester.montecarlo.synthetic import (estimate_ou_params,
                                             generate_ou_paths, otr_sweep)
ou = estimate_ou_params(close)                       # OLS of dP on P
paths = generate_ou_paths(ou, p0=close.iloc[-1], n_steps=120,
                          n_paths=100_000, seed=7)
grid = otr_sweep(paths, stop_losses=[0.05, 0.10],
                 take_profits=[0.10, 0.20])          # exit calibration over
                                                     # the whole stochastic space
```

Also `correlated_gaussian_returns(n_assets, n_obs, cov, common_shock_scale,
idio_shock_scale, ...)` for multi-asset stress tests with fat tails/regime
shifts.

**Stationary block bootstrap (OHLCV)** — `bootstrap_ohlcv(df, mean_block=20,
seed=42)`: Politis-Romano stationary bootstrap over bars; each bar
contributes its return, open gap, wick fractions and volume jointly, so OHLC
stays physically valid. The vehicle for Monte-Carlo-validating
**path-dependent strategies with no vectorized twin** (e.g. the tranche
ladder): re-run the full event engine on each synthetic path. The null
hypothesis is "BTC-like short-run structure, shuffled regimes" — not "history
will repeat".

### 6.8 The anti-overfitting gates (mandatory before trusting a result)

Ordered; later gates assume earlier ones passed:

| # | Gate | Catches | Pass criterion | Module |
| --- | --- | --- | --- | --- |
| 1 | Truncation test | Look-ahead leakage in the pipeline | `passed == True` | `validation/truncation.py` |
| 2 | CPCV (ML only) | Label overlap leaking train ↔ test | OOS Sharpe distribution across paths | `validation/cpcv.py` |
| 3 | PBO | Parameter selection by luck | `pbo < 0.10` | `validation/pbo.py` |
| 4 | DSR | Selection bias across everything tried | DSR ≥ 0.95 | `analytics/dsr.py` + registry |
| 5 | MCPT | "Edge" that exists in any random path | `p < 0.05` | `montecarlo/permutation.py` |
| — | Autocorrelation gate | Invalid iid resampling | run **before** 5 | `montecarlo/diagnostics.py` |

**Gate 1 — truncation test** (Chan): chop the last N bars, re-run the
identical program, compare overlapping positions bit-for-bit:

```python
from quantester.validation.truncation import run_truncation_test

def run(truncate_last=None):
    data = {"AAA": df.iloc[:-truncate_last] if truncate_last else df}
    handler = HistoricCSVDataHandler(data)
    ...
    engine.run_backtest()
    return portfolio.positions_history

result = run_truncation_test(run, n_truncated=20)
result.passed            # bool gate; result.mismatches debugs a leak
```

**Gate 2 — purged/embargoed CV** (whenever a model is fitted — e.g.
meta-labeling). Financial labels overlap in time, so plain k-fold leaks:

```python
from quantester.validation.cpcv import PurgedKFold, CombinatorialPurgedKFold
for train_idx, test_idx in PurgedKFold(5, t1=t1, pct_embargo=0.01).split(X):
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
cpcv = CombinatorialPurgedKFold(n_groups=6, k_test=2, t1=t1, pct_embargo=0.01)
cpcv.n_splits, cpcv.n_paths   # C(N, N−k) splits; φ[N,k] backtest paths
```

`t1` = label end-times aligned with `X.index` (e.g. triple-barrier vertical
barriers); embargo `h = pct_embargo × T` (~0.01T recommended).

**Gate 3 — PBO** after any parameter sweep (Bailey–de Prado CSCV rank-logit):

```python
from quantester.validation.pbo import pbo_cscv, PBO_GATE   # PBO_GATE = 0.10
result = pbo_cscv(pnl_dataframe, n_blocks=16)   # T × N trials; n_blocks even
result.pbo, result.passes_gate, result.logits
```

**Gate 4 — registry-driven DSR.** The `TrialsRegistry` (SQLite) logs **every
trial including failures**, because DSR needs the honest N and cross-trial
Sharpe variance — never hand-feed N:

```python
from quantester.analytics.trials_registry import TrialsRegistry
from quantester.analytics.dsr import dsr_from_registry

registry = TrialsRegistry("trials.db")          # or ":memory:"
registry.log_trial(params={"fast": 10, "slow": 40}, sharpe=1.23,
                   mean=..., std=..., skew=..., kurt=..., n_obs=...,
                   run_id="ma_sweep_2026_08")
best = registry.best_trial()
dsr = dsr_from_registry(registry, sr_hat=best["sharpe"], n_obs=best["n_obs"],
                        skew=best["skew"], kurtosis=best["kurt"])
# DSR is a probability: want >= 0.95
```

Parallel-safe pattern: workers append via
`TrialsRegistry.write_jsonl_record(path, record)`; a single thread bulk-loads
with `registry.import_jsonl(path)`. Other registry accessors:
`n_trials()`, `sharpe_values()`, `sharpe_variance()`, `best_trial()`,
`close()`. Standalone functions:
`expected_max_sharpe(n_trials, trial_variance)`,
`probabilistic_sharpe_ratio(sr_hat, sr_benchmark, n_obs, skew, kurtosis)`,
`deflated_sharpe_ratio(...)`.

**Gate 5 — MCPT** (see §6.7): `p < 0.05`, on the fast-track, after the
autocorrelation gate.

### 6.9 Meta-labeling (ML secondary model, AFML ch. 3)

The primary model decides the trade *side*; a secondary binary classifier
predicts the probability the primary is right, and that probability scales
the *size* via `SignalEvent.strength` (precision up, recall down):

```python
from quantester.strategy.meta_labeling import (MetaLabelingStrategy,
                                               triple_barrier_labels)
meta = MetaLabelingStrategy(primary, handler,
                            model=GradientBoostingClassifier(),  # any sklearn-style fit/predict_proba
                            feature_fn=my_features,              # (data_handler, signal) -> features
                            size_transform="linear")             # or "zscore" (AFML ch. 10)
meta.fit_secondary(X_train, close, events, tp_pct=0.02, sl_pct=0.02,
                   max_holding=10)   # builds triple-barrier labels, fits model
```

`triple_barrier_labels(close, events, tp_pct, sl_pct, max_holding)`:
`events` has columns `t0` (entry timestamp), `side` (+1/−1); barriers are TP
at `entry·(1+side·tp_pct)`, SL at `entry·(1−side·sl_pct)`, and a vertical
barrier `max_holding` bars out. **Validation requirement:** meta-labeling
trains a model → run it through CPCV with purging + embargo (Gate 2).

### 6.10 Analytics & tearsheets

```python
from quantester.analytics.performance import (
    log_returns, annualized_sharpe, max_drawdown, drawdown_series,
    calmar_ratio, summarize, carver_cost_drag_sr, speed_limit_warning)

stats = summarize(portfolio.equity_curve)
# total_return, sharpe ((μ_daily − Rf)/σ_daily × √252), max_drawdown,
# max_drawdown_duration_days, calmar
# annualized_sharpe(equity, risk_free_daily=0.0, periods=252)  # periods=365 for crypto

drag = carver_cost_drag_sr(annual_turnover=4.0, standardized_cost_sr=0.01)
warning = speed_limit_warning(drag)   # Carver's 0.08 SR/yr speed limit

from quantester.analytics.tearsheet import generate_tearsheet
stats = generate_tearsheet(portfolio.equity_curve, "output/tearsheet.png",
                           title="My strategy",
                           extra_stats={"DSR": "0.97"})   # PNG + stats dict
```

The tearsheet renders equity, underwater (drawdown), return histogram, and a
stats box; matplotlib runs headless (`Agg`) so it works on servers/CI.

### 6.11 Visualization (post-run display tooling)

```python
from quantester.visualization import (indicators, interactive_view,
    plot_candles, plot_equity, plot_trade_analysis, plot_monthly_returns,
    plot_rolling_metrics, plot_path_distribution, trade_stats)

plot_candles(bars, overlays={"SMA(10)": indicators.sma(bars["close"], 10)},
             subpanels={"RSI(14)": indicators.rsi(bars["close"])},
             trades=portfolio.trades, fills=portfolio.fills, path="chart.png")
plot_equity(portfolio.equity_curve,
            positions_history=portfolio.positions_history, path="eq.png")
plot_trade_analysis(portfolio.trades, path="trades.png")
plot_monthly_returns(portfolio.equity_curve, path="monthly.png")
plot_rolling_metrics(portfolio.equity_curve, window=63, path="rolling.png")
plot_path_distribution(mc_paths, path="fan.png")   # percentile fan + terminal histogram

viewer = interactive_view(bars, overlays=overlays, equity=portfolio.equity_curve,
                          trades=portfolio.trades)  # scroll zoom / drag pan / crosshair
viewer.show()            # interactive backend; headless: viewer.save(path)
```

Indicator helpers: `sma, ema, rsi(close, window=14), macd(close, 12, 26, 9),
bollinger_bands(close, 20, 2.0), atr(high, low, close), rolling_volatility`.

---

## 7. The nine example scripts (runnable workflows)

All run from the repo root. `examples/data/` and `examples/output/` are
created as needed.

| Script | What it demonstrates |
| --- | --- |
| `python examples/run_ma_cross.py` | **Canonical hello-world.** Synthetic 3-symbol data (BBB has deliberate gaps → availability mask), MA-cross sweep over 3 parameter pairs logged to the Trials Registry, registry-driven DSR, Carver cost-drag speed-limit check, tearsheet PNG, truncation test `[PASS]`. |
| `python examples/run_custom_strategy.py` | **Tutorial companion** (docs tutorial): build `MomentumStrategy` from scratch → backtest → tearsheet → truncation test → fast-track parity check (`max \|equity diff\|` ~1e-10) → MCPT p-value. |
| `python examples/run_market_data.py` | **Real data**: MA-cross on AAPL via yfinance and BTC/USD via ccxt; identical wiring to the CSV feed (needs network + extras; degrades gracefully). |
| `python examples/run_monte_carlo.py` | **Five-step MC checklist**: hat/block resampling + Ehlers parametric randomization → MCPT p-value + Trend/Bias/Skill partition + Protocol I/II permutations → double-bootstrap DD bound → O-U paths + OTR sweep → autocorrelation gate. Demo-scale reps (raise `N_REPS` ≥ 1,000, `N_OUTER` = 10,000, `N_INNER` = 1,000 for production). |
| `python examples/run_visualizations.py` | **Chart gallery**: candles + indicators + fills/trades, strategy-target view, equity/positions, trade analysis, monthly heatmap, rolling metrics, O-U fan chart, interactive-viewer snapshot. |
| `python examples/run_tranche_pullback.py` | **Tranche ladder plumbing demo** on synthetic GBM: latching, resting limits, conservative friction, 4.5% daily drawdown breaker, truncation check. (GBM has no mean-reversion edge — this validates machinery, not profitability.) |
| `python examples/run_tranche_pullback_ccxt.py` | **Real-data evaluation** of the ladder on full BTC/USD daily history (cached CCXT/Bitstamp): net vs gross, buy-and-hold benchmark on the identical window, per-calendar-year table, ATR-spacing sensitivity, Carver cost audit, Vince optimal-f audit, PBO + DSR over the spacing grid, truncation test, tearsheet. Crypto annualization = 365. |
| `python examples/run_parameter_study_ccxt.py` | **Wide parameter study with governance** (54 valid trials, stop must exceed 3× spacing): forked-process parallelism + registry JSONL merge, CSCV/PBO gate, registry DSR, and the block-bootstrap MC harness (autocorrelation gate → 64 bootstrapped OHLC paths → event-engine re-run per path → P(no edge), same-path B&H benchmark). ~4 min on 4 cores. |
| `python examples/run_parameter_study_intraday_ccxt.py --tf 4h\|1h` | **Intraday ports** of the study: calendar-equivalent windows (SMA200 days → 200×bars-per-day), daily operational cadence (`reanchor_every=bpd`, `cooldown_bars=bpd−1`), wider spacing/stop grid, same PBO/DSR/bootstrap gates. |

---

## 8. Events & constants cheat sheet (`quantester/events.py`)

```python
MARKET, SIGNAL, ORDER, FILL = "MARKET", "SIGNAL", "ORDER", "FILL"
LONG, SHORT, EXIT = "LONG", "SHORT", "EXIT"
BUY, SELL = "BUY", "SELL"
MARKET_ORDER, STOP_ORDER, LIMIT_ORDER = "MARKET", "STOP", "LIMIT"
MOC_ORDER = "MOC"        # market-on-close: this bar's close print only
CANCEL_ORDER = "CANCEL"  # purge the symbol's resting limit/stop orders
OPEN, CLOSE = "open", "close"
```

| Event | Key fields |
| --- | --- |
| `MarketEvent(timestamp, bars, phase)` | `bars`: symbol → OHLCV Series or `None`; `phase`: `"open"`/`"close"`. |
| `SignalEvent(timestamp, symbol, signal_type, strength=1.0, delay=1, fill_at="open", limit_price=None, cancel_orders=False)` | Direction + conviction; `delay`/`fill_at` implement the firewall contract. |
| `OrderEvent(timestamp, symbol, order_type, quantity, direction, earliest_fill_time, stop_price=None, limit_price=None)` | `quantity` always positive; `earliest_fill_time` enforced by the ledger. |
| `FillEvent(timestamp, symbol, quantity, direction, fill_price, commission, slippage_cost, reference_price=0.0)` | `fill_price` all-in; `commission` = cₜ; `slippage_cost` = φₜ (analytics only); `total_cost` property. |

DataHandler interface (`quantester/data/base.py`) — the firewall contract for
custom feeds:

| Member | Contract |
| --- | --- |
| `symbols`, `continue_backtest`, `current_timestamp` | Stream state. |
| `prime_data()` / `advance()` | Reset; step to next timestamp → `(timestamp, bars)` with `None` for untradeable symbols. |
| `set_phase(phase, timestamp)` | Firewall context: `"open"` or `"close"`. |
| `get_latest_bars(symbol, n=1)` | Trailing n bars visible under the current phase (open phase excludes the current bar). |
| `get_current_open(symbol)` | Current bar's open print only; `None` if untradeable. |
| `timestamp_at_offset(timestamp, n)` | Master-calendar offset for stamping `earliest_fill_time`. |
| `bar_at(symbol, timestamp)` | Execution-side full-bar lookup. |

## 9. Glossary (engine-specific terms)

- **Temporal firewall** — enforced look-ahead safety: two bar phases +
  `earliest_fill_time`, not a hardcoded T+1.
- **Availability mask** — a symbol with no bar at a master-calendar timestamp
  is `None` (untradeable); the timestamp is never deleted.
- **Vectorized twin** — a strategy's closed-form full-history target function
  (`vectorized_signals`), numerically identical to its event form; required
  for fast-track Monte Carlo.
- **Fast-track parity** — tested contract that vectorized and event-driven
  backtests produce identical equity curves under identical sizing/costs.
- **cₜ / φₜ** — proportional costs (commissions, charged to cash) /
  implementation shortfall (spread+slippage+impact, embedded in `fill_price`,
  never double-charged).
- **Gap-through** — a stop jumped at the open fills at the open (next
  available price), never at the guaranteed stop.
- **Truncation test / CPCV / PBO / DSR / MCPT** — the five anti-overfitting
  gates (Chan leak check; purged combinatorial CV; Bailey–de Prado
  probability of backtest overfitting < 0.10; deflated Sharpe ≥ 0.95 from the
  trials registry; Masters permutation p < 0.05).
- **ETF trick** — de Prado's recurrence pricing a multi-product basket as a
  $1 total-return index `K_t`, with rebalancing costs `c_t` kept strictly
  external (booked as a negative dividend; embedding them fabricates
  short-spread profits). API: `ETFTrick(weights, open_prices, close_prices,
  rebalance_times, point_values=1.0, dividends=0.0, cost_rates=0.0,
  roll_times=None, aum0=1.0).compute()` → DataFrame with columns `K` and `c`.
- **Carver speed limit** — cost drag above ~0.08 SR/yr means turnover is
  consuming the edge.

## 10. Common mistakes (and how the engine catches them)

| Mistake | Symptom | Fix |
| --- | --- | --- |
| Reading a raw DataFrame instead of `get_latest_bars` | Truncation test FAILS; results too good | Only read through the data handler. |
| Emitting the same signal every bar | Hundreds of fills, huge commissions | Keep `_position` on the instance; emit only on target changes. |
| Trading a `None` bar | `TypeError`/`KeyError` in the strategy | Guard with `event.bars.get(symbol) is None`. |
| Negative/zero `strength` | Targets silently flip/shrink | Keep `strength ∈ (0, 1]`; direction comes from `signal_type`. |
| Running MCPT without a vectorized twin | `NotImplementedError` | Implement `vectorized_signals`, or use event-loop validation (block bootstrap). |
| Hand-feeding N to DSR | Selection bias understated | Log every trial to `TrialsRegistry`; use `dsr_from_registry`. |
| iid resampling of autocorrelated returns | Artificially smooth paths, underestimated downside | `autocorrelation_gate` first → block bootstrap / O-U paths. |
| Perfect stop assumptions | Tail risk understated, optimal-f unbounded | The engine fills gap-throughs at the next available price by design. |
