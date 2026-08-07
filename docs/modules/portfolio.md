# Portfolio, Sizing & Risk

Package: `quantester/portfolio`

The portfolio manager is the central gearbox of the engine: it translates raw
strategy signals into precise target orders, keeps the cash/holdings ledger,
and enforces risk overlays and margin.

## `PortfolioManager`

`quantester/portfolio/portfolio.py`.

```python
from quantester.portfolio.portfolio import PortfolioManager, PercentEquitySizer
from quantester.portfolio.risk import DailyDrawdownBreaker, MarginMonitor

portfolio = PortfolioManager(
    handler,
    initial_capital=100_000.0,
    sizer=PercentEquitySizer(0.9),                 # default: PercentEquitySizer(0.5)
    margin_monitor=MarginMonitor(max_leverage=2.0), # optional
    drawdown_breaker=DailyDrawdownBreaker(0.045),  # optional, see Risk overlays
    cash_yield_rate=0.02, idle_cash_fraction=0.5,  # optional idle-cash yield
)
```

Idle-cash yield (notebook-verified accounting): positive cash accrues
`cash_yield_rate × idle_cash_fraction` annualized, compounded by elapsed
calendar time between close-phase valuations — Kaufman credits **half** the
3-month T-bill rate on unallocated cash, and Carver requires including the
risk-free rate on undeployed cash in non-derivative backtests. Default
`cash_yield_rate=0.0` disables accrual; borrowed (negative) cash accrues
nothing.
```

### What it does on each event

| Event | Action |
| --- | --- |
| `SignalEvent` | Computes a reference price (current open for delay-0, latest close otherwise), asks the sizer for a **target quantity**, and emits an `OrderEvent` for the difference from the current position, stamped with `earliest_fill_time = timestamp_at_offset(signal.timestamp, signal.delay)`. Signals on untradeable symbols (no reference price) or with no future bar to fill on are dropped. |
| `FillEvent` | Updates cash and positions: `cash −= signed_qty × fill_price + commission`. Slippage is already embedded in `fill_price` and is **never** double-charged. Books completed round-trips into `trades`. |
| `MarketEvent` (close) | Marks positions to the bar close, appends to the equity and positions history, and runs the margin monitor — emitting liquidation orders (next bar's open) on a leverage breach. |

### Reading results after a run

| Attribute | Type | Contents |
| --- | --- | --- |
| `equity_curve` | `pd.Series` | Daily equity (cash + marked-to-market holdings). |
| `positions_history` | `pd.DataFrame` | Position per symbol per bar. |
| `fills` | `list[FillEvent]` | Every fill with price, commission, slippage. |
| `trades` | `list[dict]` | Round-trips: `symbol, t0, t1, qty, entry_price, exit_price, pnl`. |
| `cash`, `positions` | — | Final ledger state. |
| `equity`, `gross_exposure` | properties | Live account value and Σ\|qty·price\|. |

Open lots are tracked at volume-weighted average price; realized PnL is
booked on the closed portion of a position when it reduces or flips.

## Sizers

A sizer is any callable `(signal, portfolio, ref_price) -> target_qty`
returning the **target position** (signed). Wire it via
`PortfolioManager(sizer=...)`.

| Sizer | Target |
| --- | --- |
| `PercentEquitySizer(pct=0.5)` | `± pct × equity × strength / ref_price` (0 on `EXIT`). Compounds with account size. |
| `FixedUnitSizer(units=100.0)` | `± units × strength` shares. Used for fast-track parity checks. |

Custom example — cap position by both equity fraction and a volatility target:

```python
def vol_capped_sizer(signal, portfolio, ref_price):
    if signal.signal_type == "EXIT" or ref_price <= 0:
        return 0.0
    sign = 1.0 if signal.signal_type == "LONG" else -1.0
    dollar_target = min(0.2 * portfolio.equity, my_vol_budget(signal.symbol))
    return sign * dollar_target * signal.strength / ref_price
```

## Sizing engines (research tools)

`quantester/portfolio/sizing.py` — standalone functions for computing optimal
fractions/weights *outside* the event loop (e.g. to calibrate a sizer):

| Function | Formula / purpose |
| --- | --- |
| `kelly_fraction(win_rate, win_loss_ratio)` | Binary Kelly: `f* = p − q/b`. |
| `kelly_gaussian(mean, variance)` | Continuous Kelly: `f = μ/σ²` (0 if σ² ≤ 0). |
| `volatility_parity_weights(cov)` | `w_i ∝ 1/σ_i`, normalized — equal risk contribution. |
| `hpr(trades, f, worst_loss)` | Vince's Holding Period Return: `1 + f·(−Trade_i / WorstLoss)`. |
| `twr(trades, f, worst_loss)` | Terminal Wealth Relative: Π HPR (0 if any HPR ≤ 0 — ruin). |
| `optimal_f(trades, worst_loss=None, gap_stress=1.5, f_max=1.0)` | `f* = argmax TWR(f)` over `[0, f_max]`. `worst_loss` defaults to the historical worst loss × `gap_stress` — stressed *below* the nominal stop because stops do not guarantee fills through gaps (and unconstrained optimal-f is catastrophically sensitive to the max-loss estimate). Returns `f_max` when there are no losing trades. |
| `kakushadze_effective_returns(expected, linear_costs)` | `E_eff = sign(E)·max(|E| − τ, 0)` — apply to expected returns **before** weight optimization so edges smaller than linear costs are zeroed. |

## Risk overlays

`quantester/portfolio/risk.py`.

### `MarginMonitor(max_leverage=2.0, liquidation_fraction=0.5)`

Tracks `leverage = gross_exposure / equity` (∞ when equity ≤ 0). On a breach
(`leverage > max_leverage`) at close-phase valuation, the portfolio emits
orders shrinking **every** position by `liquidation_fraction`, fillable at the
next bar's open.

### `DailyDrawdownBreaker(max_intraday_dd=0.045)`

Account-level circuit breaker against the **daily opening balance** (the
prior trading day's last valuation — the exchange-rollover carry). When
close-marked equity falls `≥ max_intraday_dd` below it, the portfolio:

1. cancels every resting order across the book (`CANCEL`),
2. market-liquidates all positions at the next bar's open (retried each open
   until filled), and
3. suspends **all** signal flow until the next trading-day rollover resets
   the halt.

0.045 provides a 0.5% cushion under the 5% daily-loss limit common in
proprietary evaluations. Signals emitted by a halted strategy are dropped
entirely — the parked liquidation already flattens the book, so strategy
exits would double-sell into a short.

### `stabilized_covariance(returns)`

Ledoit-Wolf-shrunk covariance. Always use this (not the raw sample
covariance) before eigendecomposition: with many assets relative to
observations the raw matrix is ill-conditioned and its eigenvalues are noise.

### `spectral_risk_attribution(returns, weights=None)`

Attributes portfolio variance to orthogonal principal components
(López de Prado's spectral decomposition):

- `β_n = w′v_n` — portfolio loading on component n
- `R_n = β_n²·Λ_nn / σ²` with `σ² = w′Σw` — fraction of variance carried by PC n

Returns a DataFrame `[eigenvalue, beta_sq, risk_share]` indexed `PC1…PCN`,
sorted by eigenvalue. Weights default to equal weight. If `PC1` carries 90%
of your risk, you do not have N bets — you have one.
