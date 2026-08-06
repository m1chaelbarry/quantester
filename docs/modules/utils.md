# Utilities: ETF Trick & Synthetic Data

Package: `quantester/utils`

## The ETF trick

`quantester/utils/etf_trick.py` (de Prado, AFML — the exact recurrence is
notebook-verified).

Builds a **total-return index of $1** invested in a multi-product basket —
e.g. a futures spread — letting strategies treat complex, rolling,
multi-legged positions as one simple cash instrument `K_t`:

```
K_t = K_{t-1} + Σ_i h_{i,t-1} · φ_{i,t} · (δ_{i,t} + d_{i,t}),   K_0 = aum0
```

where `h` are holdings, `φ` the USD value of one point of each instrument
(incl. FX), `d` the carry/dividend/coupon, and `δ` the price change
(close-to-close, or open-to-close on rebalance/roll bars).

```python
from quantester.utils.etf_trick import ETFTrick

trick = ETFTrick(
    weights,            # omega: (T x I) allocation vectors
    open_prices,        # o: (T x I) raw opens
    close_prices,       # p: (T x I) raw closes
    rebalance_times,    # B: rebalance/roll timestamps
    point_values=1.0,   # phi: scalar or (T x I)
    dividends=0.0,      # d: scalar or (T x I)
    cost_rates=0.0,     # tau_i per instrument
    roll_times=None,    # subset of B priced with next open (rolls)
    aum0=1.0,
)
out = trick.compute()   # DataFrame with columns K (index) and c (costs)
```

**The one rule that matters:** rebalancing costs `c_t` are computed but kept
**external** to `K_t`. Embedding them would let *shorting* the spread
generate fictitious profits at every rebalance. Treat `c` as a negative
dividend at the strategy level instead.

## Synthetic OHLCV data

`quantester/utils/synthetic.py` — deterministic (seeded) geometric Brownian
motion daily bars for examples and tests.

```python
from quantester.utils.synthetic import make_synthetic_ohlcv, write_csvs

df = make_synthetic_ohlcv(
    symbol="AAA", n_bars=750, s0=100.0,
    mu=0.08, sigma=0.20,          # annualized drift / volatility
    start="2020-01-01", seed=42,
    missing_every=None,           # k -> drop every k-th bar after warmup
)
paths = write_csvs({"AAA": df, "BBB": df2}, "examples/data")
```

- `missing_every=k` simulates illiquid/stress gaps for one symbol — exactly
  what the outer-join availability mask is built to handle.
- `write_csvs` persists `{symbol: DataFrame}` as schema-correct CSVs
  (`datetime,open,high,low,close,volume`) ready for
  `HistoricCSVDataHandler`, and returns the `{symbol: path}` map to pass
  straight into it.
