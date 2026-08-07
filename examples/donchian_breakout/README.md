# Donchian breakout examples

Event-driven `DonchianBreakoutStrategy`: SMA(200) regime, prior-20 Donchian
entries, ADX(14) filter, delay-1 fills, SMA(20) / Donchian(10) exits, 2×ATR
protective stop, `FractionalRiskSizer`.

## Scripts

| Script | Purpose |
| --- | --- |
| `run.py` | Synthetic hourly path (smoke test) |
| `run_ccxt.py` | Real Bitstamp 1h BTC backtest |
| `run_mcpt.py` | Protocol II MCPT on hourly BTC |
| `run_viz.py` | Hourly BTC chart suite |
| `run_multi_coin.py` | **Daily long-only multi-coin book** |
| `run_multi_coin_viz.py` | **Multi-coin dashboard** (equity, corr, exposure, PC1) |

## Recommended configuration

Hourly both-sides BTC failed after friction (see `mcpt_results.txt`). The
configuration that survives costs and MCPT is:

- **Daily** bars, `long_only=True`
- Book **risk budget** split across names (e.g. 2% / N), not 2% per name
- Universe filtered to names with standalone edge (BTC, ETH, XRP in the
  Bitstamp study; LTC/BCH did not clear)

```bash
python examples/donchian_breakout/run_multi_coin_viz.py \
  --universe BTC/USD,ETH/USD,XRP/USD --risk-budget 0.02
```
