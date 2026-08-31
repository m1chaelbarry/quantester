# Examples

Each strategy (or demo) lives in its own folder. Run every script from the
**repository root**. Shared OHLCV caches go in `examples/data/` (gitignored);
charts and tearsheets go in each folder’s `output/` (also gitignored).

## Start here (progressive tiers)

| Tier | Folder | What you learn | Command |
| --- | --- | --- | --- |
| **0** | [`hello_trader/`](hello_trader/) | One-call backtest + plain summary | `python examples/hello_trader/run.py` |
| **1** | [`ma_cross/`](ma_cross/) | Parameter sweep + tearsheet + DSR | `python examples/ma_cross/run.py` |
| **2** | [`custom_strategy/`](custom_strategy/) | Write your own rules | `python examples/custom_strategy/run.py` |
| **3** | [`market_data/`](market_data/) | Free bar feeds (Yahoo / crypto / Stooq / FMP / AKShare) | `python examples/market_data/run.py` |
| **3b** | [`macro_data/`](macro_data/) | World Bank + NBP overlays on a bar calendar | `python examples/macro_data/run.py` |
| **4** | [`production_research/`](production_research/) | Full research gates (PBO/MCPT/…) | `python examples/production_research/run.py` |

Read [`docs/for-traders.md`](../docs/for-traders.md) before Tier 4 if you are not a full-time coder.

## All demos

| Folder | What it demonstrates | Start here |
| --- | --- | --- |
| [`hello_trader/`](hello_trader/) | **Shortest path**: `run_backtest` + summary | `python examples/hello_trader/run.py` |
| [`ma_cross/`](ma_cross/) | SMA crossover + tearsheet + truncation + DSR | `python examples/ma_cross/run.py` |
| [`custom_strategy/`](custom_strategy/) | Tutorial companion: build a strategy from scratch | `python examples/custom_strategy/run.py` |
| [`market_data/`](market_data/) | Free bar feeds via `load_yahoo` / `load_crypto` / Stooq / FMP / AKShare | `python examples/market_data/run.py` |
| [`macro_data/`](macro_data/) | Macro overlays: World Bank + NBP + `as_daily_reindex` | `python examples/macro_data/run.py` |
| [`monte_carlo/`](monte_carlo/) | MCPT, resampling, drawdown bounds, O-U paths | `python examples/monte_carlo/run.py` |
| [`visualizations/`](visualizations/) | Chart gallery + interactive viewer | `python examples/visualizations/run.py` |
| [`production_research/`](production_research/) | **Reference workflow**: audit → grid → WF → PBO/DSR → MCPT → gates | `python examples/production_research/run.py` |
| [`ewmac_carry/`](ewmac_carry/) | EWMAC + crypto carry Combined Forecast, 13-stage gates | `python examples/ewmac_carry/run.py` |
| [`tranche_pullback/`](tranche_pullback/) | BTC tranche ladder, CCXT study, PBO/DSR grids | `python examples/tranche_pullback/run.py` |
| [`donchian_breakout/`](donchian_breakout/) | Hourly Donchian study + **daily multi-coin** sleeve | `python examples/donchian_breakout/run_multi_coin_viz.py` |

## New here?

```bash
python examples/hello_trader/run.py
# Then read docs/for-traders.md
```

## Production research (governance)

```bash
python examples/production_research/run.py
# Read examples/production_research/README.md
```

## Donchian breakout (multi-coin sleeve)

```bash
python examples/donchian_breakout/run_multi_coin_viz.py \
  --universe BTC/USD,ETH/USD,XRP/USD --risk-budget 0.02
```
