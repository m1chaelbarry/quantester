# Examples

Each strategy (or demo) lives in its own folder. Run every script from the
**repository root**. Shared OHLCV caches go in `examples/data/` (gitignored);
charts and tearsheets go in each folder’s `output/` (also gitignored).

| Folder | What it demonstrates | Start here |
| --- | --- | --- |
| [`hello_trader/`](hello_trader/) | **Shortest path**: one-call `run_backtest` + plain summary | `python examples/hello_trader/run.py` |
| [`production_research/`](production_research/) | **Reference workflow**: audit → grid → WF → PBO/DSR → MCPT → gates | `python examples/production_research/run.py` |
| [`ma_cross/`](ma_cross/) | SMA crossover + tearsheet + truncation + DSR | `python examples/ma_cross/run.py` |
| [`custom_strategy/`](custom_strategy/) | Tutorial companion: build a strategy from scratch | `python examples/custom_strategy/run.py` |
| [`monte_carlo/`](monte_carlo/) | MCPT, resampling, drawdown bounds, O-U paths | `python examples/monte_carlo/run.py` |
| [`visualizations/`](visualizations/) | Chart gallery + interactive viewer | `python examples/visualizations/run.py` |
| [`market_data/`](market_data/) | Live yfinance / CCXT feeds | `python examples/market_data/run.py` |
| [`tranche_pullback/`](tranche_pullback/) | BTC tranche ladder, CCXT study, PBO/DSR grids | `python examples/tranche_pullback/run.py` |
| [`donchian_breakout/`](donchian_breakout/) | Hourly Donchian study + **daily multi-coin** sleeve | `python examples/donchian_breakout/run_multi_coin_viz.py` |

## New here?

```bash
python examples/hello_trader/run.py
# Then read docs/for-traders.md
```

## Production research (start here for governance)

```bash
python examples/production_research/run.py
# Read examples/production_research/README.md — the teaching doc for the full stack
```

## Donchian breakout (recommended multi-coin sleeve)

```bash
# Daily long-only multi-coin dashboard (risk-budgeted)
python examples/donchian_breakout/run_multi_coin_viz.py \
  --universe BTC/USD,ETH/USD,XRP/USD --risk-budget 0.02

# Same backtest without charts
python examples/donchian_breakout/run_multi_coin.py --universe BTC/USD,ETH/USD,XRP/USD

# Earlier hourly study (friction-dominated — kept for the negative result)
python examples/donchian_breakout/run_mcpt.py --full-history --bars 2500 --reps 200
python examples/donchian_breakout/run_viz.py
```
