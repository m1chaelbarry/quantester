# Quantester Documentation

Welcome to the Quantester docs. Quantester is an institutional-grade,
event-driven quantitative backtesting engine written in pure Python. Five
decoupled modules — data, strategy, portfolio, execution, analytics —
communicate **only** through a strict four-event lifecycle
(`MarketEvent → SignalEvent → OrderEvent → FillEvent`) on a shared queue.

New here? Read the pages in this order:

1. [For traders](for-traders.md) — plain-language happy path if you are not a
   full-time coder (`run_backtest`, scoreboard, common mistakes).
2. [Getting Started](getting-started.md) — install, run the examples, run the tests.
3. [Architecture & Core Concepts](architecture.md) — how the event loop, the
   four-event lifecycle, and the temporal firewall work.
4. [Creating a Strategy & Backtesting It](tutorials/creating-a-strategy.md) —
   a beginner-friendly, step-by-step tutorial that takes you from an empty
   file to a validated backtest.
5. The module references below when you need exact signatures and options.

## Tutorials

| Guide | What you will learn |
| --- | --- |
| [Creating a Strategy & Backtesting It](tutorials/creating-a-strategy.md) | Write a custom `Strategy`, wire the engine, read results, generate a tearsheet, and run the mandatory validation gates. |
| [Validation Workflow](tutorials/validation-workflow.md) | The five anti-overfitting gates every strategy must pass before you trust it. |

## Module reference

| Module | Package | Docs |
| --- | --- | --- |
| Events | `quantester/events.py` | [Events & Constants](modules/events.md) |
| Engine | `quantester/engine.py` | [Architecture & Core Concepts](architecture.md) |
| Data handlers | `quantester/data` | [Data](modules/data.md) |
| Strategies | `quantester/strategy` | [Strategy](modules/strategy.md) |
| Portfolio | `quantester/portfolio` | [Portfolio, Sizing & Risk](modules/portfolio.md) |
| Execution | `quantester/execution` | [Execution & Transaction Costs](modules/execution.md) |
| Analytics | `quantester/analytics` | [Analytics & Tearsheets](modules/analytics.md) |
| Validation | `quantester/validation` | [Validation: Truncation, CPCV, PBO](modules/validation.md) |
| Monte Carlo | `quantester/montecarlo` | [Monte Carlo Suite](modules/montecarlo.md) |
| Utilities | `quantester/utils` | [Utilities: ETF Trick & Synthetic Data](modules/utils.md) |

## Reference material

- [Clean Code audit](clean-code-audit.md) — adherence vs maintainability rules,
  with prioritized follow-ups.
- [Glossary](glossary.md) — the quant terms used across the docs (Sharpe, DSR,
  PBO, MCPT, optimal-f, temporal firewall, …).
- [FAQ & Troubleshooting](faq.md) — common pitfalls and their fixes.

## The five invariants (read before changing code)

1. **Queue-only communication.** Components never call each other directly;
   everything flows through the event queue.
2. **Look-ahead safety is enforced, not conventional.** A state-based temporal
   firewall (two bar phases + `earliest_fill_time`) prevents any future data
   from leaking into decisions.
3. **Ledger accounting.** `fill_price` already embeds spread/slippage/impact;
   cash is charged `qty * fill_price + commission`. Slippage is recorded for
   analytics but never double-deducted.
4. **No silent history rewrites.** Multi-symbol data aligns on an outer-join
   timestamp union; missing bars mark an asset untradeable, they are never
   erased.
5. **Seeded randomness only.** All RNG uses `numpy.random.Generator(seed)`.
