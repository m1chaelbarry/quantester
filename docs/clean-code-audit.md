# Clean Code Audit — Quantester

Audit date: 2026-08-09. Scope: `quantester/`, `examples/`, `tests/`.
Rubric: the Clean Code rules for AI code generation (names, functions,
comments, formatting, objects, errors, classes, tests, DRY/YAGNI/KISS,
smells, concurrency, design, docs).

**Overall: mixed — strong domain core, weaker at hotspots and examples.**

| Area | Grade | Summary |
| --- | --- | --- |
| Meaningful names | B | Strong domain vocabulary; some misleading names (`HistoricCSV…`, `examples` strategies, `utils`) |
| Functions | B− | Most helpers small; fill/ledger/gate methods and example `main`s run long |
| Comments | A− | Library WHY/contract docs are excellent; teaching examples are intentionally noisy |
| Formatting | B+ | Consistent modules; a few long lines and section walls |
| Objects / Demeter | B | Clean event DTOs; some dict bags and deep `predict_proba` access |
| Error handling | B | Public APIs raise well; event loop often soft-returns; two bare `except`s (fixed) |
| Classes / SRP | C+ | `BacktestEngine` is focused; `PortfolioManager` is a god class |
| Unit tests | B | F.I.R.S.T. mostly; many compound scenario tests; suite ~14s |
| DRY / YAGNI / KISS | B− / B / B | Example wiring was heavily duplicated (partly fixed); dual truncation modules |
| Concurrency | A | Almost none; RNG is seeded `Generator` throughout |
| System design | A− | Five-module queue design is exemplary; a few inheritance/dependency wrong turns |
| Documentation | A− | Docs + module contracts are strong |

---

## What already adheres well

1. **Queue-only architecture** — components talk through events; `BacktestEngine` is a thin dispatcher (Clean Code “one thing”).
2. **Intention-revealing domain names** — `delay`, `earliest_fill_time`, `emit_target`, sizers, gate statuses.
3. **Public validation raises with context** — strategy windows, cost knobs, signal types.
4. **Seeded RNG only** — no global `np.random`; Monte Carlo and synthetics take `Generator`/`seed`.
5. **Almost no commented-out code.**
6. **Module docstrings** document contracts and literature verification status.
7. **Tests are independent and self-validating** with readable names for firewall/fill semantics.

---

## Priority findings (before this PR’s fixes)

### P0 — correctness of mental model / SRP

| Finding | Evidence | Rule |
| --- | --- | --- |
| `PortfolioManager` god class | `portfolio/portfolio.py` (~450 lines): signals, ledger, round-trips, cash yield, risk liquidation | Classes / SRP |
| Live sizers lived beside the god class while Kelly/Vince lived in `sizing.py` | Dual “sizing” homes | Names / DRY |
| Strategies imported indicators from `visualization` | `donchian_breakout.py`, `tranche_pullback.py` | Dependency direction |
| Examples reached into `handler._data` | donchian + tranche scripts | Encapsulation / Law of Demeter |
| Homonym `BacktestResult` | `validation/truncation_test.py` vs future trader facade | Names / disinformation |
| Bare `except Exception: pass` | `engine.py`, `data/audit.py` | Error handling |

### P1 — readability / DRY

| Finding | Evidence | Rule |
| --- | --- | --- |
| Example wiring / metrics / friction cloned | donchian `run_ccxt` ↔ `run_mcpt`; unused `examples/_common.py` | DRY |
| Long functions | `evaluate_gates` (~110), `_try_fill` (~83), example `main`s 100–170 lines | Functions |
| Long parameter lists | `log_trial` (~19), `evaluate_gates` (~15), strategy `__init__`s | Functions / primitive obsession |
| Flag arguments | `buy_and_hold=True`, `invert=True`, `use_risk_overlays` | Functions |
| Compound tests | `test_sizers`, `test_data_audit_pass_warn_fail`, tranche latch test | Unit tests / one concept |
| `RetailCostModel(CostModel)` zeros parent knobs | LSP smell — parallel model via inheritance | Design |
| `"MARKET"` overloaded | event type vs order type constants | Names |

### P2 — polish

| Finding | Evidence | Rule |
| --- | --- | --- |
| `utils/` package name | synthetic + ETF trick | Meaningful names |
| `strategy/examples.py` holds production builtins | Public API under “examples” | Names |
| `HistoricCSVDataHandler` accepts DataFrames | Transport-tied name | Names |
| Soft `return` on untradeable bars | portfolio / strategies / simulator | Error handling (often intentional for masks) |
| Module-scoped pairs fixtures | `test_pairs_strategy.py` | Test independence risk |
| Suite > 10s guideline | viz + expensive fixtures ~14s | Fast tests |

---

## Fixes included in this PR (Boy Scout)

1. **Extract live sizers** → `quantester/portfolio/sizers.py`; re-export from `portfolio`.
2. **Move indicators** → `quantester/indicators/`; visualization re-exports; strategies import domain package.
3. **`DataHandler.source_ohlcv()`** — public research accessor; examples stop using `_data`.
4. **Rename** truncation `BacktestResult` → `EngineRunArtifacts` (alias kept).
5. **Narrow swallowed exceptions** in `engine.py` / `audit.py`.
6. **DRY donchian helpers** → `examples/donchian_breakout/_shared.py`; wire `examples/_common.py`.
7. **Split flag-driven CCXT runner** into `run_donchian` / `run_buy_and_hold`.

---

## Recommended follow-ups (not in this PR)

1. Split `PortfolioManager` into ledger + signal→order translator (+ keep risk overlays as collaborators).
2. Introduce `ResearchEvidence` / `TrialRecord` dataclasses to shrink `evaluate_gates` / `log_trial`.
3. Replace `RetailCostModel` inheritance with a shared cost Protocol.
4. Rename `HistoricCSVDataHandler` / `utils/` / move builtins out of `strategy/examples.py`.
5. Split compound tests; add `match=` on bare `pytest.raises`.
6. Extract remaining example `main` bodies (tranche parameter studies, multi-coin dashboard).
7. Clarify `_max_fill_qty` `None` (“unlimited”) vs `0.0` (“no liquidity”).

---

## Rule-by-rule quick answers

- **Meaningful names** — mostly yes in domain code; several framework/transport names mislead.
- **Functions** — mostly yes in library helpers; no for several pipelines and example scripts.
- **Comments** — yes for library; teaching examples deliberately violate “quiet code.”
- **Formatting** — yes overall.
- **Objects / Demeter** — mostly yes; watch dict lots, cost component dicts, sklearn chains.
- **Error handling** — mixed; public raises good, event soft-fails intentional but opaque.
- **Classes** — engine yes; portfolio/strategies/simulator heavy.
- **Unit tests** — good hygiene, weaker one-concept discipline and speed budget.
- **DRY/YAGNI/KISS** — library good; examples were the main DRY debt.
- **Concurrency** — yes (minimal surface, seeded RNG).
- **Design** — excellent event decoupling; fix cost inheritance + indicator dependency (done) + construction side-effects.
- **Docs** — yes; keep public APIs documented with examples.
