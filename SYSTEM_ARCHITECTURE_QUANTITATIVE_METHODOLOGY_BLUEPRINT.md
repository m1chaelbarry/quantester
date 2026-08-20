# System Architecture & Quantitative Methodology Blueprint

**Artifact:** Quantester (`quantester` v0.1.0; import path `quantester`)  
**Scope:** Domain-critical trading, risk, execution, and data-alignment logic as implemented in this repository. Boilerplate (loggers, CLI wrappers, generic config) is omitted.  
**Method:** Code-extracted definitions only. Mechanisms that a professional backtester would be expected to contain, but which are not implemented, are marked **[MISSING IN CODEBASE]**.  
**Verification legend (from module docstrings):** *notebook-verified* = checked against the user’s specialist-literature notebook; *canonical paper* = implemented from a cited paper/report and explicitly not notebook-covered.

---

## 1. Executive Summary & Tech Stack

### 1.1 High-level system design

Quantester is a **synchronous, single-threaded, event-driven** research backtester. Five components communicate **only** through a shared `queue.Queue` using the four-event lifecycle

\[
\text{MarketEvent} \;\to\; \text{SignalEvent} \;\to\; \text{OrderEvent} \;\to\; \text{FillEvent}.
\]

The loop lives in `quantester/engine.py::BacktestEngine.run_backtest`. Every master-calendar timestamp is processed in two phases (`open`, then `close`). Look-ahead safety is a **state-based temporal firewall** (strategy `delay` + order `earliest_fill_time` + phase-aware `DataHandler` visibility), not a hardcoded T+1.

A second, **vectorized fast-track** (`quantester/montecarlo/fast_track.py::fast_backtest`) exists solely for Monte Carlo / MCPT scale. Its documented parity contract is: targets decided at close \(T\) execute at open \(T+1\); the same `CostModel.adverse_adjustment` is applied; equity is \(E_t = \text{cash}_t + Q_t \cdot C_t\). It does **not** implement delay-0, stops, limits, or MOC.

### 1.2 Core technological choices and reasoning

| Layer | Choice | Architectural reason (from code/docs) |
| --- | --- | --- |
| Language / runtime | Python ≥ 3.12, CPython GIL | Research-speed iteration; the event loop is not a production matching engine. |
| Event bus | `queue.Queue` (in-process, blocking `get(False)`) | Deterministic single-thread drain; no multiprocessing, no asyncio. |
| Time-series containers | `pandas.DataFrame` / `Series` with timezone-aware UTC `DatetimeIndex` | Bar storage, outer-join calendars, rolling indicators. |
| Numerical kernels | `numpy` (`numpy.random.Generator` only — no global `np.random`) | Seeded Monte Carlo, permutation, bootstrap. |
| Scientific | `scipy` (bounded scalar `optimal_f`, `norm.cdf`/`ppf` for DSR, \(\chi^2\) for Ljung–Box) | Closed-form / optimizer calls, not a solver stack. |
| Covariance | `sklearn.covariance.LedoitWolf` | Spectral risk on a shrunk \(\Sigma\) when \(N \approx T\). |
| Pairs OLS | `sklearn.linear_model.LinearRegression` | Deterministic least squares for the GLD/GDX diagnostic. |
| Persistence | SQLite (`analytics/trials_registry.py`) + per-worker JSONL | Trial \(N\) and \(\sigma^2_{\mathrm{SR}}\) for DSR; parallel writers serialize then batch-import. |
| Plotting | Matplotlib (`Agg` for tearsheets) | Post-run only; visualization is forbidden inside the event loop. |

**Not used in the hot path:** Polars, Numba, Cython, order-book simulators, tick databases (Arctic/kdb), distributed executors. Heavy MCPT is intended to go through the NumPy fast-track, not 10,000 event-loop re-runs.

**In-memory model:** all OHLCV for the run is loaded at construction into `{symbol: DataFrame}`. There is no chunked/on-disk bar store. Memory scales as \(O(\sum_i T_i)\) plus the outer-join master index.

---

## 2. Architectural Pattern & Backtesting Engine

### 2.1 Engine type and state management

**Type:** discrete-event, bar-based, two-phase. **Not** a vectorized production backtester (the vectorized path is a Monte Carlo bypass with a narrower contract).

**Clock.** A master calendar \( \mathcal{T} = \bigcup_i \mathcal{T}_i \) (outer join of per-symbol timestamps). Pointer `StreamingDataHandler._ptr` advances one timestamp per outer iteration. Simulated time is `(timestamp, phase)` with `phase ∈ {open, close}`.

**Per-bar sequence** (`engine.py::run_backtest`):

1. `dh.advance()` → `(timestamp, bars)` where `bars[symbol]` is a full OHLCV `pd.Series` or `None`.
2. **Open phase**
   - `dh.set_phase("open", timestamp)`
   - Execution `on_market` with **full** bars (needed for open prints / cost proxies).
   - Strategies see a **redacted** `MarketEvent`: each available bar is `pd.Series({"open": float})` only (`_open_visible_bars`). High/low/close of the forming bar are not in the event payload.
   - Queue drain: delay-0 strategies `calculate_signals`; resulting orders/fills process in the same drain.
3. **Close phase**
   - `dh.set_phase("close", timestamp)`
   - Full OHLCV `MarketEvent`.
   - Queue drain: `PortfolioManager.update_portfolio_valuation` (mark-to-market, cash yield, margin, daily DD breaker) → execution stop/limit/MOC ledger → delay≥1 strategies.

**Component interfaces (no bypass):**

| Role | ABC | Required methods |
| --- | --- | --- |
| Data | `DataHandler` | `prime_data`, `advance`, `set_phase`, `get_latest_bars`, `get_current_open`, `timestamp_at_offset`, `bar_at` |
| Strategy | `Strategy` | `calculate_signals`; `matches_phase`; optional `vectorized_signals` |
| Portfolio | `Portfolio` | `update_from_signal`, `update_from_fill`, `update_portfolio_valuation` |
| Execution | `ExecutionHandler` | `on_market`, `execute_order` |

Downstream code is contractually forbidden from reading raw DataFrames for live signals (`DataHandler.source_ohlcv` is documented as research/post-run only).

### 2.2 Temporal firewall (delay and `earliest_fill_time`)

Let \(T_k \in \mathcal{T}\) be the signal timestamp.

- `delay = d ∈ ℕ` (strategy attribute; default `1`). Integer, non-boolean, `≥ 0` (engine validates).
- `fill_at ∈ {open, close}`.
- If `fill_at == close`: requires `d ≥ 1`; fill time is **this** bar’s close (MOC). Engine raises if delay-0 requests MOC (the close print does not yet exist at open).
- If `fill_at == open`: fill time = `timestamp_at_offset(T_k, d)` on the **master** calendar. If that offset is past the last bar, the order is dropped (end-of-sample).

**Visibility at open phase** (`get_latest_bars`):

\[
\text{visible}(s, T_k, \text{open}) = \{ \text{bars of } s \text{ with index } < T_k \}.
\]

The current open is available only via `get_current_open`. At close phase, `index ≤ T_k` is visible.

**Delay-0 intra-bar guard:** a delay-0 strategy may see bars \(< T_k\) plus \(T_k\)’s open. It cannot see \(T_k\) high/low/close. Execution still fills at \(T_k\) open using the full bar’s open print, with cost models evaluated on a **prior-bar or zero-range proxy** (see §4.3).

### 2.3 Data ingestion pipeline and memory management

**Normalized frame contract** (`data/streaming.py::normalize_ohlcv_frame`):

- Columns: `open, high, low, close, volume` (float).
- Unique, sorted, **timezone-aware UTC** `DatetimeIndex`.
- Naive indexes are localized as UTC (provider adapters must convert exchange-local wall times **before** this helper).
- Duplicate timestamps: default `on_duplicates='raise'`.

**Multi-symbol alignment:** outer-join union; a missing symbol at \(T_k\) stores `None` (availability mask). Timestamps are never dropped for incompleteness. A `None` bar is untradeable: strategies skip; portfolio `_reference_price` returns `None` and no order is emitted; execution `_try_fill` returns `False` and the order remains pending.

**Handlers sharing that stream:**

| Handler | Source | Adjustment / universe notes |
| --- | --- | --- |
| `HistoricCSVDataHandler` | `{symbol: path\|DataFrame}`, schema `datetime,open,high,low,close,volume` | Whatever the file contains. No CA engine. |
| `YFinanceDataHandler` | yfinance `Ticker.history` | Default `auto_adjust=True` (split/dividend-adjusted OHLC). Survivorship **not** handled. |
| `CCXTDataHandler` | `fetch_ohlcv` pagination | Drops the still-forming last candle by default (`drop_incomplete=True`). |
| `StooqDataHandler` / `FMPDataHandler` | HTTP EOD | Retrieval only; daily stamps UTC-localized. |
| `AKShareDataHandler` | A-share / US daily | `adjust ∈ {"", "qfq", "hfq"}` passed through to AKShare. |

Macro overlays (`macro/align.py::as_daily_reindex`) are **not** bar feeds: they ffill (default) a lower-frequency series onto a UTC calendar. `method='bfill'` raises `ValueError` (look-ahead if used as a trading feature); `method=None` leaves NaNs on non-observation days.

**Dataset audit** (`data/audit.py`): structural FAIL/WARN/PASS on OHLC inequalities, NaNs, tz, duplicates, gaps-vs-freq. Corporate-action, delisting, and survivorship checks are **documentation gates** (WARN until the caller sets boolean flags). They do not ingest CA or delist files.

**Bar construction (research, pre-engine)** — `data/bars.py`, notebook-verified intent:

- **Dollar bars:** emit when \(\sum_t p_t v_t \ge \theta\); leftover ticks always become a final bar (even if below \(\theta\)).
- **Tick rule:** \( b_0 = +1\); \( b_t = \operatorname{sign}(\Delta p_t) \) if \(\Delta p_t \neq 0\), else \( b_{t-1}\).
- **Imbalance bars:** \(\theta_T = \sum b_t\) (TIB) or \(\sum b_t v_t\) (VIB/DIB). After `warmup` completed bars, threshold = \(\max(\mathrm{EWMA}_{\mathrm{span}}(\text{lengths}) \cdot |\mathrm{EWMA}_{\mathrm{span}}(\text{per-tick flows})|, 10^{-12})\).

**Deviation vs AFML (material):** the docstring cites \( \mathbb{E}_0[\theta_T] = \mathbb{E}_0[T]\cdot(2\mathbb{P}[b=1]-1) \). The implementation concatenates **tick-level** flows across completed bars and takes \(|\mathrm{EWMA}|\) of that series, not a bar-level estimate of \(\mathbb{P}[b=1]\) or of \(2v^+ - \mathbb{E}[v]\). Warmup uses a constant `initial_expected_len`. Residual ticks always flush a last bar.

---

## 3. Quantitative Methodology & Data Integrity

### 3.1 Mathematical definitions of signals and indicators (as coded)

Unless noted, indicators are pure functions on `pd.Series` (`indicators/__init__.py`). Live strategies must call them on `get_latest_bars` windows. Verification: *not notebook-covered* — Wilder (1978), Appel (1979), Bollinger, Donchian.

Let \(C_t, H_t, L_t, O_t\) be close/high/low/open.

| Indicator | Definition in code | Types |
| --- | --- | --- |
| SMA(\(n\)) | `rolling(n).mean()` | `pd.Series[float]` |
| EMA(\(n\)) | `ewm(span=n, adjust=False).mean()` | `pd.Series[float]` |
| RSI(\(n\)) | \(\Delta C_t\); gain \(=(\Delta C)_+\), loss \(=(-\Delta C)_+\); Wilder \(\alpha=1/n\); \(\mathrm{RSI}=100-100/(1+RS)\); all-gain → 100, both-zero → 50 | `[0,100]` |
| MACD | \(\mathrm{EMA}_{12}-\mathrm{EMA}_{26}\); signal = EMA\(_9\) of line; histogram = line − signal | DataFrame |
| Bollinger(\(n,k\)) | mid = SMA\(_n\); \(\sigma\) = `rolling.std(ddof=0)` (population); bands = mid \(\pm k\sigma\) | DataFrame |
| ATR(\(n\)) | \(\mathrm{TR}_t=\max(H_t-L_t,|H_t-C_{t-1}|,|L_t-C_{t-1}|)\); Wilder \(\alpha=1/n\) | `pd.Series` |
| ADX(\(n\)) | \(+\mathrm{DM}/-\mathrm{DM}\) Wilder; \(+\mathrm{DI},-\mathrm{DI}=100\cdot\mathrm{DM}/\mathrm{ATR}\); \(\mathrm{DX}=100\,|+\mathrm{DI}--\mathrm{DI}|/(+\mathrm{DI}+-\mathrm{DI})\); ADX = Wilder of DX | DataFrame |
| Donchian(\(n\), `shift=1`) | \( B^{\uparrow}_t=\max(H_{t-1},\ldots,H_{t-n}) \), \( B^{\downarrow}_t=\min(L_{t-1},\ldots,L_{t-n}) \) | DataFrame |
| Rolling vol(\(n\), \(P=252\)) | \(\mathrm{std}_{n,\mathrm{ddof}=1}(\log(C_t/C_{t-1}))\cdot\sqrt{P}\) | `pd.Series` |

**Moving-average cross** (`strategy/examples.py::crossover_positions`):

\[
d_t = \mathrm{SMA}_{\mathrm{fast}}(C)_t - \mathrm{SMA}_{\mathrm{slow}}(C)_t,\quad
\text{target}_t =
\begin{cases}
+1 & d_t>0 \land d_{t-1}\le 0 \\
-1 & d_t<0 \land d_{t-1}\ge 0 \\
\text{held} & \text{otherwise (ffill)}
\end{cases}
\]

Event form uses the same cross test on `get_latest_bars(..., slow+2)`. Default `delay=1`. Direction can clip to long-only / short-only.

**Donchian breakout** (`strategy/donchian_breakout.py`), delay=1:

- Regime: bullish iff \(C_t > \mathrm{SMA}_{200}(t)\); bearish iff \(C_t < \mathrm{SMA}_{200}(t)\).
- Enter long iff \(C_t > B^{\uparrow}_{20,t}\) and bullish and \(\mathrm{ADX}_{14}(t) > 25\).
- Protective floor latched at **fill-bar open** \(\mp 2\cdot\mathrm{ATR}_{14}\) (ATR from the **signal** bar).
- Touch is observed on the bar’s high/low at **close**; exit is delay=1 at the **next open** (explicitly not same-bar close after seeing the extreme).
- Additional exits: close through SMA\(_{20}\) or opposite Donchian(10) trail.
- Size via `FractionalRiskSizer` reading `signal.stop_distance = 2\cdot\mathrm{ATR}_{14}\).
- **No vectorized twin** (path-dependent stop latch).

**Pairs / log-spread** (`strategy/pairs_trading.py`), Chan rolling OLS, *not notebook-covered*:

\[
\ln P^Y_t = \alpha_t + \beta_t \ln P^X_t + z_t,
\quad
s_t = \frac{z_t - \hat\mu_{z,t}}{\hat\sigma_{z,t}}
\]

with \((\alpha_t,\beta_t)\) from OLS on the last `ols_window=252` **inner-joined** log-closes visible under the firewall; \(\hat\mu,\hat\sigma\) on the last `zscore_window=20` residuals (`std` ddof=1). State machine: enter long spread if \(s_t < -2\), short if \(s_t > +2\); exit to flat iff \(|s_t|\le 0.5\); **no direct long↔short flips**. Both legs emit in the same event cycle. If either bar is `None`, hold state (no fabricated spread).

**Hedge-ratio sizing gap:** each leg is independently passed to the portfolio sizer with `strength=1.0`. There is **no** \(\beta\)-share or dollar-neutral mapping of quantities. A `PercentEquitySizer` will size each leg as \(\mathrm{pct}\cdot E / P_i\), which is not a cointegrating residual position.

**Tranche pullback** (`strategy/tranche_pullback.py`):

- Arm only if \(C_t > \mathrm{SMA}_{200}\).
- Peak \(P^{\mathrm{peak}}_t = \max(C_{t-19},\ldots,C_t)\) (rolling 20 including current).
- Limits \( T_k = P^{\mathrm{peak}} - k\cdot 1.5\cdot\mathrm{ATR}_{14}\), \(k=1,2,3\); fractions \((0.25,0.35,0.40)\).
- Re-anchor every `reanchor_every` bars until first fill; then freeze.
- Hard stop \( P^{\mathrm{peak}} - 5\cdot\mathrm{ATR}_{14} \); touch on low at close → delay=1 next open.
- Mean-reversion exit: \(C_t \ge \mathrm{SMA}_5\).
- **No vectorized twin.**

**Triple-barrier meta-labels** (`strategy/meta_labeling.py`), AFML ch.3 notebook-verified **intent**, but path is **close-only**:

For event at index \(i_0\), side \(s\in\{+1,-1\}\), entry \(C_{i_0}\):

\[
\mathrm{TP} = C_{i_0}(1 + s\cdot\mathrm{tp\_pct}),\quad
\mathrm{SL} = C_{i_0}(1 - s\cdot\mathrm{sl\_pct}).
\]

Path = `close.iloc[i0+1 : i0+max_holding]`. Hit TP vs SL compared by first `True` timestamp (`idxmax` on boolean). If neither barrier: \( y=1 \) iff \((C_{\mathrm{end}}-C_{i_0})\cdot s > 0\).

**Limitation vs AFML:** intra-bar high/low barrier touches are invisible. This is **not** look-ahead; it is barrier misspecification (under-detection of TP/SL).

Optional size transform (AFML ch.10, *not notebook-covered*):

\[
z = \frac{p-1/2}{\sqrt{p(1-p)}},\quad m = 2\Phi(z)-1,
\]

then `strength = max(0, m)` so side is never inverted.

**ETF trick** (`utils/etf_trick.py`), notebook-verified:

\[
K_t = K_{t-1} + \sum_i h_{i,t-1}\,\phi_{i,t}\,(\delta_{i,t}+d_{i,t}),\quad K_0=\mathrm{aum0},
\]

\[
\delta_{i,t} =
\begin{cases}
p_{i,t}-o_{i,t} & \text{if } t-1 \in B \text{ (rebalance/roll)} \\
p_{i,t}-p_{i,t-1} & \text{otherwise.}
\end{cases}
\]

On rebalance, \( h_{i,t} = \omega_{i,t} K_t / (P_{i,t}\,\phi_{i,t}\,\sum_j|\omega_j|) \), with roll bars using \( o_{i,t+1} \). Rebalancing cost \(c_t\) is **returned in a separate column and not subtracted from \(K_t\)** (de Prado: embedding \(c_t\) fabricates short-spread profits). Booking \(c_t\) as a negative dividend is left to the strategy layer.

### 3.2 Data leakage and look-ahead prevention

**Enforced in the engine**

- Phase-restricted strategy dispatch (`matches_phase`).
- Open-phase event bars stripped to `open`.
- `get_latest_bars` excludes the current bar at open.
- Open-phase Kaufman/Kyle **must not** use the same bar’s high/low (`SimulatedExecutionHandler._cost_proxy_bar`: prior bar via `get_latest_bars`, else zero-range proxy at the open).
- Fast-track `_open_cost_bar(i)` uses bar \(i-1\) range (or zero-range on \(i=0\)).
- Chan truncation diagnostic: full vs last-\(N\)-bars-dropped position ledgers must match on the overlap (`|a-b| \le \mathrm{atol}`).

**Not a proof of zero leakage.** Truncation is a diagnostic. Strategies that call `source_ohlcv` or compute indicators on a full future-aware frame bypass the firewall. Visualization helpers are post-run.

**CPCV / Purged K-Fold** (`validation/cpcv.py`), notebook-verified, **label-interval** overlap (not a fixed bar count):

Train label \([t_{i0}, t_{i1}]\) vs test \([t_{j0}, t_{j1}]\) is purged if any of:

1. \( t_{j0} \le t_{i0} \le t_{j1} \)
2. \( t_{j0} \le t_{i1} \le t_{j1} \)
3. \( t_{i0} \le t_{j0} \le t_{j1} \le t_{i1} \)

Embargo: drop train with \( t_{j1} \le t_{i0} \le t_{j1} + h \), \( h = \lfloor \mathrm{pct\_embargo}\cdot T \rfloor \) converted to time via **median** \(\Delta t\) on a `DatetimeIndex`. If `t1` is omitted, label end = sample timestamp (point labels). Combinatorial path count is coded as `int(k_test / n_groups * n_splits)` i.e. \(\lfloor (k/N)\,C(N, N-k)\rfloor\).

**CPCV fold test interval:** `PurgedKFold` treats the test fold as a single interval \([t_0^{\mathrm{first\ test}}, t_1^{\mathrm{last\ test\ row}}]\). If `t1` is not monotone in index, a middle test row whose label ends *after* the last row’s `t1` is not fully covered — a residual overlap hole relative to AFML’s per-label purge.

**PBO / CSCV** (`validation/pbo.py`), notebook-verified Bailey–de Prado:

Matrix \(M \in \mathbb{R}^{T\times N}\) of synchronous trial PnL. Even \(S\) blocks; all \( C(S, S/2) \) splits. \( n^* = \arg\max_n R_n^{\mathrm{train}} \); \(\bar\omega_c = \mathrm{Rank}(R^{c}_{n^*})/(N+1)\); \(\lambda_c = \log(\bar\omega_c/(1-\bar\omega_c))\); \(\mathrm{PBO}=\mathbb{P}(\lambda<0)\) as the empirical fraction \(\lambda_c<0\). Gate `PBO_GATE = 0.10`. Requires \(N\ge 2\) (single trial would spuriously report 0). Performance default: Sharpe of the PnL **block** (simple mean/std, **not** annualized — ranking-invariant).

**Code defect:** `validation/truncation.py` empty-overlap branch references `n_truncate` (undefined) instead of `n_truncated` → `NameError` if the two runs share no index.

### 3.3 Survivorship bias and corporate actions

| Mechanism | Status |
| --- | --- |
| Point-in-time universe membership file (index constituents through time) | **[MISSING IN CODEBASE]** — audit flag `historical_universe_documented` only. |
| Delisting / halt feed; residual value / last trade | **[MISSING IN CODEBASE]** — if a CSV simply ends, the symbol becomes `None` on later union timestamps (untradeable), which is correct **only if** dead names are present in the input map. Loading today’s survivors reproduces classic survivorship bias. YFinance docstring states this explicitly. |
| Split / dividend as first-class events (adjust positions, book cash dividends, gap prices) | **[MISSING IN CODEBASE]** — no `CorporateActionEvent`, no position multiplier, no cash dividend in `update_from_fill`. |
| Adjusted vs raw semantics | Provider-dependent. YFinance default `auto_adjust=True` bakes splits/dividends into OHLC (total-return-like prices) **without** booking dividend cash in the ledger → double-counting risk if the user also models dividends, or understated cash if they expected unadjusted + cash. CSV/Stooq/FMP: pass-through. AKShare: `qfq`/`hfq` flags. Audit WARNs until `adjustment_policy` is documented. |
| Symbol mapping (FIGI/permno vs ticker reuse) | **[MISSING IN CODEBASE]** — symbols are opaque strings. |

**Risk of absence:** equity backtests on “current S&P” CSVs will overstate returns and understate gap risk on names that died. Crypto CCXT feeds do not have corporate actions but **do** have delistings/renames that are similarly unmodeled.

---

## 4. Market Microstructure & Execution Assumptions

### 4.1 Order matching logic

`SimulatedExecutionHandler` is a **resting-order ledger**, not an L2 matching engine. Eligible fills:

| `order_type` | Eligible phase | Reference price \(R\) |
| --- | --- | --- |
| `MARKET` | **open only** | \(O_t\) of the fill bar |
| `STOP` | close only | see below |
| `LIMIT` | close only | see below |
| `MOC` | close of `earliest_fill_time` **only**; stale MOCs expire | \(C_t\) |
| `CANCEL` | immediate | purges **all** pending orders for the symbol (including residual MARKET slices) |

**Stop (gap-through, never a perfect stop)** — Cross-Ref-2 §4.2:

- Buy stop: if \( H_t < P_{\mathrm{stop}} \), no fill; else \( R = \max(P_{\mathrm{stop}}, O_t) \).
- Sell stop: if \( L_t > P_{\mathrm{stop}} \), no fill; else \( R = \min(P_{\mathrm{stop}}, O_t) \).

Overnight/limit-down gaps fill at the open, not at the stop. This is intentional: guaranteed stops unbind Vince optimal-\(f\).

**Limit (rest until touched):**

- Buy: if \( L_t > P_{\mathrm{lim}} \), no fill; else \( R = \min(O_t, P_{\mathrm{lim}}) \).
- Sell: if \( H_t < P_{\mathrm{lim}} \), no fill; else \( R = \max(O_t, P_{\mathrm{lim}}) \).

Gaps through the limit fill at the open (price improvement or worse, depending on side).

**MOC:** all-or-nothing under a participation cap. If `requested > max_fill_qty`, the order is **rejected entirely** (no silent residual). Market orders under `liquidity_policy='partial'` clip and leave a residual pending until a later **open**.

**Portfolio emission of stops:** `STOP_ORDER` is implemented in the simulator and tested, but `PortfolioManager.update_from_signal` **never constructs a `STOP_ORDER`**. Protective stops in Donchian/tranche are **strategy-level**: observe touch at close, emit `EXIT` market with `delay=1`. That is economically a stop **plus one bar of gap risk**, not a resting stop that can fill intra-bar at the stop/open.

**Intrabar path assumption:** OHLC is a four-print summary. Touch tests use high/low existence, not tick path, volume-at-price, or open→high vs open→low ordering. A bar that both gaps through a buy stop and trades below it still fills at \(\max(P_{\mathrm{stop}},O_t)\).

### 4.2 Fill price, ledger, and cost split

FillEvent fields (`events.py`):

- `fill_price` — already **includes** adverse adjustment \(\phi\) per share.
- `commission` — \(c_t\) (proportional/fixed).
- `slippage_cost` — \(\phi_t = \text{adjustment}\times q\) in currency, **analytics only**.
- `reference_price` — pre-cost \(R\).

\[
R_{\mathrm{fill}} =
\begin{cases}
R + a & \text{BUY} \\
\max(R - a, 10^{-12}) & \text{SELL}
\end{cases}
\qquad a = \texttt{cost\_model.adverse\_adjustment}(R, q, \text{bar}).
\]

Cash (`portfolio.py::update_from_fill`):

\[
\mathrm{cash} \leftarrow \mathrm{cash} - s\cdot q\cdot R_{\mathrm{fill}} - c_t,
\quad s = +1\ \mathrm{BUY},\ -1\ \mathrm{SELL}.
\]

\(\phi_t\) is **not** subtracted again. Round-trip PnL uses fill prices (hence embeds \(\phi\)) minus entry+exit commissions (pro-rated on partial closes).

### 4.3 Slippage / impact models (deterministic)

All models are **deterministic functions of (order, bar)** so event engine and fast-track can share them. **No** simulated noise-trader flow \(dy\) (explicitly rejected: Kyle \(dy\) is unobservable and would break reproducibility).

**`CostModel`** (defaults: `fixed_commission=1.0`, `per_share_commission=0.005`, `spread_pct=0.0005`, `slippage_vol_coef=0.1`, `impact_coef=0.1`):

\[
\begin{aligned}
c_t &= \mathbf{1}_{q>0}\bigl(c_{\mathrm{fix}} + c_{\mathrm{ps}}\cdot q\bigr)
\quad\text{(price-independent; `price` ignored)} \\
a_{\mathrm{spread}} &= R \cdot \texttt{spread\_pct}/2 \\
a_{\mathrm{Kaufman}} &= R \cdot \lambda_K \cdot \frac{\max(H-L,0)}{R} = \lambda_K \max(H-L,0) \\
\lambda_{\mathrm{Kyle}} &= \lambda_I \cdot \frac{(H-L)/R}{V},\quad
a_{\mathrm{Kyle}} = R \cdot \lambda_{\mathrm{Kyle}} \cdot q
= \lambda_I \cdot \frac{H-L}{V}\cdot q \\
a &= a_{\mathrm{spread}} + a_{\mathrm{Kaufman}} + a_{\mathrm{Kyle}}.
\end{aligned}
\]

Kaufman form and Amihud-style \(\lambda\) estimation: *not notebook-covered* (module docstring).

**`ConservativeFrictionCostModel`:** \( a = m \cdot a_{\mathrm{spread}} \) (default \(m=2\) ⇒ full spread); \( c_t = m \cdot \mu_{\mathrm{fee}} \cdot q \cdot R \) (default \(\mu_{\mathrm{fee}}=4\) bps). Kaufman/Kyle zeroed. Applied to **market and limit** fills (taker-grade on makers: deliberate pessimism).

**`RetailCostModel`** (OHLCV-only; *not notebook-covered*):

\[
\begin{aligned}
a_{\mathrm{spread}} &= R \cdot \frac{\mathrm{spread\_bps}}{10^4}/2 \\
\mathrm{vol\_bps} &= \frac{H-L}{R}\cdot 10^4 \\
a_{\mathrm{vol}} &= R \cdot \frac{f_{\mathrm{vol}}\cdot\mathrm{vol\_bps}}{10^4} \\
\pi &= q/V \\
a_{\mathrm{imp}} &= R \cdot \frac{f_{\mathrm{imp}}\cdot\mathrm{vol\_bps}\cdot \pi^{\gamma}}{10^4}
\quad \gamma\in(0,2],\ \mathrm{default}\ 0.5 \\
a &= a_{\mathrm{spread}}+a_{\mathrm{vol}}+a_{\mathrm{imp}} \\
q_{\max} &= V \cdot \pi_{\max}\quad (\mathrm{default}\ 0.05).
\end{aligned}
\]

Presets `retail_cost_scenario("BASE"|"CONSERVATIVE"|"STRESS")` raise spread, vol factor, impact, exponent, and tighten \(\pi_{\max}\).

**Open-phase cost proxy:** \(H,L,C,V\) from the **previous visible bar**; open from the fill bar. If no prior bar: \(H=L=C=O_t\) (zero Kaufman/Kyle range). Same-bar range is never used at open.

### 4.4 Latency, queue position, TCA

| Mechanism | Status |
| --- | --- |
| Order-to-venue latency (ms), jitter, colocation | **[MISSING IN CODEBASE]** — `earliest_fill_time` is bar-time, not clock-time. Delay-0 fills at the same open print as the decision. |
| Queue position / priority / hidden liquidity | **[MISSING IN CODEBASE]** |
| Maker vs taker fee schedules (except ConservativeFriction’s uniform taker pessimism) | **[MISSING IN CODEBASE]** as a book model |
| Level-2 / auction imbalance | **[MISSING IN CODEBASE]** |
| Tick size / lot size / integer shares | **[MISSING IN CODEBASE]** — `quantity: float` |
| Perpetual funding / borrow / locate | **[MISSING IN CODEBASE]** — docstring: “No borrow charge on negative cash — documented simplification.” |
| TCA beyond fill diagnostics | Partial: `ExecutionDiagnostics` records participation, impact bps, partial/reject counts. No implementation-shortfall vs arrival-price VWAP benchmark series. |

**Risk of absence:** delay-0 equity strategies are simulated as if the open print is freely tradable at decision time with only spread/Kaufman/Kyle. HFT and open-auction strategies will be optimistically filled.

**Fast-track vs event costs:** fast-track uses `adverse_adjustment` and `commission(..., price=open)` but **does not** implement stops/limits/MOC/delay-0. Liquidity cap uses `max_fill_quantity(volume[i])` on the **fill** bar’s volume (event engine uses the same). Fast-track Sharpe uses **simple** `pct_change` × \(\sqrt{252}\); event analytics use **log** Sharpe (§6) — metric parity is **not** guaranteed even when equity paths match.

---

## 5. Risk Management & Portfolio Allocation

### 5.1 Position sizing (event loop)

Sizers are callables `(signal, portfolio, ref_price) -> target_qty` (`portfolio/sizers.py`). Default in `PortfolioManager`: `PercentEquitySizer(0.5)`.

Let \(E\) = `portfolio.equity` (cash + Σ \(q_i P_i^{\mathrm{last}}\)), \(P\) = reference price, \(k\) = `signal.strength > 0` (EXIT excluded).

| Sizer | Target quantity |
| --- | --- |
| `FixedUnitSizer(u)` | \(\pm u\cdot k\) |
| `PercentEquitySizer(rho)` | \(\pm (E\cdot\rho\cdot k)/P\) |
| `FractionalRiskSizer(f)` | \(\pm (E\cdot f) / \delta_{\mathrm{stop}}\) requiring `signal.stop_distance>0` |

Reference price: `limit_price` if set (size at the limit); else delay-0 → `get_current_open`; else last visible **close**. If untradeable (`None`/empty), no order.

**Research math not wired into the event loop** (`portfolio/sizing.py`):

- Discrete Kelly: \( f^* = p - q/b \), \( b=\) win/loss ratio. No fraction cap, no Cox–Kelly discrete correction.
- Continuous Kelly: \( f = \mu/\sigma^2 \) (Gaussian, variance not volatility).
- Volatility parity: \( w_i \propto 1/\sigma_i \), \(\sigma_i=\sqrt{\Sigma_{ii}}\), renormalized. **Not** inverse-covariance / ERC with correlations.
- Vince optimal-\(f\) (notebook-verified):

\[
\mathrm{HPR}_i(f)=1+f\cdot\frac{-\mathrm{Trade}_i}{W},\quad
\mathrm{TWR}(f)=\prod_i \mathrm{HPR}_i,\quad
f^*=\arg\max_{f\in[0,f_{\max}]} \mathrm{TWR}(f)
\]

with \(W = \mathrm{gap\_stress}\cdot\min_i\mathrm{Trade}_i\) (default `gap_stress=1.0` — raw BiggestLoss per ruling D3; the 1.5 stress is opt-in) if any loss exists; TWR = 0 if any HPR ≤ 0 (ruin). Optimizer: `scipy.optimize.minimize_scalar` bounded. Unconstrained \(f^*\) is **not** attached to `PortfolioManager`.

- Kakushadze: \( E_{\mathrm{eff}} = \operatorname{sign}(E)\max(|E|-\tau,0) \) applied **before** any weight optimization — also not in the live sizer path.

**Default-sizer discrepancy:** `PortfolioManager` defaults to `PercentEquitySizer(0.5)`. The one-call API `quantester.simple.run_backtest` defaults to `PercentEquitySizer(0.9)` and `CostModel()` (including `fixed_commission=1.0` currency per fill). On crypto notionals a $1 ticket fee is often negligible; on fractional-share equity tests it is not.

**Procyclicality:** percent-equity and fractional-risk both scale with mark-to-market \(E\), including unrealized PnL. No volatility targeting overlay in the event loop.

**Pairs:** see §3.1 — independent per-leg percent equity is not a residual-weighted book.

### 5.2 Stops, take-profits, margin, circuit breakers

| Control | Implementation |
| --- | --- |
| Native resting stop from portfolio | **[MISSING IN CODEBASE]** as a portfolio product (simulator supports `STOP_ORDER`; PM never emits it). |
| Strategy stops / TP | Donchian ATR floor + trail + SMA exit; tranche 5×ATR + SMA\(_5\); meta-labeling barriers are **labels**, not live orders. |
| Intra-bar stop fill | **[MISSING IN CODEBASE]** for strategy stops (always delay=1 next open after close observation). |
| Daily DD breaker | `DailyDrawdownBreaker`: trip iff \((E_{\mathrm{day\ open}}-E_t)/E_{\mathrm{day\ open}} \ge \delta\) (default \(\delta=0.045\)). Day open = last equity of **previous calendar date** (UTC `.date()`). Halt: cancel all, market-liquidate at next open, **drop all strategy signals** until next date. First backtest day seeds baseline from the first valuation (cannot trip until equity falls vs that same-day print). *Not notebook-covered* (prop-eval spec). Crypto 24/7: “day” is UTC midnight, not session. |
| Margin | `MarginMonitor`: \(\ell = G/E\) with \(G=\sum |q_i|P_i\), \(E\le 0 \Rightarrow \ell=\infty\). Breach if \(\ell > \ell_{\max}\) (default 2). While `restricted`, **increasing** \(|q|\) is blocked; shrinks/exits allowed. Liquidation targets: \( q' = q(1-\lambda) \) (default \(\lambda=0.5\)). Re-issued every close while still breached (partial fills). Restriction clears only when \(\ell \le \ell_{\max}\). No per-asset haircuts, SPAN, or overnight vs intraday rates. |
| Cash yield | \( \Delta\mathrm{cash} = \mathrm{cash}\cdot r \cdot \eta \cdot \Delta\mathrm{days}/365 \) on **positive** cash only (`idle_cash_fraction` \(\eta\) default 0.5; Kaufman half T-bill / Carver RF inclusion; notebook-verified). Simple, 365-day, not 252, not continuous. **No** debit interest. |
| Dividends / borrow / short rebate | **[MISSING IN CODEBASE]** in the ledger. |
| Multi-currency / FX conversion | **[MISSING IN CODEBASE]** in portfolio; NBP FX is a macro overlay only. |

Mark-to-market uses **close** (`last_prices[symbol]=close`). Equity identity: \( E = \mathrm{cash} + \sum q_i P_i^{\mathrm{close}} \) (`accounting_invariant`).

---

## 6. Performance Metrics & Statistics

Module: `analytics/performance.py` unless noted. `TRADING_DAYS = 252` is **hardcoded** as the annualization period count (not detected from bar frequency).

### 6.1 Return representations (`analytics/returns.py`)

\[
r_t = \frac{P_t}{P_{t-1}}-1,\quad
\ell_t = \log\frac{P_t}{P_{t-1}},\quad
\ell=\log(1+r),\quad r=e^{\ell}-1.
\]

Wealth: \( W = W_0 \prod(1+r) \) or \( W_0 \exp(\sum \ell) \). Mixing P&L, simple, and log is explicitly forbidden in the docstring.

### 6.2 Event-loop headline metrics

Let \(E_t\) be the close-phase equity series (`pd.Series`, DatetimeIndex). Log returns \(\ell_t = \log(E_t/E_{t-1})\), `dropna` after a shift.

**Annualized Sharpe** (`annualized_sharpe`):

\[
\widehat{\mathrm{SR}}
= \frac{\bar\ell - R_f^{\mathrm{daily}}}{s_{\ell}}\,\sqrt{P},
\quad P=\texttt{periods}\ \mathrm{default}\ 252.
\]

- \(\bar\ell, s_{\ell}\): pandas `mean`, `std` (**sample**, `ddof=1`).
- \( R_f^{\mathrm{daily}} \) default **0**. This is a **per-bar** rate, not an annual yield. Passing 0.05 (5% annual) would be a unit error.
- If \( n<2 \) or \( s_{\ell}=0 \): return `0.0`.
- **Log** Sharpe, not simple-return Sharpe (Lo 2002 / Bailey conventions differ; here it is log by construction).

**Max drawdown:**

\[
\mathrm{HWM}_t = \max_{s\le t} E_s,\quad
\mathrm{DD}_t = E_t/\mathrm{HWM}_t - 1,\quad
\mathrm{MDD} = \min_t \mathrm{DD}_t.
\]

Peak = `equity.loc[:trough].idxmax()`; duration = **calendar** `.days` from peak to first recovery \(E \ge E_{\mathrm{peak}}\), or to last index if never recovered. For hourly series this is wall-clock days, not bar count.

**Calmar:**

\[
Y = \max\bigl(n_E / P,\ 10^{-12}\bigr),\quad
R_{\mathrm{ann}} = (E_T/E_0)^{1/Y}-1,\quad
\mathrm{Calmar} = R_{\mathrm{ann}} / |\mathrm{MDD}|.
\]

\( n_E = \mathrm{len}(\mathrm{equity}) \) after dropna — **bar count / 252**, not calendar years. Intraday or weekend-inclusive crypto will mis-annualize. If MDD = 0: `+inf` if \(R_{\mathrm{ann}}>0\) else 0.

**Total return:** \( E_T/E_0 - 1 \).

**Carver cost drag** (notebook-verified): \(\mathrm{drag}_{\mathrm{SR}} = \mathrm{turnover}_{\mathrm{RT/year}} \times \mathrm{cost}_{\mathrm{SR}}\). Warning if drag \(> 0.08\) SR/year.

**`summarize` returns:** `total_return, sharpe, max_drawdown, max_drawdown_duration_days, calmar`. Tearsheet extra keys include `max_drawdown_duration_days` (performance dict) vs tearsheet label `MDD duration`.

### 6.3 Metrics that are not implemented as named analytics

| Metric | Status |
| --- | --- |
| Sortino (downside deviation) | **[MISSING IN CODEBASE]** |
| Omega, information ratio, Treynor, Jensen \(\alpha\) | **[MISSING IN CODEBASE]** |
| VaR / CVaR / expected shortfall as portfolio analytics | **[MISSING IN CODEBASE]** (spectral PCA risk is a different object) |
| CAGR distinct from Calmar’s \(R_{\mathrm{ann}}\) | Only the Calmar numerator |
| Hit rate / profit factor of round-trips as `summarize` fields | Trades list exists; not aggregated in `summarize` |
| Intra-horizon Ulcer, Pain, Burke | **[MISSING IN CODEBASE]** |

Rolling Sharpe in `visualization/static.py` uses **simple** `pct_change` × \(\sqrt{252}\) (`ddof=1`) — a third Sharpe convention vs log `annualized_sharpe` vs fast-track simple Sharpe.

### 6.4 Deflated / probabilistic Sharpe (`analytics/dsr.py`)

*Not notebook-covered; Bailey & López de Prado 2014.*

Euler–Mascheroni \(\gamma \approx 0.5772156649\).

\[
\mathbb{E}[\max \mathrm{SR}_N]
= \sqrt{V}\left[(1-\gamma)\Phi^{-1}\!\left(1-\tfrac{1}{N}\right)
+ \gamma\,\Phi^{-1}\!\left(1-\tfrac{1}{Ne}\right)\right]
\]

(\(N\le 1\) or \(V\le 0\) → 0).

\[
\mathrm{PSR}
= \Phi\!\left(
\frac{(\widehat{\mathrm{SR}}-\mathrm{SR}_b)\sqrt{T-1}}
{\sqrt{\max\bigl(1 - \gamma_3\widehat{\mathrm{SR}} + \tfrac{\gamma_4-1}{4}\widehat{\mathrm{SR}}^2,\ 10^{-12}\bigr)}}
\right)
\]

\(\gamma_4\) is **Pearson** kurtosis (normal = 3), not Fisher excess. `annualized=True` divides both Sharpes by \(\sqrt{P}\) and trial variance by \(P\) before the formula — mixing annualized SR with bar-count \(T\) is documented as a massive inflation bug; the flag exists to prevent it.

\(\mathrm{DSR} = \mathrm{PSR}\) with \(\mathrm{SR}_b = \mathbb{E}[\max\mathrm{SR}_N]\). Registry-driven `dsr_from_registry` uses `n_trials()` and `sharpe_variance()` (`np.var(..., ddof=1)`). **DSR is only as honest as trial logging**; omitting losers understates \(N\) and typically \(V\).

### 6.5 Spectral risk (`portfolio/risk.py`)

Ledoit–Wolf \(\hat\Sigma\) on a returns DataFrame. Eigendecomposition `np.linalg.eigh`, descending eigenvalues \(\Lambda_{nn}\), eigenvectors \(v_n\).

\[
\beta_n = w^\top v_n,\quad
\sigma^2 = w^\top \hat\Sigma w,\quad
R_n = \frac{\beta_n^2 \Lambda_{nn}}{\sigma^2}.
\]

Equal weights if `weights is None`. **Analysis-only** — not a live risk overlay.

### 6.6 Monte Carlo statistics

**Masters MCPT p-value** (`montecarlo/permutation.py`), notebook-verified counting:

\[
p = \frac{1 + \#\{\mathrm{perm}_j \ge \mathrm{orig}\}}{n_{\mathrm{reps}}},
\quad n_{\mathrm{reps}} = 1 + \#\text{permutations}.
\]

Gate `p < 0.05`. Protocol I: identical permutation indices on all assets’ log-differences (keeps contemporaneous correlation). Protocol II: shuffle intra-bar \((\log H/O,\log L/O,\log C/O)\) **jointly** and inter-bar gaps \(\log(O_t/C_{t-1})\) independently; reconstruct with \( H=\exp(\max(r_H,r_C,0))O \) etc. so OHLC inequalities hold. *Protocol II reconstruction: not notebook-covered.*

Masters partition: \(\mathrm{Bias}=R_{\mathrm{perm}}-B_{\mathrm{perm}}\), \(R_{\mathrm{unbiased}}=R_{\mathrm{orig}}-\mathrm{Bias}\), \(\mathrm{Skill}=R_{\mathrm{unbiased}}-B_{\mathrm{orig}}\), \(\mathrm{Trend}=B_{\mathrm{orig}}\).

**Drawdown double bootstrap** (`montecarlo/drawdown.py`): wealth \(= \prod(1+r)\) on **simple** returns; DD = \(\max(\mathrm{HWM}-E)/\mathrm{HWM}\). Inner quantile `dd_conf=0.95`, outer `bound_conf=0.70`. *Quantile indices not notebook-covered.* Single-loop OOS resample is documented as anti-conservative.

**Ehlers parametric paths:** win if \(u\le p\), payoff \(=|\overline{L}|\cdot\)(`profit_factor` as **avg-win/avg-loss**, not gross PF); equity is **arithmetic** \( e_0 + \sum\mathrm{PnL} \) (not compounded). Empirical hat resample **does** compound \(\prod(1+r)\). Autocorrelation gate: Wald–Wolfowitz runs on \(\mathrm{sign}(x-\mathrm{median})\) and Ljung–Box; if either p < \(\alpha\), IID shuffle is invalid (Kaufman trap) → block bootstrap or OU paths.

**OU synthetic:** \( dP_t = \theta(\mu-P_{t-1})dt + \sigma dW_t \); OLS \( \Delta P = a + b P_{t-1} \); \(\theta=-b/dt\), \(\mu=-a/b\), \(\sigma=\mathrm{std}(\varepsilon)/\sqrt{dt}\); if \(b\ge 0\), \(\theta=0\) (RW around sample mean). Euler–Maruyama.

**GBM fixtures** (`utils/synthetic.py`): \( r_t \sim \mathcal{N}(\mu/252,\ \sigma/\sqrt{252}) \), \( C = s_0\exp(\sum r) \); OHLC from Gaussian open noise and a high/low band. Seeded `numpy.random.Generator`.

---

## 7. Known Limitations & Optimization Vectors

### 7.1 Implicit assumptions vs live trading

1. **Bar = four prints.** No tick path, no auction, no quote flicker. Limit/stop “touch” is high/low existence.
2. **Open is tradable at the decision** for delay-0, subject only to spread/Kaufman/Kyle and optional participation caps — **[MISSING: latency]**.
3. **Close MOC is a single print** at \(C_t\), all-or-nothing vs a volume cap — not a close auction with imbalance.
4. **Shares are real-valued**; no lot/tick/notional rounding; no borrow inventory.
5. **One currency, one cash account.** Negative cash is a silent loan at 0%.
6. **Adjusted Yahoo prices ≠ total-return ledger.** Dividends are in the price path, not in cash.
7. **Universe = whatever frames you passed.** Survivorship is a research-governance WARN, not a data product.
8. **Annualization = 252 bars/year** in Sharpe, Calmar, DSR de-annualization, rolling vol, fast-track Sharpe, GBM scaling. Hourly crypto (~8760) or 24/5 FX will inflate SR by \(\sqrt{8760/252}\approx 5.9\) if left at default.
9. **Three Sharpe conventions** (log event analytics, simple fast-track, simple rolling viz) can disagree on the same equity curve.
10. **Kelly / optimal-\(f\) / vol-parity / spectral risk** are library functions; live size is percent-equity or fractional-risk or fixed units.
11. **Pairs are not residual-weighted.**
12. **Imbalance-bar threshold** ≠ textbook \(\mathbb{E}_0[T]|2P-1|\).
13. **Triple barrier** is close-path, not high/low.
14. **Strategy stops ≠ resting STOP orders** (extra bar of gap risk).
15. **Fast-track ⊄ event engine** (delay-1 market-at-open subset). Path-dependent strategies cannot use MCPT fast-track (`NotImplementedError` on `vectorized_signals`).
16. **PBO/DSR** require honest trial registration; the registry cannot see experiments that were never logged.
17. **Idle-cash 365 vs metric 252** mixes day-count conventions inside the same book.

### 7.2 Mathematical / implementation defects (code-level)

| Issue | Location | Effect |
| --- | --- | --- |
| `n_truncate` NameError on empty overlap | `validation/truncation.py` | Truncation diagnostic crashes instead of returning FAIL. |
| Fast-track Sharpe vs log Sharpe | `montecarlo/fast_track.py` vs `analytics/performance.py` | MCPT “Sharpe” ranking need not match tearsheet Sharpe. |
| Fixed commission per fill slice | `CostModel.commission` + partial fills | Splitting an order multiplies \(c_{\mathrm{fix}}\). |
| `phi[N,k]` integer truncation | `CombinatorialPurgedKFold.n_paths` | `int((k/N)*n_splits)` can undershoot the combinatorial count. |
| Embargo via median \(\Delta t\) | `cpcv._as_offset` | Irregular calendars (halts, 24/7) get a smeared embargo. |
| Daily breaker `.date()` | `DailyDrawdownBreaker` | UTC day ≠ exchange session; overnight futures/crypto mis-roll. |
| Dollar/imbalance leftover bar | `data/bars.py` | Last incomplete information bar is always emitted. |

### 7.3 Performance bottlenecks (GIL / I/O / architecture)

- **Event loop:** pure Python per-bar, per-event object construction (`dataclass` events, pandas Series bars, queue put/get). Suitable for daily/hourly research; not for tick-by-tick multi-year universes.
- **GIL:** no parallel symbol processing inside a run. Parallelism is **across trials** (external workers + JSONL → SQLite), not inside `BacktestEngine`.
- **pandas in the hot path:** `get_latest_bars` is `df.loc[mask].tail(n)` every signal. Rolling indicators recompute on the trailing window each bar (Donchian/ADX/SMA).
- **Imbalance bars:** Python `for t in range(len(ticks))` with a pandas EWMA on growing lists each tick — \(O(T^2)\) risk on long tick files.
- **Memory:** full OHLCV resident; outer-join index duplicated in `_position_of` dict.
- **MCPT:** event-loop 10k re-runs are documented as intractable; fast-track is the intended path but only for strategies with a vectorized twin and delay-1 market semantics.
- **sklearn LinearRegression per bar** in pairs trading (252-point OLS every close) dominates that strategy’s runtime.

### 7.4 Anti-patterns relative to institutional stacks

- Research math (Vince, Kelly, Kyle, DSR) is **adjacent** to the book, not **coupled** (easy to compute \(f^*\) and then size with 50% equity).
- Execution richness (STOP/LIMIT/MOC/participation) exceeds what the portfolio **emits** (mostly MARKET ± LIMIT from `signal.limit_price`).
- Data quality is a **checklist**, not a point-in-time corporate-action engine (CRSP/Compustat-class).
- No research-to-live broker adapter despite `ExecutionHandler` ABC comments; the ABC is filled only by the simulator.
- Visualization indicators can be computed on full frames (safe only post-run; the package does not technically prevent a strategy from importing them on `source_ohlcv`).

### 7.5 Highest-value literature cross-check list

When this blueprint is lined up against de Prado (AFML), Bailey DSR/PBO, Vince, Carver, Chan, Kaufman, Masters:

1. Confirm **external \(c_t\)** on the ETF trick (implemented correctly vs a common “embed in \(K_t\)” error).
2. Confirm **label-overlap purge + embargo**, not a symmetric lookback-only guard (implemented).
3. Confirm **DSR moments and annualization contract** (implemented, but unused if the registry is empty).
4. Challenge **close-only triple barrier**, **imbalance EWMA estimator**, **252-bar Calmar**, **non-β pairs sizing**, **strategy-level stops vs gap-through STOP orders**, **YFinance auto-adjust without cash dividends**, **survivorship as a WARN flag**.
5. Challenge **delay-0 = same-bar open with no latency** as a live-tradable assumption.
6. Challenge **fast-track Sharpe** as a drop-in for MCPT if the paper’s SR is the tearsheet log SR.

---

*End of blueprint. All formulas and absences above are taken from the `quantester/` package as of the `main` revision this document was generated against. Items marked **[MISSING IN CODEBASE]** are not inferred from comments or design notes unless the comment itself states the omission.*
