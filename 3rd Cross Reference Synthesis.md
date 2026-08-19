# 3rd Cross-Reference Synthesis

Deduplicated findings from `3rd Cross Reference.md` (per-book audits of `SYSTEM_ARCHITECTURE_QUANTITATIVE_METHODOLOGY_BLUEPRINT.md`). Identical items are merged; similar items are one finding with a source list. Conflicts are isolated in §4 — do not treat those rows as settled defects.

Severity in the source dump is not authoritative: several books label a missing *specialty method* as 🔴 even when the engine is a general backtester. This synthesis keeps the source severity, then notes when the claim is a toolkit gap rather than a mathematical error in existing code.

---

## 1. Critical flaws (merged)

### 1.1 Three Sharpe conventions (log vs simple) — MCPT ranking mismatch

**Claim.** Event-loop `annualized_sharpe` uses log returns \(\ell_t=\log(E_t/E_{t-1})\); `fast_backtest.sharpe` and `visualization/static.py` rolling Sharpe use simple `pct_change` \(\times\sqrt{252}\). Parity is not guaranteed even when equity paths match, so MCPT (fast-track) can rank a different statistic than the tearsheet.

**Sources.** AFML ch. 11/14; Masters *Assessing…* pp. 32, 149–150; Brunton & Kutz ch. 1.5; Chan *Quantitative Trading*; Masters *Statistically Sound ML…* (Passage 269); Carver *Systematic Trading*; Masters *Testing and Tuning…* ch. 1; Kaufman *Trading Systems and Methods*.

**Action (uncontested part).** Unify `analytics/performance.py`, `montecarlo/fast_track.py`, and `visualization/static.py` on **one** return representation.

**Do not implement yet without resolving §4.1** (books disagree on *which* representation).

---

### 1.2 Hardcoded `TRADING_DAYS = 252`

**Claim.** Sharpe, Calmar (\(n_E/252\) as years), DSR de-annualization, rolling vol, GBM scaling, and fast-track Sharpe all assume 252 bars/year. Hourly crypto \(\sqrt{8760/252}\approx 5.9\) inflation. Cash yield uses 365-day simple interest in the same book.

**Sources.** AFML ch. 2/18; Ruggiero *Cybernetic Trading Strategies*; Ehlers *Cycle Analytics*; Hilpisch *Python for Algorithmic Trading*; Chan *Quantitative Trading* ch. 3 (\(N_T=252\times 6.5=1638\) hourly NYSE); Masters *SSML*; Carver *Systematic Trading*; Penfold *Universal Tactics*; Peterson *Trading on Sentiment*; Kaufman *TSM*; Vince *Mathematics of Money Management*.

**Action (uncontested part).** Stop silently applying 252 to non-daily indexes.

**Do not implement yet without resolving §4.2** (Carver 256 vs Chan actual frequency vs dynamic median \(\Delta t\)).

---

### 1.3 Strategy stops are delay-1 EXIT, not resting `STOP_ORDER`

**Claim.** Simulator implements `STOP_ORDER` (gap-through at \(\max(P_{\mathrm{stop}},O_t)\)). `PortfolioManager.update_from_signal` never emits it. Donchian / tranche observe touch at close and emit `EXIT` with `delay=1` (next open). Extra bar of gap risk; intra-bar stop activation never happens.

**Sources.** AFML ch. 3.2; Ehlers *Cybernetic Analysis*; Ruggiero; Ehlers *Cycle Analytics* ch. 17; Chan *Quantitative Trading*; Masters *SSML*; Carver *Systematic Trading*; Vince *Mathematics of Money Management* ch. 1; Penfold *Universal Tactics*; Harris *Trading and Exchanges*.

**Action.** Wire strategies that *intend* intra-bar stops through resting `STOP_ORDER` on the execution ledger.

**Caveat — §4.3.** Delay-1 *entries* after the close are separately **aligned** with Ehlers/Penfold/Chan. Do not collapse “signal delay-1” with “stop delay-1”. Clenow (§4.10) requires *no* stop at all.

---

### 1.4 Survivorship, delistings, and corporate-action ledger

**Claim.** No PIT constituent file, no delist/halt residual, no `CorporateActionEvent`. Audit flags are WARN-only. `YFinanceDataHandler(auto_adjust=True)` bakes splits/dividends into OHLC without booking cash — understated cash or double-count if dividends are also modeled. Loading today’s survivors is classic survivorship bias (Chan Example 3.3: −42% → +388%).

**Sources.** AFML ch. 11.1; Chan *Quantitative Trading* ch. 2–3; Masters *SSML*; Penfold *Universal Tactics*; Kaufman *TSM*; Clenow *Stocks on the Move*; Peterson *Trading on Sentiment* (CA as 🟡 in that audit, 🔴 when bundled with PIT).

**Action.** PIT universe ingestion; split multipliers and cash dividends on the ledger; do not treat `auto_adjust=True` as a total-return *and* cash book.

---

### 1.5 Chan truncation diagnostic `NameError`

**Claim.** `validation/truncation.py` empty-overlap branch uses undefined `n_truncate` instead of `n_truncated` → crash instead of FAIL.

**Sources.** AFML (bundled with embargo); Brunton & Kutz ch. 4.6; Chan ch. 3.3; Masters *SSML*; Carver; Masters *Testing and Tuning*; Vince *MoMM*; Peterson; Kaufman; Clenow.

**Action.** Rename to `n_truncated`. Unambiguous code defect; no literature conflict.

---

### 1.6 Volume / dollar imbalance bars ≠ AFML estimator

**Claim.** AFML ch. 2: threshold from EWMA of \(P[b_t=1]\) and conditional sizes, \(|2v^+-E_0[v_t]|\). Code: concatenate tick flows across completed bars, take \(|\mathrm{EWMA}|\times\mathrm{EWMA}(\mathrm{length})\). Warmup is a constant; leftover ticks always emit a bar.

**Sources.** AFML ch. 2 only.

**Action.** Track tick-level \(E_0[T]\), \(P[b=1]\), \(v^\pm\) as specified.

---

### 1.7 Triple-barrier path is close-only

**Claim.** AFML ch. 3 requires first touch on the price *path*. Implementation uses `close.iloc[i0+1:end]`; intra-bar H/L touches are invisible (under-detect TP/SL).

**Sources.** AFML ch. 3 only.

**Action.** Label on high/low (and vertical barrier), not close-only.

---

### 1.8 Masters skill partition uses \(B_{\mathrm{orig}}\) not \(B_{\mathrm{perm}}\)

**Claim.** Code: \(\mathrm{Bias}=R_{\mathrm{perm}}-B_{\mathrm{perm}}\), \(\mathrm{Skill}=R_{\mathrm{unbiased}}-B_{\mathrm{orig}}\). Masters C++ (p. 276): skill should subtract mean *permuted* inherent bias \(B_{\mathrm{perm}}\), not sample-specific \(B_{\mathrm{orig}}\).

**Sources.** Masters *Assessing and Improving Prediction and Classification* ch. 5 / p. 276.

**Action.** Confirm against the notebook-verified partition already in `trend_bias_skill`; if the C++ form differs, document which formula is intended before changing.

**Caveat.** Repo docstring claims notebook-verified Masters partition. Treat as a **formula conflict with a secondary source**, not a proven bug, until the notebook page is checked (§4.8).

---

### 1.9 CPCV embargo not tied to lookback / look-ahead; median \(\Delta t\) smear

**Claim.** Embargo \(h=\lfloor\mathrm{pct\_embargo}\cdot T\rfloor\) then \(\times\) median bar spacing. Too small on short samples (leak); too large on long samples; irregular calendars smear the mask. Masters wants \(\mathrm{shrink}=\min(\mathrm{lookback},\mathrm{look\text{-}ahead})-1\) in *bars*.

**Sources.** AFML ch. 7.4.2 (🟡 in one audit); Masters *Assessing…* ch. 1 (🔴); Masters *Testing and Tuning* (🔴).

**Conflict.** Masters *SSML* marks the same CPCV overlap purge as 🟢 aligned (§4.4).

---

### 1.10 `as_daily_reindex(..., method='bfill')` look-ahead

**Claim.** Default `ffill` is causal; `bfill` is offered and would leak future macro prints into past bars.

**Sources.** Brunton & Kutz ch. 7.1; Hilpisch; Masters *SSML*.

**Action.** Disallow `bfill` on trading-feature joins (or hard-error). Unambiguous if used as a signal input; the API currently allows it.

---

### 1.11 GBM fixture missing Itô drift

**Claim.** Hilpisch Euler GBM: \(S_T=S_0\exp((r-0.5\sigma^2)T+\sigma z\sqrt{T})\). Code: \(r_t\sim\mathcal{N}(\mu/252,\sigma/\sqrt{252})\), \(C=s_0\exp(\sum r)\) → \(E[S]\) grows like \(\exp((\mu+0.5\sigma^2)T)\).

**Sources.** Hilpisch *Python for Algorithmic Trading*.

**Action.** Use \(\mathcal{N}((\mu-0.5\sigma^2)/252,\sigma/\sqrt{252})\) if the fixture is meant to be a martingale at rate \(\mu\). Does not affect the event engine.

---

### 1.12 Delay-0 fill at the decision open (no latency)

**Claim.** Delay-0 may see \(T_k\) open and fill at that same print. `earliest_fill_time` is bar-time. No ms latency, auction, or queue.

**Sources.** Harris *Trading and Exchanges* (🔴); Peterson *Trading on Sentiment* (🔴); Kaufman *TSM* (🟡 microstructure).

**Conflict.** Quantester’s temporal firewall *intentionally* supports delay-0 under an intra-bar visibility guard (overnight MR / open execution). Harris/Peterson want delay-0 **removed** (§4.6).

---

### 1.13 Pairs legs independently sized (`strength=1.0`)

**Claim.** OLS hedge \(\beta_t\) is computed but not mapped to quantities. `PercentEquitySizer` does \(\mathrm{pct}\cdot E/P_i\) per leg → not dollar-neutral / cointegrating residual.

**Sources.** Kaufman *TSM* (🔴); AFML / Chan (🟡 decoupled sizing).

**Action.** `HedgeRatioSizer`: \(q_X = -\beta_t q_Y\) (and optional ATR/vol balance).

---

### 1.14 Vince `gap_stress=1.5` on \(W\)

**Claim.** \(W=\mathrm{gap\_stress}\cdot\min_i\mathrm{Trade}_i\) inflates BiggestLoss, silently de-levers left of \(f^*\) (~33% size cut at 1.5).

**Sources.** Vince *Leverage Space* ch. 1–2; Vince *Mathematics of Money Management* ch. 1.

**Conflict.** Repo invariant (Cross-Ref-2 §4.2): unconstrained \(f^*\) on *nominal* stop losses is ruinous because fills gap through. Gap-stress is deliberate (§4.5).

---

### 1.15 Vince risk = covariance / vol-parity, not joint scenario probabilities

**Claim.** Live/offline risk uses Ledoit–Wolf \(\hat\Sigma\) and \(w_i\propto 1/\sigma_i\). Vince LSM discards correlation; sizes from joint scenario tables.

**Sources.** Vince *Leverage Space* ch. 4.

**Conflict.** Ledoit–Wolf + spectral risk is an AFML/Brunton feature, not an accident. Rewriting it away is a product choice, not a universal bug (§4.7).

---

### 1.16 Procyclical sizers on mark-to-market \(E\)

**Claim.** `PercentEquitySizer` / `FractionalRiskSizer` scale on cash + unrealized MTM. Trend reversal unwinds inflated size.

**Sources.** Penfold *Universal Tactics*.

---

### 1.17 `FractionalRiskSizer` requires `stop_distance`

**Claim.** Clenow momentum sizes \(E\times 0.001/\mathrm{ATR}_{20}\) with **no** stop. Current sizer cannot represent that without a fake stop (which would then fire).

**Sources.** Clenow *Stocks on the Move*.

**Conflict.** Harris/AFML demand resting stops; Clenow forbids them (§4.10).

---

### 1.18 Nasdaq/dealer volume double-count in impact caps

**Claim.** Raw `volume` feeds Kyle \(\lambda\) and `max_fill_quantity`. Harris: quote-driven prints can double-count dealer legs.

**Sources.** Harris *Trading and Exchanges*.

---

### 1.19 Daily DD breaker rolls on UTC `.date()`

**Claim.** Day-open equity = last valuation of previous calendar date. 24/7 crypto / FX “mis-rolls” at UTC midnight.

**Sources.** *Detecting Regime Change in Computational Finance*.

---

### 1.20 Directional-Change (DC) event clock absent

**Claim.** No downturn/upturn confirmation, overshoot, \(TMV\), \(T\), or \(T\)-\(TMV\) space. Engine is chronological (plus dollar/imbalance bars).

**Sources.** *Detecting Regime Change…* only.

**Note.** This is a missing *research framework*, not a defect in the bar engine that exists. Severity 🔴 in that audit is specialty-scope.

---

### 1.21 Fisher Transform / non-Gaussian Bollinger

**Claim.** Raw \(\sigma\) bands on non-Gaussian prices are “just plain wrong”; Ehlers requires Fisher \(y=\frac12\ln\frac{1+x}{1-x}\).

**Sources.** Ehlers *Cybernetic Analysis*.

**Note.** Specialty indicator demand; the backtester is not required to Fisher-transform all bands. Keep as 🔴 only if the product claims Ehlers-faithful oscillators.

---

### 1.22 CPCV `n_paths` integer truncation

**Claim.** `int((k/N)*n_splits)` can undershoot \(\phi[N,k]=(k/N)C(N,N-k)\).

**Sources.** Brunton & Kutz (🔴); Masters *Testing and Tuning* (🟡 with walk-forward).

---

### 1.23 Pandas hot-path / \(O(TN)\) rolling recomputation

**Claim.** `get_latest_bars` → `df.loc[mask].tail(n)` every signal; indicators re-roll the window; imbalance bars EWMA growing lists \(\sim O(T^2)\).

**Sources.** Hilpisch (🔴 vectorize); Ehlers *Cybernetic* (🟡 \(O(1)\) SMA); Brunton (🟡).

**Conflict.** Severity only (§4.9). Not a look-ahead bug.

---

### 1.24 Decoupled Vince/Kelly as 🔴 (Ruggiero)

Merged with §2.1. Ruggiero rates it critical; AFML/Chan/Vince *MoMM* rate it a gap. See §4.7.

---

## 2. Architectural gaps (merged)

Research math or specialty modules **not wired** (or not present). Not the same as a wrong formula in existing code.

| Gap | What is missing | Sources |
| --- | --- | --- |
| Live sizers ≠ research allocators | Kelly, Vince \(f^*\), vol-parity, Ledoit–Wolf/HRP, Kakushadze \(E_{\mathrm{eff}}\) are library-only; live default `PercentEquitySizer(0.5)` / one-call `0.9` | AFML ch. 10/16; Chan ch. 6; Ruggiero ch. 20; Vince *MoMM*; Carver (vol target) |
| Carver vol targeting + FDM + 10% inertia | Cash vol target, forecast diversification \(1/\sqrt{WHW^\top}\) cap 2.5, rebalance only if \(\Delta q>10\%\) | Carver *Systematic Trading* ch. 8, 11 |
| Stationary / tapered block bootstrap + optimal \(b\) | Audits say missing; **code already has** Politis–Romano `bootstrap_ohlcv` and a block-bootstrap path after the autocorrelation gate — see §4.11 | Masters *Assessing* ch. 3; *SSML* |
| Walk-forward / NTEST–EXTRA | CPCV exists; no sliding Dev/Test/OOS or `NTEST=1`, `EXTRA=LOOKAHEAD-1` | Ruggiero; Masters *Testing and Tuning* |
| Profit-criterion fit vs OLS/MSE | Pairs uses `LinearRegression` MSE; no Ulcer/Martin/profit-factor training | Masters *SSML* |
| Cross-section fractile / rotation | No rank-across-names long/short quantiles; no Buzz filter | Masters *SSML*; Peterson |
| Feature selection (MI / PLD / Fleuret) | No entropy / mRMR screen | Masters *Assessing* ch. 9 |
| Ensembles / gating | No Borda, Fuzzy Integral, GRNN gate | Masters *Assessing* ch. 6–8 |
| Ehlers cycle stack | Hilbert I/Q, dominant cycle, Super Smoother, ITrend, roofing filter, AGC, autocorrelation periodogram | *Cybernetic Analysis*; *Cycle Analytics* |
| KAMA | Efficiency-ratio adaptive MA | Kaufman ch. 17 |
| Clenow rank / clocks | \(\mathrm{slope}\times R^2\) on \(\log C\); weekly vs bi-weekly rebalance | Clenow |
| DC regime stack | NBC B-Simple/B-Strict; JC1/JC2/CT1; HMM on DC return | *Detecting Regime Change* ch. 3–6 |
| Vince LSM extras | Joint \(HPR(f_1\ldots f_N)\); \(RD(b,q)\), \(RR(b,q)\); daily HPR vs \(f\$\) | *Leverage Space* ch. 4–5; *MoMM* |
| Control / identification | Gavish–Donoho hard threshold; LQR/LQG+Kalman; closed-loop excitation (DMDc/SINDYc) | Brunton & Kutz ch. 1.7, 8–10 |
| Ruggiero extras | Equity-curve feedback MA; predictive correlation; C4.5 / Rough Sets | Ruggiero ch. 10, 11, 19 |
| Masters extras | Monotonic tail-only cleaning; log profit-factor bootstrap | *Testing and Tuning* ch. 2 |
| Microstructure / TCA | Latency, tick/lot, maker-taker, iceberg, size precedence, effective/realized spread, Perold IS, NBBO dual-trading | Harris; Kaufman; Hilpisch (sockets) |
| Storage / live | HDF5/TsTables; chunked bars; broker sockets | Hilpisch |
| Headline metrics | Sortino; simple alpha vs benchmark; round-trip PF in `summarize` | Kaufman; Hilpisch; Masters *Testing* |
| Donchian over-stack | SMA200 + Donchian20 + ADX>25 vs Penfold “pure channel” | Penfold |
| Sentiment field parser | TRMI unipolar vs bipolar / negation | Peterson |
| `source_ohlcv` not sealed | Strategies *can* bypass the firewall if they call it | Kaufman (suggested permission error) |

---

## 3. Aligned (merged)

These survived many independent books. Treat as keep / do not “fix.”

| Alignment | Evidence | Sources |
| --- | --- | --- |
| ETF trick: \(c_t\) **external** to \(K_t\) | Separate column; embedding fabricates short-spread profits | AFML ch. 2.4.1; Chan; Carver; Kaufman; Clenow; Peterson; Brunton; DC book; Vince *MoMM* |
| MCPT Protocol I: **same shuffle index** across assets | Preserves contemporaneous correlation | AFML ch. 13; Ehlers *Cybernetic*; Ruggiero; Brunton; Kaufman; Peterson |
| Masters p-value: \(p=(1+\#\{\mathrm{perm}\ge\mathrm{orig}\})/(1+n_{\mathrm{perm}})\) | Conservative count | Masters *Assessing* pp. 266–267; *SSML*; *Testing and Tuning*; Carver; Vince *MoMM* |
| Delay-1 **entries**: close \(T\) → open \(T+1\); open-phase H/L/C redacted | Live-replicable, anti look-ahead | Ehlers *Cycle Analytics*; Penfold; Chan ch. 3.3; Clenow |
| Dual event + vectorized fast-track with **documented** subset contract | Vectorized cannot do stops/limits/MOC/delay-0 | Hilpisch |
| Idle cash yield on **positive** cash, Kaufman half T-bill / Carver RF | \(\Delta\mathrm{cash}=\mathrm{cash}\cdot r\cdot\eta\cdot\Delta\mathrm{days}/365\) | Carver; Kaufman |
| Limit gap: buy \(R=\min(O,P_{\mathrm{lim}})\); sell \(\max(O,P_{\mathrm{lim}})\) | Price improvement through the limit | Harris |
| MOC all-or-nothing vs participation cap | No silent residual | Harris |
| Drawdown **double** bootstrap (single-loop anti-conservative) | `montecarlo/drawdown.py` | Masters *Testing and Tuning* |
| CSCV / PBO | `validation/pbo.py`, gate 0.10 | Masters *Testing and Tuning*; AFML/Bailey |
| Vince **single-name** HPR/TWR/\(f^*\) algebra | Matches ch. 1 except `gap_stress` | Vince *Leverage Space* |
| Ehlers parametric MC (win rate, avg win/loss, arithmetic equity) | Distinct from compounding hat resample | Ehlers *Cycle Analytics* |

---

## 4. Conflicts — do not “fix” until resolved

### 4.1 Unify Sharpe on **simple** or **log**?

| Camp | Prescription | Sources |
| --- | --- | --- |
| Simple / TWRR / cost drag | Tearsheet + MCPT on simple \(r_t\); Carver drag is linear in simple SR | AFML (audit); *SSML* Passage 269; Carver *Systematic Trading* |
| Log / symmetry / IID resampling | Log differences for pooling and MCPT | Masters *Assessing* pp. 32, 149–150; *Testing and Tuning* ch. 1; Brunton (audit prefers log for resamplers) |

**Implication.** Implementing AFML’s “preferably simple” **contradicts** Masters’ “preferably log.” Hilpisch **aligns** with *documenting* the split; other books call the same split 🔴.

**Future check.** Pick a canonical SR for (a) tearsheet, (b) MCPT objective, (c) DSR registry. Convert explicitly; do not mix.

---

### 4.2 Annualization: 252 vs 256 vs measured frequency

| Camp | Rule | Sources |
| --- | --- | --- |
| Carver | **256** business days, \(\sqrt{256}=16\) exactly | *Systematic Trading* |
| Chan / Kaufman / AFML / Ehlers | \(N_T =\) actual periods per year (hourly NYSE 1638, etc.) | Chan ch. 3; others in §1.2 |
| Dynamic median \(\Delta t\) | Infer from index | Repeated audit action text |

Median-\(\Delta t\) on US daily equities ≈ 252, **not** Carver’s 256. Crypto 24/7 is neither.

**Future check.** Parameterize `periods_per_year` (and day-count for cash yield) per instrument calendar; default is a product decision, not a theorem.

---

### 4.3 Delay-1 is both a critical flaw and a gold-standard alignment

| Object | Rating | Sources |
| --- | --- | --- |
| **Entries** after close, fill next open | 🟢 | Ehlers *Cycle Analytics*; Penfold; Chan look-ahead; Clenow |
| **Stops** observed at close, fill next open | 🔴 extra gap bar | Harris; AFML 3.2; Chan stops; Carver; Vince *MoMM* |

Audits that say “delay-1 violates Ehlers rapid execution” are talking about *stops*, while Ehlers *Cycle Analytics* explicitly wants signals **after the close**. Split the engine: delay-1 market entries vs optional resting stops.

---

### 4.4 CPCV purge: aligned vs leaking

| View | Sources |
| --- | --- |
| 🟢 Label-interval overlap purge matches Masters/de Prado | *SSML* |
| 🔴 Embargo length is `% of T` + median time, not lookback/lookahead bars | Masters *Assessing* ch. 1; *Testing and Tuning* |

Both can be true: overlap *geometry* is right; embargo *length policy* is not AFML/Masters-tight.

---

### 4.5 Vince `gap_stress` vs Quantester stop-gap invariant

| View | Claim |
| --- | --- |
| Vince | \(W=\) raw BiggestLoss; 1.5× is silent de-lever |
| Quantester / Cross-Ref-2 §4.2 | Perfect stops unbind \(f^*\); gap-through fills make historical min loss too small |

**Future check.** Keep gap-stress as an explicit conservative \(f\) input (documented), or compute \(W\) from **realized** gap-through PnL, not a magic 1.5.

---

### 4.6 Delay-0: feature vs unphysical leak

| View | Sources |
| --- | --- |
| Keep delay-0 + open-only visibility | Quantester firewall; AFML-style overnight MR at the open |
| Delete delay-0; minimum 1-bar or ms latency | Harris; Peterson NFP/TRMI timing |

Not a coding error either way; it is a **simulation fidelity** switch.

---

### 4.7 Covariance / Kelly / Vince: bug vs adjacent library

| View | Sources |
| --- | --- |
| 🔴 Must discard \(\Sigma\), use joint scenarios; couple \(f^*\) to every fill | Vince *Leverage Space*; Ruggiero |
| 🟡 Library is fine; live sizer is a separate policy | AFML; Chan; Vince *MoMM* (gap) |
| Ledoit–Wolf spectral risk is **correct** for ill-conditioned \(\Sigma\) | AFML / Brunton (repo design) |

Do not rip out Ledoit–Wolf because Vince LSM rejects MPT.

---

### 4.8 Masters skill formula vs notebook-verified `trend_bias_skill`

Blueprint/code: \(\mathrm{Skill}=R_{\mathrm{unbiased}}-B_{\mathrm{orig}}\) with \(\mathrm{Trend}=B_{\mathrm{orig}}\).  
*Assessing…* p. 276 audit: \(\mathrm{Skill}=R_{\mathrm{unbiased}}-B_{\mathrm{perm}}\).

Repo marks the implemented partition **notebook-verified**. Re-read the notebook/C++ before changing.

---

### 4.9 Hilpisch dual-track: documenting divergence is 🟢 *and* 🔴

Same fact (fast-track ≠ full engine Sharpe):

- Hilpisch: 🟢 honest dual architecture  
- AFML/Chan/Carver/Masters: 🔴 metric divergence  

**Future check.** Keep two engines; add a **parity metric mode** (fast-track Sharpe calls `annualized_sharpe` on the same equity series).

---

### 4.10 Stops required vs stops forbidden

| Strategy family | Stop policy | Sources |
| --- | --- | --- |
| Trend / microstructure | Resting intra-bar stops | Harris, AFML, Penfold expectancy |
| Clenow momentum | **No** stop; ATR vol sizing | *Stocks on the Move* |

Engine should support both; `FractionalRiskSizer` must not be the only vol sizer.

---

### 4.11 “Stationary bootstrap missing” vs code

Audits (*Assessing…*, *SSML*) claim SB/TBB and Politis–White optimal \(b\) are absent.

**Code already has** `montecarlo/synthetic.py::bootstrap_ohlcv` (Politis–Romano stationary bootstrap, shape-preserving OHLC) and `adaptive_empirical_resample` (block length heuristic, not TBB / not Paparoditis–Politis optimal \(b\)).

**Future check.** Gap is **TBB + automatic \(b_{\mathrm{SB}}\)**, not “no stationary bootstrap.” Do not implement a second SB from a false-negative audit.

---

### 4.12 Adjusted prices vs unadjusted + cash dividends

Clenow/Peterson/Kaufman: total-return ranking vs cash booking. YFinance `auto_adjust=True` is a **documented** total-return-like path without cash events. Fix is policy (`auto_adjust=False` + CA file) not “Yahoo is wrong.”

---

## 5. Suggested implementation order (non-conflicting)

Safe without a literature ruling:

1. `n_truncate` → `n_truncated`  
2. Seal or warn on `source_ohlcv` from `calculate_signals`  
3. Disable or hard-fail `bfill` on trading features  
4. Hedge-ratio pairs sizer  
5. Resting `STOP_ORDER` **opt-in** for Donchian/tranche (leave delay-1 entries)  
6. Fast-track Sharpe = same function as tearsheet (pick representation after §4.1)  
7. `periods_per_year` argument (default left explicit)  
8. PIT / dividend events when a dataset exists  
9. Triple-barrier high/low; AFML imbalance EWMA  
10. Itô term in *synthetic* GBM only  

Park behind §4: log vs simple SR, 252 vs 256, remove delay-0, strip `gap_stress`, replace Ledoit–Wolf with Vince LSM, Fisher-on-all-indicators, DC/HMM/TRMI productization.

---

*Raw per-book write-ups remain in `3rd Cross Reference.md`. This file is the working merge.*
