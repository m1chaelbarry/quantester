# What bootstrap methods remain missing after the existing stationary bootstrap?

Type: research
Ticket: [What bootstrap methods remain missing after the existing stationary bootstrap?](../issues/04-stationary-bootstrap-gap.md)
Part of: [Literature remediation decision map](../map.md)

## Answer

The audit dump’s claim that the **Stationary Bootstrap is missing** is a false negative. Quantester already implements a Politis–Romano (1994) stationary bootstrap (geometric block lengths, circular wrap) in `quantester/montecarlo/synthetic.py` (`_stationary_bootstrap_indices` + `bootstrap_ohlcv`). Do **not** implement a second stationary bootstrap.

What remains, relative to the papers the audits invoke, is:

1. **Tapered Block Bootstrap (TBB)** — Paparoditis–Politis (2001); not present anywhere in `quantester/montecarlo/`.
2. **Automatic optimal expected block length \(b_{\mathrm{SB}}\)** — Politis–White (2004) as corrected by Patton–Politis–White (2009); not present. The existing `mean_block` is a caller-supplied heuristic (default 20 bars).
3. **Automatic optimal fixed block length \(b_{\mathrm{CB}} / b_{\mathrm{MB}} / b_{\mathrm{TBB}}\)** — same Politis–White pipeline for circular/moving blocks, and the Paparoditis–Politis \(n^{1/5}\) TBB selector; not present. The return-path block bootstrap uses a different, weaker heuristic (`suggest_block_length`).

Those remaining items are **inference-optimal block-size / taper machinery** for long-run variance of the sample mean, not a missing SB engine. The repo already has SB as a synthetic-path generator.

## Sources used (primary)

Code (this repo):

- `quantester/montecarlo/synthetic.py` — `_stationary_bootstrap_indices`, `bootstrap_ohlcv`
- `quantester/montecarlo/adaptive.py` — `suggest_block_length`, `adaptive_empirical_resample`
- `quantester/montecarlo/trade_resampling.py` — `_block_draws`, `empirical_resample`
- `quantester/montecarlo/diagnostics.py` — `autocorrelation_gate`
- `quantester/montecarlo/permutation.py` — MCPT shuffles (not a bootstrap)
- `quantester/montecarlo/drawdown.py` — nested DD bound (IID inner/outer draws)
- `tests/test_montecarlo.py` — SB contiguity / `mean_block=1` iid, adaptive routing
- Callers: `examples/tranche_pullback/run_parameter_study.py` (`MEAN_BLOCK = 20`), `examples/production_research/run.py` (`mean_block=20`)

Papers / official descriptions (not the audit dump):

- Politis, D. N. and Romano, J. P. (1994). “The Stationary Bootstrap.” *JASA* 89(428): 1303–1313. [doi:10.1080/01621459.1994.10476870](https://doi.org/10.1080/01621459.1994.10476870)
- Politis, D. N. and White, H. (2004). “Automatic Block-Length Selection for the Dependent Bootstrap.” *Econometric Reviews* 23(1): 53–70. [doi:10.1081/ETC-120028836](https://doi.org/10.1081/ETC-120028836). Author PDF: [public.econ.duke.edu/~ap172/Politis_White_2004.pdf](https://public.econ.duke.edu/~ap172/Politis_White_2004.pdf)
- Patton, A., Politis, D. N. and White, H. (2009). Correction to the 2004 paper. *Econometric Reviews* 28(4): 372–375. [doi:10.1080/07474930802459016](https://doi.org/10.1080/07474930802459016). Author PDF: [public.econ.duke.edu/~ap172/Patton_Politis_White_2009.pdf](https://public.econ.duke.edu/~ap172/Patton_Politis_White_2009.pdf)
- Paparoditis, E. and Politis, D. N. (2001). “Tapered Block Bootstrap.” *Biometrika* 88(4): 1105–1119. [doi:10.1093/biomet/88.4.1105](https://doi.org/10.1093/biomet/88.4.1105)
- Paparoditis, E. and Politis, D. N. (2002). “The tapered block bootstrap for general statistics from stationary sequences.” *Econometrics Journal* 5(1): 131–148. (TBB beyond the sample mean.)
- Politis, D. N. publications list (canonical bibliographic record): [mathweb.ucsd.edu/~politis/DPpublication.html](https://mathweb.ucsd.edu/~politis/DPpublication.html)

Not treated as primary: `3rd Cross Reference.md` audit dump; Masters *Assessing* C++ listing page numbers (161 / 169 / 171) — the book is not in this repo. Those listing names (`optimal_SB_size`, `QuantileMeanTBB`, `optimal_TBB_size`) map onto the papers above; the formulas below are taken from the papers, not from Masters.

---

## 1. Two distinct resampling stacks in the repo

The audits collapse “block bootstrap” into one missing object. The code has two independent stacks.

| Stack | Entry point | What is resampled | Block law | Wrap | Block-length choice |
| --- | --- | --- | --- | --- | --- |
| **A. Stationary bootstrap of OHLCV** | `bootstrap_ohlcv` | Bar descriptors (return, gap, wick fractions, volume), then reconstructed OHLC | Geometric, mean `mean_block`; \(p = 1/\texttt{mean_block}\) | Yes: `(i+1) mod n` | Caller argument; default **20**; **not** Politis–White |
| **B. Return-path “hat” resample** | `empirical_resample` via `adaptive_empirical_resample` | Simple returns | **Fixed** length \(L\), or IID if `block_length is None` | **No** (starts in `[0, n-L]`) | `suggest_block_length` heuristic, or caller \(L\) |

Stack A is Politis–Romano SB. Stack B is a fixed-length overlapping moving-block bootstrap in the Künsch (1989) / Liu–Singh (1992) family, without circular wrap and without tapering. They do not share an index generator, a block-length selector, or a data type.

---

## 2. What *is* implemented: Politis–Romano stationary bootstrap

### 2.1 Algorithm in the 1994 paper

Politis–Romano (1994, §2) construct a pseudo-series that is **stationary conditional on the original sample** by concatenating blocks of **random geometric length**:

- Block \(B_{i,b} = \{X_i,\ldots,X_{i+b-1}\}\). For \(j > N\), wrap: \(X_j = X_{j \bmod N}\) with \(X_0 = X_N\).
- Lengths \(L_1, L_2, \ldots\) iid Geometric(\(p\)), \(P(L=m) = p(1-p)^{m-1}\) for \(m=1,2,\ldots\), so \(E[L] = 1/p\).
- Starts \(I_1, I_2, \ldots\) iid Uniform\(\{1,\ldots,N\}\).
- Concatenate \(B_{I_k, L_k}\) until length \(N\).

Politis–White (2004, §3.1) restate the same scheme in a unified circular wrapping: \(Y_t = X_{t \bmod N}\); block sizes drawn from a Geometric with **mean** \(b\); this is case (B) of their general algorithm, “the stationary bootstrap (SB) of Politis and Romano (1994).” Case (A) (unit mass on integer \(b\)) is the **circular** bootstrap, not implemented here.

An equivalent one-step Markov form (same law): draw \(I_1\) uniform; then with probability \(1-p\) set \(I_t = I_{t-1}+1 \pmod n\), else draw a fresh uniform start. Block length is geometric with mean \(1/p\).

### 2.2 What the code does

`_stationary_bootstrap_indices` (`synthetic.py`):

- Rejects \(n < 2\) and `mean_block < 1` (`mean_block = 1` is documented as iid shuffle).
- `idx[0] = rng.integers(0, n)` — uniform start.
- `jump = rng.random(n-1) < (1.0 / mean_block)` — so \(p = 1/\texttt{mean_block}\).
- `idx[t] = fresh` on jump, else `(idx[t-1] + 1) % n`.

That is the 1994 Markov form, including circular wrap. Tests encode the two boundary behaviours (`tests/test_montecarlo.py::test_stationary_bootstrap_block_contiguity`): `mean_block=200` keeps consecutive indices with mean contiguity \(> 0.9\); `mean_block=1` drops below \(0.1\).

Module docstring and `docs/modules/montecarlo.md` already name this as Politis–Romano (1994). Verification status in the module: protocol endorsed by the notebook cross-reference (Masters’ SB/TBB *as a protocol*); **implemented from Politis–Romano (1994)**; the OHLC reconstruction is **not** notebook-covered.

### 2.3 Shape-preserving OHLC layer (not in Politis–Romano)

`bootstrap_ohlcv` does **not** resample raw close prices. It decomposes each original bar \(j\) into close-to-close return, open gap \(O_j / C_{j-1}\), wick fractions \(H/\max(O,C)\) and \(L/\min(O,C)\), and volume; then walks a synthetic close, rebuilding OHLC so \(H \ge \max(O,C)\) and \(L \le \min(O,C)\) hold by construction. The **index sequence** is SB; the **observations** being concatenated are these descriptors.

This reconstruction is a domain-specific application of SB. It is not stated in Politis–Romano (1994), which resamples the series \(\{X_t\}\) itself for standard errors / confidence regions of estimators on weakly dependent stationary data. Caveats that follow from the extra layer:

- Calendar timestamps are reused as-is (documented). The pseudo-path is not a new clock.
- Bar 0’s return is forced to 0; gap/return use `prev_close[0] = close[0]`.
- Circular wrap of **indices** can splice the last original bar onto the first. That is faithful to PR wrap of \(X_t\); on OHLC descriptors it can join an end-of-sample bar’s wick/gap onto a start-of-sample bar.
- Volume is copied, not regenerated.
- Adjacent geometric blocks still have a join discontinuity in the descriptor sequence — SB does not taper those joins (that is TBB’s job; see §4).
- The reconstructed price level is a multiplicative walk, not a circular permutation of the original price path. Intra-bar geometry is preserved per bar; the long-run price *level* sequence is not.

Callers use this as a **synthetic market for re-running path-dependent strategies** (tranche pullback MC harness, production-research bootstrapped Sharpes), which is a different use than PR’s “approximate the sampling distribution of \(T_N\).”

### 2.4 `mean_block` is a heuristic, not automatic \(b_{\mathrm{SB}}\)

- Function default: `mean_block: float = 20.0`.
- Tranche study: `MEAN_BLOCK = 20` with comment “~one trading month of preserved local structure.” Intraday variant: `mean_block_days: 20` scaled by bars-per-day.
- Production research: `bootstrap_ohlcv(..., mean_block=20, ...)`.

Nothing in `quantester/montecarlo/` estimates \(b\) from the autocovariance, a flat-top lag window, or a spectral density. Sensitivity is left to the caller. Politis–White (2004, p. 57) note that SB is *less sensitive to block-size misspecification* than circular/moving blocks — that is a reason a crude default can still be a valid SB — but it is **not** their automatic selector.

---

## 3. What *is* implemented: fixed-length block bootstrap after the ACF gate

### 3.1 Gate

`autocorrelation_gate` (`diagnostics.py`): Wald–Wolfowitz runs on \(\mathrm{sign}(x-\mathrm{median})\) and Ljung–Box on raw autocorrelations. If either \(p < \alpha\), `recommended_method = "block_bootstrap_or_ou_paths"`. This is the Kaufman “autocorrelation trap” router, not a Politis–White bandwidth rule.

### 3.2 Fixed-length overlapping blocks

`_block_draws` (`trade_resampling.py`): for each sim, draw `n_blocks = ceil(horizon / L)` starts uniformly from `{0,…,n-L}`, concatenate length-\(L\) slices, truncate to `horizon`. No wrap. No geometric lengths. No taper. This is overlapping moving blocks of **fixed** \(L\), not SB and not TBB.

### 3.3 `suggest_block_length` vs Politis–White \(b_{\mathrm{SB}}\)

`suggest_block_length` (`adaptive.py`) is documented as “a sensitivity starting point — not claimed optimal”:

- First lag \(k \in \{1,\ldots,\min(\texttt{max_lag}, n-1)\}\) with \(|\hat\rho_k| < 2/\sqrt{n}\).
- Else fallback `max(2, round(n**(1/3)))`.
- Clamped to `[2, n//2]`.

When the gate fires, `adaptive_empirical_resample` uses that \(L\) (or a caller `block_length`) and emits a warning that block length is a modelling assumption.

Contrast with Politis–White (2004, eqs. 6 and 9; 2009 correction for \(D_{\mathrm{SB}}\)):

\[
b_{\mathrm{opt,SB}} = \left(\frac{2G^2}{D_{\mathrm{SB}}}\right)^{1/3} N^{1/3},
\quad
\hat b_{\mathrm{opt,SB}} = \left(\frac{2\hat G^2}{\hat D_{\mathrm{SB}}}\right)^{1/3} N^{1/3}.
\]

After Patton–Politis–White (2009, items 1 and 4), the **correct** variance constant is \(D_{\mathrm{SB}} = 2g^2(0)\) (and \(\hat D_{\mathrm{SB}} = 2\hat g^2(0)\)). The 2004 paper’s \(D_{\mathrm{SB}} = 4g^2(0) + \frac{2}{\pi}\int_{-1}^{1}(1+\cos w)\hat g^2(w)\,dw\) form is **wrong**; any future selector must use the 2009 constant. (The 2004 \(D_{\mathrm{CB}} = \frac{4}{3}g^2(0)\) for circular/moving blocks was not withdrawn.)

\(\hat G\) and \(\hat g\) are **flat-top lag-window** estimators (Politis–Romano 1995 trapezoid \(\lambda(t) = 1\) for \(|t|\le 1/2\), \(2(1-|t|)\) for \(1/2<|t|\le 1\)): \(\hat G = \sum_{k=-M}^{M} \lambda(k/M)\,k\,\hat R(k)\). Bandwidth \(M=2m\), where \(m\) is the smallest integer after which the correlogram is negligible. The implied test (PW 2004 footnote, citing Politis 2001/2003): \(m\) = smallest positive integer such that \(|\hat\rho_{m+k}| < c\sqrt{\log_{10} N / N}\) for \(k=1,\ldots,K_N\), with practical \(c=2\) and \(K_N = \max(5, \log_{10} N)\).

`suggest_block_length` shares only a family resemblance (look at ACF until it is “small,” fallback \(n^{1/3}\)). It does **not** estimate \(G\) or \(g(0)\), does **not** use a flat-top window, does **not** form \(\hat b_{\mathrm{opt,SB}}\) or \(\hat b_{\mathrm{opt,CB}}\), and its output is a **fixed** moving-block length for stack B, not an expected geometric length for stack A.

### 3.4 Downstream consumers still IID

Even with stacks A and B present, several Monte Carlo consumers do **not** use either:

- `double_bootstrap_dd_bound` / `single_loop_dd_quantile` (`drawdown.py`): both inner and outer loops are `rng.integers` IID draws of simple returns. Masters nested “bound on a bound,” but the resample is iid, not SB/TBB.
- `permutation.py`: MCPT shuffles log-changes (Protocol I synchronized; Protocol II intra/inter-bar). That is a **permutation test**, not a bootstrap. The SSML audit’s request to “refactor `permutation.py` to support SB/TBB” mixes two protocols: MCPT destroys chronology while preserving the exact multiset of changes; SB *preserves* short-run dependence. They answer different questions.

---

## 4. What remains: Tapered Block Bootstrap

Paparoditis–Politis (2001, *Biometrika*): TBB is a **moving-block** variant. Each overlapping block is **centered**, multiplied by a data-taper window \(w_b\) that down-weights the edges, and RMS-normalized so tapering does not shrink variance:

\[
B_i^*(t) = \bar X + \frac{w_t\,(X_{i+t-1}-\bar X)}{\bigl(\ell^{-1}\sum_s w_s^2\bigr)^{1/2}}.
\]

Canonical window is trapezoidal with edge fraction \(c\); Paparoditis–Politis recommend \(c \approx 0.43\) (reused as \(\omega^{\mathrm{TRAP}}_{0.43}\) in Parker–Paparoditis–Politis 2015). Tapered blocks are then resampled like MBB.

Why TBB exists (2001; also PW 2004 footnote *a*): ordinary MBB/SB/CB have bias of order \(1/b\) for the variance estimator of the sample mean, MSE rate \(O(n^{-2/3})\). Tapering reduces that bias; MSE of the TBB variance estimator is \(O(n^{-4/5})\). The 2002 paper extends TBB to approximately linear statistics (smooth functions of means, M-estimators). Shao (2010) later proposed an *extended* TBB (taper the bootstrap weights rather than the data); that is also absent.

Optimal TBB block length (Paparoditis–Politis 2001, as restated in Parker–Paparoditis–Politis 2015 §7) is of order \(n^{1/5}\), not \(n^{1/3}\):

\[
b_{\mathrm{opt,TBB}} = \left(\frac{4\Gamma^2}{\Delta}\right)^{1/5} n^{1/5},
\]

with \(\Gamma,\Delta\) depending on the taper’s self-convolution and on \(\sum k^2 R(k)\) and \(\sigma^4\), again estimated with the same flat-top \(\lambda\) and \(M=2m\) correlogram rule.

**Repo scan:** no taper window, no `w_t`, no RMS normalization, no \(c=0.43\), no `b_opt,TBB`, no `QuantileMeanTBB`-style weighted quantile. Stack B concatenates raw return slices; stack A concatenates raw bar descriptors. Neither tapers.

**Design caveat if TBB is ever added:** TBB *alters* the observations (center + taper + rescale). Dropping that onto `bootstrap_ohlcv`’s multiplicative OHLC rebuild is not a mechanical overlay: wick fractions and gaps are not a sample-mean statistic, and tapering would break the “shape-preserving by construction” invariant unless redesigned. TBB as specified is a tool for **variance / distribution approximation of (approximately linear) statistics**, not a drop-in synthetic-OHLC generator.

Masters *Assessing* ch. 3 (per the audit, not independently read here) treats SB **or** TBB plus automatic \(b\) as the valid dependent-data bootstrap. The code already supplies the SB half of that “or.”

---

## 5. What remains: automatic \(b_{\mathrm{SB}}\) (and \(b_{\mathrm{CB}}\))

Implemented: user/heuristic `mean_block` (stack A) and ACF-threshold / \(n^{1/3}\) `suggest_block_length` (stack B).

Missing, from Politis–White (2004) + Patton–Politis–White (2009):

| Estimator | Formula | In repo? |
| --- | --- | --- |
| \(\hat g(w)\) flat-top spectral density | \(\sum_{k=-M}^{M} \lambda(k/M)\hat R(k)\cos(wk)\) | No |
| \(\hat G = \sum k\lambda(k/M)\hat R(k)\) | PW (2004) eq. (8) | No |
| \(\hat D_{\mathrm{SB}} = 2\hat g^2(0)\) | **2009** correction of PW eq. (8) | No |
| \(\hat b_{\mathrm{opt,SB}} = (2\hat G^2 / \hat D_{\mathrm{SB}})^{1/3} N^{1/3}\) | PW eq. (9) with 2009 \(D_{\mathrm{SB}}\) | No |
| \(\hat D_{\mathrm{CB}} = \frac{4}{3}\hat g^2(0)\) | PW eq. (13) | No |
| \(\hat b_{\mathrm{opt,CB}} = \hat b_{\mathrm{opt,MB}} = [(2\hat G^2 / \hat D_{\mathrm{CB}})^{1/3} N^{1/3}]\) | PW eq. (14); also the right selector for stack B *if* one wanted PW-optimal fixed \(L\) | No |
| Correlogram \(m\) then \(M=2m\) | PW §3.2 / Politis (2001) test | No (`suggest_block_length` is a different rule) |

PW optimality is **MSE of the bootstrap estimator of \(\sigma^2_\infty = \mathrm{Var}(\sqrt{N}\bar X)\)** under mixing and moment conditions (PW Theorem 3.1 hypotheses). It is **not** proven optimal for (a) reconstructed OHLCV strategy equity, (b) drawdown quantiles, or (c) Sharpe on `bootstrap_ohlcv` paths. Wiring \(\hat b_{\mathrm{opt,SB}}\) into `mean_block` would still be a heuristic relative to those statistics; it would only be the paper-correct choice for the sample-mean long-run variance problem the papers actually solve.

2009 also records \(\mathrm{ARE}_{\mathrm{CB/SB}} \to (2/3)^{2/3} \approx 0.763\): even with optimal \(b\), SB is asymptotically less efficient than circular/moving blocks for that variance estimator, because of extra randomization of block length. SB’s compensating property is **stationarity of the pseudo-path** (PW 2004, p. 57) — which is exactly why stack A uses it for synthetic markets.

---

## 6. Mapping the audit claims to the code

| Audit claim (*Assessing* / *SSML* dump) | Fact from code + papers |
| --- | --- |
| Stationary Bootstrap (Politis–Romano 1994) is missing | **False.** `_stationary_bootstrap_indices` + `bootstrap_ohlcv`. |
| Tapered Block Bootstrap (Paparoditis–Politis 2001) is missing | **True.** No taper, no TBB selector. |
| Automatic \(b_{\mathrm{SB}}\) / \(b_{\mathrm{TBB}}\) via flat-top autocovariance (Politis–White / Masters `optimal_SB_size` / `optimal_TBB_size`) is missing | **True** for the paper estimators. `mean_block=20` and `suggest_block_length` are not those estimators. |
| “Generic block bootstrap with unoptimized \(b\)” is all that exists | **Half-true.** Stack B is generic fixed-length MBB with a heuristic \(L\). Stack A is genuine SB with heuristic mean length. |
| Therefore drawdown double bootstrap and MCPT are unreliable | **Overstated / mixed protocols.** DD bound is IID (a real gap *if* one wants dependent-data DD CIs). MCPT is a permutation test by design, not a failed SB. |
| Action: implement a second SB, plus TBB, plus `QuantileMeanTBB` | **Do not implement a second SB.** Remaining optional specialty is TBB + PW/PPW automatic \(b\), and only if the product wants paper-optimal *inference* for mean-like statistics — not because SB is absent. |

Synthesis §4.11 already had this right: gap is **TBB + automatic \(b_{\mathrm{SB}}\)**, not “no stationary bootstrap.”

---

## 7. Inventory (for later spec work; not an implementation plan)

Already present — do not duplicate:

- Politis–Romano SB index law (geometric \(p=1/\texttt{mean_block}\), circular wrap, seeded `numpy.random.Generator`).
- Shape-preserving OHLCV reconstruction on top of that index law.
- Autocorrelation gate → fixed-length overlapping block bootstrap of simple returns, with an explicit non-optimal length heuristic.

Absent relative to the named papers:

- TBB (2001/2002): taper kernel, RMS normalization, recommended \(c=0.43\).
- Politis–White / Patton–Politis–White automatic \(\hat b_{\mathrm{opt,SB}}\) (flat-top \(\hat G,\hat g\), **2009** \(D_{\mathrm{SB}}=2g^2(0)\)).
- The same pipeline’s \(\hat b_{\mathrm{opt,CB}}=\hat b_{\mathrm{opt,MB}}\) for stack B.
- TBB’s \(\hat b_{\mathrm{opt,TBB}} \propto n^{1/5}\).
- Any use of SB/TBB inside `drawdown.py` (currently IID).

Out of scope for “missing SB”: MCPT permutation protocols; Ehlers parametric paths; OU paths; de Prado sequential bootstrap (mentioned in the `synthetic.py` docstring as an alternative, not implemented — a different method, not a second SB).
