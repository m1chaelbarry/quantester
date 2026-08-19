# What is the AFML imbalance-bar threshold estimator versus current code?

Ticket: [What is the AFML imbalance-bar threshold estimator versus current code?](../issues/02-afml-imbalance-estimator.md).
This note is the **AFML Chapter 2 fact versus `quantester/data/bars.py`**, not a product recommendation.

## Verdict

The AFML sampling **condition** for volume / dollar imbalance bars is

\[
T^\ast=\arg\min_T\bigl\{\,|\theta_T|\ge E_0[T]\,\lvert 2v^+-E_0[v_t]\rvert\,\bigr\},
\qquad
\theta_T=\sum_{t=1}^{T}b_t v_t,
\qquad
v^+=P[b_t=1]\,E_0[v_t\mid b_t=1].
\]

In practice de Prado estimates **two** EWMA inputs from **prior bars**: \(E_0[T]\) from completed bar lengths \(T^\ast\), and the composite \(2v^+-E_0[v_t]\) from signed sizes \(b_t v_t\) (tick analogue: \(2P[b_t=1]-1\) from \(b_t\)). He does **not** require three separate EWMAs of \(P[b=1]\), \(E[v\mid b=1]\), and \(E[v\mid b=-1]\) in the practical estimator — those identities **define** \(2v^+-E_0[v_t]\).

Quantester uses the same **product shape** after warmup: \(\mathrm{EWMA}(\text{bar lengths})\times\lvert\mathrm{EWMA}(\text{concatenated per-tick signed flows})\rvert\). That is **not** the textbook estimator. The EWMA of lengths is one observation per completed bar; the EWMA of flows is one observation per **tick**, and both share the same pandas `span` (default 10). Memory for expected imbalance is then ~10 **ticks**, not ~10 **bars**. That unit mismatch is the material deviation.

Two further deviations the book does not specify at all: (1) until `warmup` bars complete, the threshold is the **constant** `max(initial_expected_len, 1.0)`, used as a \(|\theta|\) cutoff in whatever units \(\theta\) has; (2) leftover ticks that never hit \(T^\ast\) are **always flushed** as a final bar.

**López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018) is not in this tree.** No PDF/epub of Chapter 2 was found. Textbook steps below are from (a) a Quant.SE transcription of the TIB pages and (b) de Prado’s own 2017 LBNL slides that he labels as the forthcoming AFML text. No official Wiley/quantresearch.org errata revising this estimator was found.

---

## 1. Sources

### 1.1 Primary (used)

| Source | What it is | Access |
| --- | --- | --- |
| `quantester/data/bars.py` | `_imbalance_bars`, `tick_imbalance_bars`, `volume_imbalance_bars`, `dollar_imbalance_bars`, `_tick_rule`, `_ewma_last` | Workspace |
| `tests/test_data.py` `test_tick_imbalance_bars_structure` | Only structural TIB test (OHLC envelope + last close). No VIB/DIB, leftover, or warmup-unit test. | Workspace |
| López de Prado, *Advances in Financial Machine Learning*, Wiley 2018, **Chapter 2.3.2 Information-Driven Bars** (TIB ~p. 29; VIB/DIB follows). | Target text. | **Not in `/workspace`.** No PDF/epub. |
| Quant.SE [Tick Imbalance Bars — clarification on T index](https://quant.stackexchange.com/questions/44757/tick-imbalance-bars-clarification-on-t-index) (2019-03-24), block labelled “From textbook”. | Long TIB extract of the tick rule, \(\theta_T\), \(E_0[\theta_T]=E_0[T](2P[b_t=1]-1)\), EWMA practice sentence, and \(T^\ast\). | Fetched |
| López de Prado, “The 7 Reasons Most Machine Learning Funds Fail,” LBNL Computational Research Division, 2 Sep 2017. Header: contents based on forthcoming *AFML*, Wiley (2017). Pitfall #3 “Example 1: Dollar Bars” (math is signed-volume / dollar **imbalance**). | First-party VIB/DIB identities \(v^\pm\), \(E_0[\theta_T]=E_0[T](2v^+-E_0[v_t])\), EWMA practice sentence, \(T^\ast\). | [PDF](https://pdfs.semanticscholar.org/bbf7/bc8f68d22cb8089a4860b111ba9ef60fc957.pdf) |

### 1.2 Not available / not treated as primary

- **Printed AFML PDF / Google Books interior of pp. 28–30.** Not in the tree. Prose around the VIB/DIB EWMA sentence in the bound book is therefore **not line-checked here**.
- **Official errata** (Wiley companion or [quantresearch.org](https://www.quantresearch.org/)): none found that revises the Chapter 2 imbalance threshold.
- User Gemini Notebook: not readable in this environment. Module docstring still claims “notebook-verified formulas.”
- [`SYSTEM_ARCHITECTURE_QUANTITATIVE_METHODOLOGY_BLUEPRINT.md`](../../../SYSTEM_ARCHITECTURE_QUANTITATIVE_METHODOLOGY_BLUEPRINT.md) §2.3, [`3rd Cross Reference.md`](../../../3rd%20Cross%20Reference.md), [`3rd Cross Reference Synthesis.md`](../../../3rd%20Cross%20Reference%20Synthesis.md) §1.6: **claim under test**, not evidence for the textbook algorithm.
- Hudson & Thames `mlfinlab` / Quant.SE implementation answers: published **interpretations** of unspecified EWMA windows, cited only in §6 as such.

---

## 2. Textbook algorithm (AFML Ch. 2)

Indexing is intra-bar: \(t=1,\ldots,T\) restarts after each emitted bar. \(v_t\) is 1 for TIB, share/contract volume for VIB, and dollar volume \(p_t v_t\) for DIB.

### 2.1 Tick rule (shared)

From the Quant.SE textbook block:

\[
b_t=\begin{cases}
b_{t-1} & \text{if }\Delta p_t=0,\\
|\Delta p_t|/\Delta p_t & \text{if }\Delta p_t\neq 0,
\end{cases}
\qquad b_t\in\{-1,1\}.
\]

Boundary: \(b_0\) of a new bar is set to the terminal \(b_T\) of the immediately preceding bar. The first tick of the sample is not specified in the extract.

### 2.2 Tick imbalance bars

1. Accumulate \(\theta_T=\sum_{t=1}^{T}b_t\).
2. At the **beginning** of the bar, form
   \(E_0[\theta_T]=E_0[T]\,(P[b_t=1]-P[b_t=-1])=E_0[T]\,(2P[b_t=1]-1)\).
3. **Practice sentence (book extract):** estimate \(E_0[T]\) as an EWMA of \(T\) values from **prior bars**, and \((2P[b_t=1]-1)\) as an EWMA of \(b_t\) values from **prior bars**.
4. Emit at the first \(T\) with \(|\theta_T|\ge E_0[T]\,|2P[b_t=1]-1|\).
5. Reset \(\theta\) and start the next contiguous subset.

The Quant.SE paste labels step 2’s left-hand side as “\(E_o[T]\)”; that is a transcription slip. The algebra is \(E_0[\theta_T]\).

### 2.3 Volume / dollar imbalance bars

From the 2017 slides (forthcoming-AFML source), reconstructing OCR:

1. \(\theta_T=\sum_{t=1}^{T}b_t v_t\), with \(v_t\) either size or dollar amount.
2. Decompose
   \(v^+=P[b_t=1]\,E_0[v_t\mid b_t=1]\),
   \(v^-=P[b_t=-1]\,E_0[v_t\mid b_t=-1]\),
   so \(E_0[v_t]=v^++v^-\) and
   \(E_0[\theta_T]=E_0[T](v^+-v^-)=E_0[T](2v^+-E_0[v_t])\).
3. **Practice sentence (slides):** estimate \(E_0[T]\) as an EWMA of \(T\) values from prior bars, and \(2v^+-E_0[v_t]\) as an EWMA of **signed-size values from prior bars**.
4. A bar is a \(T^\ast\)-contiguous subset with
   \(T^\ast=\arg\min_T\{|\theta_T|\ge E_0[T]\,|2v^+-E_0[v_t]|\}\).
5. When flow is more one-sided than expected, a **small** \(T\) meets the inequality.

### 2.4 What the book/slides do not specify

- Initial \(E_0[T]\) (or initial \(2P-1\) / \(2v^+-E[v]\)) before any bar exists.
- A count of completed bars (`warmup`) during which the threshold is a raw constant.
- Flushing a trailing incomplete subset that never hits \(T^\ast\).
- The EWMA span, \(\alpha\), or whether the two EWMAs share a window.
- pandas `ewm(span=…)` versus a recursive \(\alpha\)-filter.

“\(b_t\) values from prior bars” is tick-level wording; “\(T\) values from prior bars” is bar-level. The two series do not have the same observation frequency. The book does not say to feed them the same `span`.

---

## 3. Algorithm as coded (`quantester/data/bars.py`)

One private loop `_imbalance_bars(ticks, weighted, span, warmup, initial_expected_len)` implements TIB, VIB, and DIB.

**Inputs.** DataFrame indexed by datetime, columns `price`, `volume`. Defaults: `span=10`, `warmup=3`, `initial_expected_len=50.0`.

**Tick rule.** `_tick_rule` walks the **entire** price series once. First tick is hard-coded \(b_0=+1\). Later ticks: \(\mathrm{sign}(\Delta p)\), with zeros inheriting the previous sign. Because the series is global, bar boundaries do not reset \(b\); that is consistent with “\(b_0\) matches previous bar’s \(b_T\)” once the first tick exists.

**Per-tick flow.**

| Wrapper | `weighted` | Flow \(f_t\) |
| --- | --- | --- |
| `tick_imbalance_bars` | False | \(b_t\) |
| `volume_imbalance_bars` | True | \(b_t\times\mathrm{volume}_t\) |
| `dollar_imbalance_bars` | True, after `volume ← price×volume` | \(b_t\times p_t v_t\) |

**State.** `bar_lengths: list` (one int per **completed** bar). `bar_flows: list` (every per-tick flow from **all completed** bars concatenated). Current bar: `start`, running `theta`.

**Each tick \(t\):**

1. `theta += flow[t]`; `length = t - start + 1`.
2. **Warmup** (`len(bar_lengths) < warmup`):
   `threshold = max(initial_expected_len, 1.0)` — a constant, not a product.
3. **After warmup:**
   `expected_len = pandas.Series(bar_lengths).ewm(span=span).mean().iloc[-1]`
   `expected_imb = abs(pandas.Series(bar_flows).ewm(span=span).mean().iloc[-1])`
   `threshold = max(expected_len * expected_imb, 1e-12)`.
4. If `abs(theta) >= threshold`: emit OHLCV on `ticks[start:t+1]`, append `length` to `bar_lengths`, **extend** `bar_flows` with `flow[start:t+1]`, reset `start=t+1` and `theta=0`.

`bar_lengths` / `bar_flows` update only on emit, so the post-warmup threshold is **frozen during a bar** (textbook \(E_0\) at bar start). `_ewma_last` rebuilds a pandas Series on every tick after warmup (performance, not sampling semantics). pandas `ewm(span)` uses \(\alpha=2/(\mathrm{span}+1)\) (default `adjust=True`) over the **whole** stored list.

**Leftover.** After the tick loop: `if start < len(ticks): emit ticks[start:]`. The tail is a bar even if \(|\theta|\) never reached the threshold. `dollar_bars` does the same flush.

**Docstring vs body.** The module docstring writes \(E_0[\theta_T]=E_0[T](2P[b=1]-1)\) for TIB and \(E_0[T](2v^+-E_0[v_t])\) “via EWMA of \(b_t v_t\)” for VIB/DIB. The body never forms \(P[b=1]\), \(v^+\), or \(E[v_t]\) as named statistics. It multiplies an EWMA of bar lengths by the absolute EWMA of concatenated signed ticks.

---

## 4. Steps side by side

| Step | Textbook / de Prado 2017 slides | `bars.py` |
| --- | --- | --- |
| Classify ticks | Tick rule; \(b_0\) from previous bar’s \(b_T\) | Same rule; sample \(b_0=+1\) |
| Running imbalance | \(\theta=\sum b_t\) or \(\sum b_t v_t\) inside the current bar | Same (`theta`) |
| \(E_0[T]\) | EWMA of **prior bar lengths** \(T^\ast\) | EWMA of `bar_lengths` (bar-level) with `span` |
| Expected imbalance | TIB: EWMA of **\(b_t\) from prior bars**. VIB/DIB: EWMA of **signed sizes** from prior bars, equal to \(2v^+-E_0[v_t]\) by definition | \(\lvert\mathrm{EWMA}_{\mathrm{span}}(\texttt{bar\_flows})\rvert\) on concatenated ticks, **same `span`** |
| Components \(P[b=1]\), \(E[v\mid\pm 1]\) | Used to **define** \(2v^+-E[v]\). Practice estimator is the composite EWMA | Never computed |
| Threshold | \(E_0[T]\times\lvert 2P-1\rvert\) or \(E_0[T]\times\lvert 2v^+-E[v]\rvert\), fixed at bar open | After warmup: `expected_len * expected_imb`. During warmup: `max(initial_expected_len, 1.0)` |
| Emit | First \(T\) with \(\lvert\theta_T\rvert\ge\) that product | Same inequality against `threshold` |
| Incomplete tail | Not a bar under \(T^\ast=\arg\min\{\ldots\}\) | Always emitted |
| First bars | Unspecified initial \(E_0[T]\) (and initial imbalance) | `warmup` completed bars under a **constant** \(\lvert\theta\rvert\) cap |

---

## 5. Deviations

1. **Shared `span` on unlike series (material).** `bar_lengths` has one number per bar; `bar_flows` has one number per tick. Default `span=10` ⇒ expected length remembers ~10 bars, expected imbalance remembers ~10 ticks (\(\alpha=2/11\)). Textbook: \(E_0[T]\) is bar-frequency; imbalance EWMA is “from prior bars,” not “last `span` ticks.”

2. **Concatenation, not bar-level imbalance.** Completing a bar **extends** `bar_flows` with every tick in that bar. Longer bars dump more points into the EWMA. A bar-level reading of “prior bars” would EWMA one imbalance statistic per bar (mean \(b_t\), or \(\theta_{T^\ast}/T^\ast\), or mean signed size). Concatenation plus tick-`span` makes `expected_imb` ≈ \(\lvert\)EWMA of the last handful of ticks\(\rvert\).

3. **Composite EWMA vs named components — not a textbook miss.** Synthesis text that the code must EWMA \(P[b=1]\) and the two conditional sizes **separately** overstates the practical sentence. Definition uses \(v^\pm\); practice EWMAs \(T\) and signed size. Code’s *target* \(\mathrm{EWMA}(b_t v_t)\approx E[b_t v_t]=2v^+-E[v_t]\) is the right scalar. The miss is the **window**, not the algebra of \(v^+\).

4. **Constant warmup (extra vs book).** See §7. Book/slides do not define a `warmup` counter or a non-product threshold.

5. **Leftover flush (extra vs book).** See §6. Same pattern as `dollar_bars`.

6. **Sample \(b_0=+1\).** Book extract only specifies the between-bar boundary.

7. **Docstring.** Claims the AFML product and “notebook-verified formulas.” Body is \(\mathrm{EWMA}(T)\times\lvert\mathrm{EWMA}(\text{ticks})\rvert\) with a constant warmup cap.

---

## 6. Leftover-tick behavior

**Code.** After the last tick, any non-empty buffer `ticks[start:]` is passed to `_make_bar` and appended. There is no \(\lvert\theta\rvert\) check and no drop.

**Textbook.** A bar is only a \(T^\ast\)-contiguous subset that meets the inequality. A residual shorter than that is not \(T^\ast\).

**Probe (no source edits):** 13 all-buy ticks, `initial_expected_len=5`, `warmup=5` → TIB volumes `[5, 5, 3]`. The `3` never reached 5. Ten ticks at $30/tick through `dollar_bars(threshold=100)` → volumes `[12, 12, 6]` (same leftover policy). Mixed-sign TIB of 80 ticks: bar volumes summed to 80 — every tick sits in some output row.

`tests/test_data.py` `test_dollar_bars_threshold` already allows `len(bars) in (crossings, crossings+1)` because of this flush. The TIB test only checks that the last close equals the last tick price, which **requires** leftover emission.

---

## 7. Constant-warmup behavior

**Code.** While `len(bar_lengths) < warmup` (default 3), every tick is tested against `threshold = max(initial_expected_len, 1.0)`. That scalar is **not** an initial \(E_0[T]\) multiplied by an initial \(\lvert 2v^+-E[v]\rvert\). It is compared directly to \(\lvert\theta\rvert\).

Units therefore follow \(\theta\):

| Mode | \(\theta\) | Default cap 50 means |
| --- | --- | --- |
| TIB | net tick count | need \(\lvert\sum b_t\rvert\ge 50\) |
| VIB | signed share volume | need \(\lvert\sum b_t v_t\rvert\ge 50\) (often 1–2 ticks if \(v_t\gg 50\)) |
| DIB | signed dollar volume | still smaller in tick count |

**Probe:** 120 all-buy ticks, volume 10, `initial_expected_len=20`, `warmup=3`.

- TIB: six bars of 20 ticks (`volume` 200). Cap 20 vs \(\theta=\#\text{ticks}\).
- VIB: 60 bars of **2** ticks (`volume` 20). Cap 20 vs \(\theta=10\) per tick → emit on tick 2. After warmup, `bar_lengths` are 2s and `bar_flows` are \(+10\)s, so the product stays \(2\times 10=20\) and the 2-tick cadence continues.

The book/slides leave the first \(E_0[T]\) unspecified. A practitioner initial **tick count** used as \(E_0[T]\) in the **product** \(E_0[T]\times\lvert\text{imbalance}\rvert\) is a gap-fill; using that same number as the entire \(\lvert\theta\rvert\) threshold during `warmup` bars is a further, coded rule. It is especially unlike VIB/DIB, where \(\theta\) is not dimensionless.

After warmup the first product uses however many completed bars exist (`warmup` points in `bar_lengths`, and every tick inside them in `bar_flows`). pandas `ewm(span=10)` on three length observations is a short-history EWMA, not a seeded recursive filter at `initial_expected_len`.

---

## 8. EWMA inputs (explicit)

| List | When appended | Observation | `span` meaning |
| --- | --- | --- | --- |
| `bar_lengths` | On emit | Tick count of that bar | ~`span` **bars** |
| `bar_flows` | On emit, `extend` of every \(f_t\) in the bar | Signed tick (TIB) or signed size/dollar (VIB/DIB) | ~`span` **ticks** |

Current-bar ticks are **not** in `bar_flows` until emit — no look-ahead into the open bar’s own flows for \(E_0\).

---

## 9. Ambiguity left in the book (not resolved by this repo)

The practice sentence does not fix the imbalance EWMA’s window in ticks versus bars. Third-party code (e.g. Hudson & Thames `mlfinlab` comments citing AFML p. 29) uses **two** windows: `num_prev_bars` for \(E[T]\) and a large tick window for expected imbalance. That is an implementation choice, not a sentence in the sources of §1.1. This note does not adopt it as AFML.

---

## 10. Fact close

Mismatch versus AFML Ch. 2 is **real**: same product *shape* after warmup, wrong EWMA **unit** on the imbalance side, plus leftover flush and a constant warmup cap the text does not define. The mismatch is **not** “code EWMAs concatenated ticks instead of estimating \(P[b=1]\) and \(v^\pm\) as three named series” — de Prado’s practical estimator is already the composite \(2v^+-E_0[v_t]\) (or \(2P-1\)) times \(E_0[T]\).
