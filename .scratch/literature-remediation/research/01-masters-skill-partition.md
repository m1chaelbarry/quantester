# What does Masters *Assessing* specify for the Trend/Bias/Skill partition?

**Verdict.** Timothy Masters, *Assessing and Improving Prediction and Classification* (Apress, 2018), does **not** use the names Trend / Bias / Skill. Its MCPT gain partition (companion listing `MC_TRAIN.CPP`, Chapter 5, printed book pp. 205–278) is:

\[
\begin{aligned}
\text{training\_bias} &= \text{mean\_permuted\_gain} - \text{mean\_inherent\_bias} \\
\text{unbiased\_actual\_gain} &= \text{original\_gain} - \text{training\_bias} \\
\text{unbiased\_gain\_above\_inherent\_bias} &= \text{unbiased\_actual\_gain} - \text{mean\_inherent\_bias}
\end{aligned}
\]

The last line is the Ability analogue: it subtracts **mean permuted inherent bias**, not the original-run inherent bias. The original-run quantity is saved only for display and is commented “not really important.”

Quantester `trend_bias_skill` implements \(\mathrm{Skill}=R_{\mathrm{unbiased}}-B_{\mathrm{orig}}\). Against *Assessing*, that is a **real formula discrepancy**, not a \(B_{\mathrm{orig}}\) vs “mean inherent bias” naming mix-up.

The same Skill formula **does** match Masters’ later trading-system listing (*Testing and Tuning Market Trading Systems*, `MCPT_TRN.CPP`): \(\mathrm{skill}=\mathrm{unbiased\_return}-\mathrm{original\_trend\_component}\). The repo’s “notebook-verified” claim asserts the TTMTS-style identities, not the *Assessing* Ability line.

The printed book PDF was **not available** in this environment. Formulas below are from the official Apress companion source tagged as corresponding to the published book.

---

## 1. Sources

### 1.1 Primary (used)

| Source | What it is | Access |
| --- | --- | --- |
| `quantester/montecarlo/permutation.py` | Repo docstring + `trend_bias_skill` | Workspace |
| Callers: `tests/test_montecarlo.py`, `examples/monte_carlo/run.py`, `examples/donchian_breakout/run_mcpt.py` | What \(B\) is passed as | Workspace |
| Timothy Masters, *Assessing and Improving Prediction and Classification*, Apress 2018, DOI [10.1007/978-1-4842-3336-8](https://doi.org/10.1007/978-1-4842-3336-8) | Target text. Ch. 5 “Miscellaneous Resampling Techniques” = **pp. 205–278** ([Springer chapter record](https://link.springer.com/chapter/10.1007/978-1-4842-3336-8_5)) | TOC / pagination only. **Full text and p. 276 listing not readable here.** |
| Apress companion C++ [Apress/assessing-and-improving-prediction-and-classification](https://github.com/Apress/assessing-and-improving-prediction-and-classification), file `MC_TRAIN.CPP`, commit `28736ace4e23f260aa4a19dbab092f668a96480a` (2017-12-18). README: “Release v1.0 corresponds to the code in the published book, without corrections or updates.” | First-party listing of the MCPT gain partition | Cloned and read |

### 1.2 Primary (related, not the ticketed book)

Used only to test whether \(R_{\mathrm{unbiased}}-B_{\mathrm{orig}}\) is Masters’ **trading** partition rather than a misread of *Assessing*.

| Source | Why |
| --- | --- |
| Apress companion [Apress/testing-and-tuning-market-trading-systems](https://github.com/Apress/testing-and-tuning-market-trading-systems) `MCPT_TRN/MCPT_TRN.CPP` and `MCPT_BARS/MCPT_BARS.CPP`, HEAD `0e195ba12afb7a54dba3a296bb354bf802ca819e`. TTMTS Ch. “Permutation Tests” = pp. 283–318 ([Springer book TOC](https://link.springer.com/book/10.1007/978-1-4842-4173-8)). | Names **Trend / Training bias / Skill**; Skill subtracts **original** trend. |
| Author site [timothymasters.info/market-trading.html](http://www.timothymasters.info/market-trading.html) | States TTMTS gives a limited MCPT overview; *Permutation and Randomization Tests for Trading System Development* is the dedicated MCPT book. That later book’s ZIP was **not** downloaded (no Apress GitHub mirror found). |

### 1.3 Not available / not treated as primary

- **Printed *Assessing* PDF / Google Books interior of p. 276.** Not accessible. Prose around the listing (definitions of “Ability”, worked numerical example, any erratum) is **unverified**.
- **2013 CreateSpace edition** of *Assessing* (ISBN 1484137450, 562 pp.) has different pagination; a “p. 276” cite is only consistent with the **2018 Apress** chapter-5 end (ch. 5 ends p. 278).
- Author errata pages for *Assessing*: none found for `MC_TRAIN`.
- [`3rd Cross Reference.md`](../../../3rd%20Cross%20Reference.md) and [`3rd Cross Reference Synthesis.md`](../../../3rd%20Cross%20Reference%20Synthesis.md): **claim under test only**, not evidence.
- In-repo restatements (`SYSTEM_ARCHITECTURE_QUANTITATIVE_METHODOLOGY_BLUEPRINT.md` §6.6, `docs/modules/montecarlo.md`, `docs/glossary.md`, `NOTEBOOK_CONTEXT.md`, `Monte Carlo.md`): secondary copies of the code formula, not Masters.

---

## 2. Claim under test

From [issues/01-masters-skill-partition.md](../issues/01-masters-skill-partition.md) and the Assessing audit in `3rd Cross Reference.md` (quoted only as the claim):

Repo today (notebook-verified):

- \(\mathrm{Bias}=R_{\mathrm{perm}}-B_{\mathrm{perm}}\)
- \(R_{\mathrm{unbiased}}=R_{\mathrm{orig}}-\mathrm{Bias}\)
- \(\mathrm{Skill}=R_{\mathrm{unbiased}}-B_{\mathrm{orig}}\) with \(\mathrm{Trend}=B_{\mathrm{orig}}\)

Audit claim: *Assessing* p. 276 C++ computes Ability as \(R_{\mathrm{unbiased}}-B_{\mathrm{perm}}\) (mean of permuted inherent biases).

The same audit writes \(\text{TrainingBias}=B_{\mathrm{perm}}-B_{\mathrm{inherent\_mean}}\). That identity is **not** in the companion C++ (see §4). Training bias there is \(\text{mean\_permuted\_gain}-\text{mean\_inherent\_bias}\).

---

## 3. What the repo’s “notebook-verified” claim asserts

Repo convention ([`.cursor/rules/quant-literature-notebook.mdc`](../../../.cursor/rules/quant-literature-notebook.mdc)): *notebook-verified* means the formula was checked against the user’s Gemini Notebook of specialist books, including Masters. The rule lists “Masters p-value + Trend/Bias/Skill partition” among already-verified items.

`permutation.py` module docstring asserts all of:

```
Bias       = R_perm - B_perm
R_unbiased = R_orig - Bias = Skill + Trend
Skill      = R_unbiased - B_orig        (Trend = B_orig)
Benchmarks are recomputed on permuted paths, not assumed zero.
```

`trend_bias_skill` is tagged with the same phrase: `"""Masters' partition of total return (notebook-verified)."""`

Protocol II reconstruction in the same file is explicitly **not** notebook-covered.

The notebook itself was **not** re-read in this research pass. The claim above is what the docstring asserts, not independent confirmation that the notebook page matches *Assessing* vs TTMTS.

---

## 4. Exact textbook-side formulas (*Assessing* companion C++)

File: `MC_TRAIN.CPP` — “Demonstrate Monte-Carlo permutation training” on synthetic credit-card fraud (linear model + optimized decision threshold). Not a price-series Trend/Skill trading demo.

Identifiers (lines 31–33):

```
original_gain
inherent_bias, mean_inherent_bias, original_inherent_bias
mean_permuted_gain, training_bias
unbiased_actual_gain, unbiased_gain_above_inherent_bias
```

There is **no** identifier `Ability`, `Skill`, or `Trend`. The audit’s “Ability” maps to `unbiased_gain_above_inherent_bias`.

### 4.1 Inherent bias (worthless-system expected gain)

Per replication, after the model’s predicted-fraud rate \(c_{\mathrm{fraud}}\) is known (lines 242–247):

```
inherent_bias = p_legit * c_legit * gain_ll
              + p_legit * c_fraud * gain_lf
              + p_fraud * c_legit * gain_fl
              + p_fraud * c_fraud * gain_ff
```

Comment on that block: “Gain expected from a similar but worthless system.” Class priors \(p_{\mathrm{fraud}},p_{\mathrm{legit}}\) are computed once from the **unpermuted** labels (lines 138–139) and are **not** recomputed after shuffling. Predicted-class rates \(c\) **are** recomputed each replication.

### 4.2 Original vs permuted accumulation (lines 148–149, 249–254)

```
mean_inherent_bias = 0.0 ;   // Computed from permuted only
mean_permuted_gain = 0.0 ;   // Ditto
...
if (irep == 0)
   original_inherent_bias = inherent_bias ; // Needed only to display for user; not really important
else {
   mean_inherent_bias += inherent_bias ;    // These are cumulated for permutations only, not original
   mean_permuted_gain += best_gain ;
}
```

`original_inherent_bias` is **not** an input to training bias, unbiased gain, or Ability.

### 4.3 Partition after all replications (lines 276–281, 288–293)

```
original_gain /= ncases ;
mean_inherent_bias /= nreps-1 ;
mean_permuted_gain /= ncases * (nreps-1) ;
training_bias = mean_permuted_gain - mean_inherent_bias ;
unbiased_actual_gain = original_gain - training_bias ;
unbiased_gain_above_inherent_bias = unbiased_actual_gain - mean_inherent_bias ;
```

Print annotations (the book listing’s own algebra):

- Training bias = mean permuted gain **minus** mean permuted inherent bias
- Unbiased actual gain = original gain **minus** training bias
- Unbiased gain above inherent bias = unbiased actual gain **minus** mean permuted inherent bias

Algebraic collapse (the two \(\text{mean\_inherent\_bias}\) terms cancel):

\[
\text{unbiased\_gain\_above\_inherent\_bias}
= \text{original\_gain} - \text{mean\_permuted\_gain}.
\]

### 4.4 Symbol map onto the repo’s \(R,B\) notation

| *Assessing* C++ | Repo / audit symbol |
| --- | --- |
| `original_gain` | \(R_{\mathrm{orig}}\) |
| `mean_permuted_gain` | \(R_{\mathrm{perm}}\) |
| `original_inherent_bias` | \(B_{\mathrm{orig}}\) (display only) |
| `mean_inherent_bias` | \(B_{\mathrm{perm}}\) |
| `training_bias` | \(\mathrm{Bias}=R_{\mathrm{perm}}-B_{\mathrm{perm}}\) |
| `unbiased_actual_gain` | \(R_{\mathrm{unbiased}}\) |
| `unbiased_gain_above_inherent_bias` | Ability / Skill analogue \(= R_{\mathrm{unbiased}}-B_{\mathrm{perm}}\) |

p. 276: not independently confirmed. It is consistent with Apress Ch. 5 ending at p. 278 and with `CODE_DESCRIPTION.TXT` listing `MC_TRAIN.CPP` as the Monte-Carlo permutation-training program.

---

## 5. Exact code formulas (Quantester)

`trend_bias_skill` (`quantester/montecarlo/permutation.py`):

```python
def trend_bias_skill(r_orig: float, b_orig: float, r_perm: float,
                     b_perm: float) -> dict:
    """Masters' partition of total return (notebook-verified)."""
    bias = r_perm - b_perm
    unbiased = r_orig - bias
    return {
        "trend": b_orig,
        "training_bias": bias,
        "unbiased_return": unbiased,
        "skill": unbiased - b_orig,
    }
```

\[
\begin{aligned}
\mathrm{Bias} &= R_{\mathrm{perm}} - B_{\mathrm{perm}} \\
R_{\mathrm{unbiased}} &= R_{\mathrm{orig}} - \mathrm{Bias} \\
\mathrm{Trend} &= B_{\mathrm{orig}} \\
\mathrm{Skill} &= R_{\mathrm{unbiased}} - B_{\mathrm{orig}}
\end{aligned}
\]

Unit test (`tests/test_montecarlo.py::test_trend_bias_skill_partition`): \(R_{\mathrm{orig}}=0.30\), \(B_{\mathrm{orig}}=0.10\), \(R_{\mathrm{perm}}=0.15\), \(B_{\mathrm{perm}}=0.08\) expects `training_bias=0.07`, `unbiased_return=0.23`, `skill=0.13`, `trend=0.10`. The *Assessing* Ability line on the same numbers is \(0.23-0.08=0.15\).

### 5.1 Callers (what \(B\) means in practice)

The helper is a four-scalar identity; it does not compute inherent bias the *Assessing* way (class-prior expected gain) nor the TTMTS way (\((N_{\mathrm{long}}-N_{\mathrm{short}})\times\) trend-per-return).

| Caller | `b_orig` | `b_perm` |
| --- | --- | --- |
| `examples/monte_carlo/run.py` | annualized mean simple return of the **original** close | mean of the same statistic on **fresh** `permute_log_changes` draws (20 extra shuffles, not the MCPT sample) |
| `examples/donchian_breakout/run_mcpt.py` | window buy-and-hold \(C_{\mathrm{end}}/C_{\mathrm{start}}-1\) | mean buy-and-hold of the Protocol-II permuted windows |

`permutation_test` itself does **not** call `trend_bias_skill`.

---

## 6. \(R_{\mathrm{unbiased}}-B_{\mathrm{orig}}\) vs \(R_{\mathrm{unbiased}}-B_{\mathrm{perm}}\)

**Against *Assessing* companion C++: real discrepancy.**

| Piece | *Assessing* `MC_TRAIN.CPP` | Quantester `trend_bias_skill` |
| --- | --- | --- |
| Training bias | \(R_{\mathrm{perm}}-B_{\mathrm{perm}}\) | \(R_{\mathrm{perm}}-B_{\mathrm{perm}}\) — **same** |
| Unbiased gain/return | \(R_{\mathrm{orig}}-\mathrm{Bias}\) | \(R_{\mathrm{orig}}-\mathrm{Bias}\) — **same** |
| Skill / Ability | \(R_{\mathrm{unbiased}}-B_{\mathrm{perm}}\) | \(R_{\mathrm{unbiased}}-B_{\mathrm{orig}}\) — **differs** |
| Role of \(B_{\mathrm{orig}}\) | Display only (“not really important”) | Equals Trend; subtracted to form Skill |

Difference:

\[
\mathrm{Skill}_{\text{code}} - \mathrm{Ability}_{\textit{Assessing}} = B_{\mathrm{perm}} - B_{\mathrm{orig}}.
\]

They coincide iff \(B_{\mathrm{orig}}=B_{\mathrm{perm}}\). The C++ comment that the original inherent bias is display-only is incompatible with treating that substitution as a notation mix-up **inside Assessing**.

It is **not** unverifiable without the PDF for the **listing**: Apress states the tagged source matches the printed code. What remains unverifiable is whether surrounding **prose** on p. 276 uses the word “Ability”, whether a figure uses different symbols, and whether any later erratum changed the subtraction. No such erratum was found.

---

## 7. Why the notebook formula can still be “Masters” (different book)

*Testing and Tuning Market Trading Systems* companion `MCPT_TRN.CPP` (header: “Estimate true skill and unbiased future return”) uses the **names** Trend / Training bias / Skill and subtracts the **original** trend:

```
trend_per_return = (prices[nprices-1] - prices[max_lookback-1]) / (nprices - max_lookback);
...
trend_component = (nlong - nshort) * trend_per_return;   // each replication
...
training_bias = opt_return - trend_component;            // permutations only
mean_training_bias /= (nreps - 1);
unbiased_return = original - mean_training_bias;
skill = unbiased_return - original_trend_component;
```

Identical Skill line in `MCPT_BARS.CPP`.

`trend_per_return` is computed **once** on the unpermuted log prices. `do_permute` shuffles log-changes and rebuilds from the first price, so the endpoint (and thus `trend_per_return`) is invariant; `trend_component` still varies through \((n_{\mathrm{long}}-n_{\mathrm{short}})\).

Mapping:

\[
\begin{aligned}
\mathrm{Bias} &= \overline{R_{\mathrm{perm},i}-T_i} \\
R_{\mathrm{unbiased}} &= R_{\mathrm{orig}}-\mathrm{Bias} \\
\mathrm{Skill} &= R_{\mathrm{unbiased}} - T_{\mathrm{orig}}
\end{aligned}
\]

That is the docstring identity with \(B_{\mathrm{orig}}=T_{\mathrm{orig}}\) and \(B_{\mathrm{perm}}=\overline{T_i}\). Quantester’s four-scalar helper matches this **structure**. It does **not** implement TTMTS’s position-count trend \(T=(N_{\mathrm{long}}-N_{\mathrm{short}})\times\) trend-per-return; callers pass a raw benchmark return.

So:

- Ticket question is *Assessing* → Ability subtracts \(B_{\mathrm{perm}}\). Code does not.
- A different first-party Masters listing (TTMTS) → Skill subtracts \(T_{\mathrm{orig}}\). Code does.

Those two Masters listings disagree with each other by \(B_{\mathrm{perm}}-B_{\mathrm{orig}}\) (or \(T_{\mathrm{perm}}-T_{\mathrm{orig}}\)).

---

## 8. Facts only (no product recommendation)

1. *Assessing* MCPT partition (companion `MC_TRAIN.CPP`): \(\mathrm{Ability}=R_{\mathrm{unbiased}}-B_{\mathrm{perm}}\). \(B_{\mathrm{orig}}\) is unused in the arithmetic.
2. Quantester `trend_bias_skill`: \(\mathrm{Skill}=R_{\mathrm{unbiased}}-B_{\mathrm{orig}}\). Training-bias and unbiased-return lines match *Assessing*; the Skill line does not.
3. That Skill line matches TTMTS `MCPT_TRN.CPP` / `MCPT_BARS.CPP`.
4. The notebook-verified docstring asserts the TTMTS-style identities and the name “Trend/Bias/Skill,” which *Assessing* does not use.
5. Printed *Assessing* prose at p. 276 was not read. The listing formulas above do not depend on that prose.
6. The third-cross-reference TrainingBias identity \(B_{\mathrm{perm}}-B_{\mathrm{inherent\_mean}}\) is not in the companion C++. Its Ability identity \(R_{\mathrm{unbiased}}-B_{\mathrm{perm}}\) is.
