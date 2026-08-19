# What does Masters Assessing specify for the Trend/Bias/Skill partition?

Type: research
Status: resolved
Part of: [Literature remediation decision map](../map.md)

## Question

What does Timothy Masters, *Assessing and Improving Prediction and Classification*, actually specify for the MCPT Trend / Bias / Skill (Ability) partition — especially around p. 276 / the C++ listing — and how does that compare to the notebook-verified formulas in `quantester/montecarlo/permutation.py`?

Repo today (module docstring, marked notebook-verified):

- Bias = \(R_{\mathrm{perm}} - B_{\mathrm{perm}}\)
- \(R_{\mathrm{unbiased}} = R_{\mathrm{orig}} - \mathrm{Bias}\)
- Skill = \(R_{\mathrm{unbiased}} - B_{\mathrm{orig}}\) with Trend = \(B_{\mathrm{orig}}\)

The third-cross-reference Assessing audit claims Skill / Ability must subtract \(B_{\mathrm{perm}}\) (mean permuted inherent bias), not \(B_{\mathrm{orig}}\).

Resolve the **fact**, not the product call. Cite the book/C++ (or an honest “not available”) and the code. Do not treat [`3rd Cross Reference.md`](../../../3rd%20Cross%20Reference.md) as primary.

Write findings to [`../research/01-masters-skill-partition.md`](../research/01-masters-skill-partition.md).

## Answer

Against *Assessing* companion `MC_TRAIN.CPP`, Ability is \(R_{\mathrm{unbiased}}-B_{\mathrm{perm}}\) (mean permuted inherent bias). Repo `trend_bias_skill` uses \(R_{\mathrm{unbiased}}-B_{\mathrm{orig}}\). That is a real *Assessing* discrepancy, not a naming mix-up. The same Skill line **does** match Masters *Testing and Tuning* `MCPT_TRN.CPP`. Notebook-verified identities follow TTMTS, not *Assessing*. Printed p.276 prose was not available; formulas are from the official Apress listing.

Detail: [research/01-masters-skill-partition.md](../research/01-masters-skill-partition.md).

## Product decision

**KEEP TTMTS** `Skill = R_{\mathrm{unbiased}} - B_{\mathrm{orig}}` as the engine formula (notebook **D6**, TTMTS ch. 7). Do not switch Ability to \(R_{\mathrm{unbiased}}-B_{\mathrm{perm}}\). Document the *Assessing* discrepancy in the module docstring; do not treat it as a defect.

No implementation ticket. MCPT still shuffles **log** price changes; that is a documented exception vs simple tearsheet Sharpe ([Which Sharpe representation is canonical for tearsheet, MCPT, and DSR?](05-canonical-sharpe.md)), not a convert-on-the-boundary of the same function.

## Comments

- 2026-08-19 notebook ruling **D6**. Research fact stands; product call is KEEP TTMTS.
