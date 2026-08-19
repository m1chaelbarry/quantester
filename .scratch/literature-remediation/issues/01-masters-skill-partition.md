# What does Masters Assessing specify for the Trend/Bias/Skill partition?

Type: research
Status: open
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
