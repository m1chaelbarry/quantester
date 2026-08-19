# What is the AFML imbalance-bar threshold estimator versus current code?

Type: research
Status: open
Part of: [Literature remediation decision map](../map.md)

## Question

What is the exact AFML chapter 2 recurrence for volume / dollar imbalance-bar thresholds — \(E_0[T]\), \(P[b_t=1]\), \(v^\pm\), and \(|2v^+ - E_0[v_t]|\) with EWMA — versus the current tick-flow concatenation in this repo?

The synthesis calls the mismatch a critical flaw (AFML only). Confirm against de Prado’s text (or an honest “book not in tree”) and against the implementation (likely `quantester/data/` bar construction). Record: (1) textbook algorithm steps, (2) what the code actually computes, (3) whether leftover-tick emission and constant warmup are extra deviations.

Do not treat the blueprint or the synthesis as primary for the textbook side.

Write findings to [`../research/02-afml-imbalance-estimator.md`](../research/02-afml-imbalance-estimator.md).
