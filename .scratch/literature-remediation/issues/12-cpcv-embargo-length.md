# Is CPCV overlap geometry enough, or must embargo length be lookback/lookahead bars?

Type: grilling
Status: resolved
Part of: [Literature remediation decision map](../map.md)

## Question

Is Combinatorial Purged CV **overlap geometry** (purge overlapping label intervals) sufficient alignment with Masters / de Prado, or must embargo **length policy** change from `% of T` + median \(\Delta t\) to integer bars derived from lookback / lookahead (\(\min(\mathrm{lookback},\mathrm{look-ahead})-1\))?

Synthesis §4.4: both can be true — geometry aligned (*SSML*); length policy not *Assessing* ch. 1 / *Testing and Tuning* tight. Short datasets can under-embargo; long datasets can over-discard; median \(\Delta t\) smears irregular calendars.

This is a validation-policy decision, not a rewrite of purge geometry unless grilling finds geometry itself wrong.

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.

## Answer

**Keep overlap-geometry purging** (already aligned). Change **embargo length policy** from `% of T` + median \(\Delta t\) to an **integer bar** embargo derived from lookback / lookahead: \(\min(\mathrm{lookback},\mathrm{look-ahead})-1\) (Masters *Assessing* ch. 1 / TTMTS). `pct_embargo` may remain as an explicit override; it must not stay the silent default.

**Authority caveat:** the same notebook marked *Assessing* ch. 1 **NOT IN NOTEBOOK** (index p. 120 only). This is the weakest D-row. Implement the integer-bar policy as the product call, and record “not covered by the notebook — implemented from Assessing ch. 1 / TTMTS” in the module docstring. Do not rewrite purge geometry.

Implement: [Integer B/F-bar CPCV embargo](24-cpcv-embargo-bars.md).

## Comments

- 2026-08-19 notebook ruling **D8**. Weakest row: Assessing ch. 1 not in the notebook.
