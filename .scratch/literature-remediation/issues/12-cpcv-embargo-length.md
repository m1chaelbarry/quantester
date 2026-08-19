# Is CPCV overlap geometry enough, or must embargo length be lookback/lookahead bars?

Type: grilling
Status: open
Part of: [Literature remediation decision map](../map.md)

## Question

Is Combinatorial Purged CV **overlap geometry** (purge overlapping label intervals) sufficient alignment with Masters / de Prado, or must embargo **length policy** change from `% of T` + median \(\Delta t\) to integer bars derived from lookback / lookahead (\(\min(\mathrm{lookback},\mathrm{look-ahead})-1\))?

Synthesis §4.4: both can be true — geometry aligned (*SSML*); length policy not *Assessing* ch. 1 / *Testing and Tuning* tight. Short datasets can under-embargo; long datasets can over-discard; median \(\Delta t\) smears irregular calendars.

This is a validation-policy decision, not a rewrite of purge geometry unless grilling finds geometry itself wrong.

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.
