# Must the engine require stops, forbid them, or support both families?

Type: grilling
Status: resolved
Part of: [Literature remediation decision map](../map.md)
Blocked by: 07

## Question

After [Are delay-1 market entries and resting intra-bar stops orthogonal policies?](07-delay-entries-vs-stops.md): must every strategy use resting stops, must some families forbid them, or must the engine support **both**?

Synthesis §4.10:

- Trend / microstructure: resting intra-bar stops (Harris, AFML, Penfold).
- Clenow momentum: **no** stop; ATR vol sizing (*Stocks on the Move*).

`FractionalRiskSizer` must not be the only vol sizer if Clenow is in-scope. Opt-in resting `STOP_ORDER` for Donchian/tranche is the synthesis’s non-conflicting suggestion — confirm or reject it here.

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.

## Answer

**Support both families.** Trend / microstructure strategies may rest intra-bar stops. Clenow-style momentum may forbid them and size on ATR / percent equity. `FractionalRiskSizer` is not the only vol sizer.

Opt-in resting `STOP_ORDER` for Donchian and tranche freeze **already shipped** (PR #27): stamp `stop_price` on delay-1 entry; default Donchian/Chan/Ehlers path remains delay-1 EXIT-on-touch. No further implementation ticket on this map.

## Comments

- 2026-08-19 notebook: both families; resting STOP opt-in is done.
