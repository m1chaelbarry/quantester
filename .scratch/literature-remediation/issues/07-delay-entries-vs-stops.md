# Are delay-1 market entries and resting intra-bar stops orthogonal policies?

Type: grilling
Status: open
Part of: [Literature remediation decision map](../map.md)

## Question

Must the engine split **delay-1 market entries** (close \(T\) → open \(T+1\), open-phase H/L/C redacted) from **resting intra-bar stops** (`STOP_ORDER` on the ledger, gap-through at the next available price)?

Synthesis §4.3: delay-1 *entries* are aligned (Ehlers *Cycle Analytics*, Penfold, Chan look-ahead, Clenow). Strategy stops today are delay-1 `EXIT` from close, not resting `STOP_ORDER`, which Harris / AFML 3.2 / Chan / Carver / Vince rate as an extra gap bar.

Do not collapse “signal delay-1” with “stop delay-1”. Clenow’s no-stop family is [Must the engine require stops, forbid them, or support both families?](11-stops-required-vs-forbidden.md). Delay-0 is [Keep delay-0 as a firewall feature, or require minimum latency?](08-delay-0-policy.md).

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code.
