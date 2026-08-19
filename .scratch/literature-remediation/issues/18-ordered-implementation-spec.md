# What is the ordered implementation spec after the rulings?

Type: grilling
Status: resolved
Part of: [Literature remediation decision map](../map.md)
Blocked by: 06, 08, 09, 11, 12, 14, 15, 16, 17

## Question

Once the blocking rulings exist, what is the **ordered implementation spec** a later effort must follow — first wave, parked, and explicit do-not-touch alignments — without re-opening literature conflicts?

Start from synthesis §5 (safe without a ruling) and rewrite it under the actual decisions. This ticket **is** the destination artifact. Link it from the map’s Decisions-so-far; do not paste the full spec into the map.

Invoke `/grilling` and `/domain-modeling`. Do not answer for the human. Do not write engine code on this map.

## Answer

Locked registry: [`../spec.md`](../spec.md).

Implementation order (later effort; this map does not patch the engine):

1. [Switch tearsheet Sharpe to simple returns](19-simple-tearsheet-sharpe.md) (D1)
2. [Annualize with measured periods-per-year](20-measured-periods-per-year.md) (D2)
3. [Fast-track Sharpe must call annualized_sharpe](23-fast-track-sharpe-parity.md) (D7; blocked by 19, 20)
4. Independent: [Default Vince gap_stress to 1.0](21-vince-gap-stress-default.md), [Forbid delay-0 fills by default](22-forbid-delay-0-default.md), [Integer B/F-bar CPCV embargo](24-cpcv-embargo-bars.md), [Unadjusted Yahoo prices plus dividend cash](25-unadjusted-dividend-cash.md), [Size live sizers on cash, not MTM equity](26-cash-base-sizers.md), [Session-close drawdown breaker](27-session-close-dd-breaker.md), [Default triple-barrier to high/low](28-triple-barrier-default-hl.md)

KEEP / already shipped: delay-1 entries; both stop families with opt-in resting STOP; adjacent live sizers (no LSM-on-every-fill); TTMTS Skill partition; Ledoit–Wolf as library; cash yield `/365`; visualization rolling Sharpe simple; PRs #24–#27 uncontroversial defects.

Do-not-touch alignments: ETF-trick \(c_t\) external to \(K_t\); MCPT Protocol I same shuffle index; fill_price already embeds costs; stops gap-through; RNG via `numpy.random.Generator`.

## Comments

- 2026-08-19 notebook D1–D12 folded into `spec.md` and task tickets 19–28.
