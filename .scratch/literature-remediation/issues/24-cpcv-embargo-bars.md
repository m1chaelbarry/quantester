# Integer B/F-bar CPCV embargo

Type: task
Status: resolved
Part of: [Literature remediation decision map](../map.md)

## Goal

Keep label-overlap **purge geometry**. Change embargo **length** from `pct_embargo * T` converted with median \(\Delta t\) to an integer bar window \(\min(B, F)-1\) (lookback / lookahead).

Ruling: [Is CPCV overlap geometry enough, or must embargo length be lookback/lookahead bars?](12-cpcv-embargo-length.md) (notebook D8).

**Authority caveat:** *Assessing* ch. 1 is **not in the notebook** (index p. 120 only). This is the weakest D-row. Docstring must say “not covered by the notebook — implemented from Assessing ch. 1 / TTMTS,” not “notebook-verified.”

## Files

- `quantester/validation/cpcv.py` — `PurgedKFold`, `CombinatorialPurgedKFold`, `_as_offset`
- `tests/test_validation.py` — `pct_embargo=0.1` comments (“h = 3 bars”)
- Module docstring currently claims notebook-verified purge + `h = pct_embargo * T`

## Acceptance criteria

- New primary knobs: `lookback: int | None`, `lookahead: int | None` (or a single `embargo_bars: int | None`). Default embargo bars = \(\max(\min(B,F)-1, 0)\) when both are set; if only one is set, use that minus 1.
- Embargo is **integer index positions** on `X`, not `median(diff(index)) * pct * T`.
- `pct_embargo` remains an explicit override for de Prado ~0.01T research; it must not stay the silent default once lookback/lookahead are provided. Pick one product default and document it: prefer `embargo_bars` required-or-default-0 with lookback/lookahead recommended in the docstring, rather than silently keeping 0.01T.
- Purge overlap geometry (three de Prado conditions) unchanged.
- `n_paths` binomial identity already shipped — do not touch.

## Tests

- Irregular calendar: a weekend/gap in the index does **not** stretch embargo by median \(\Delta t\).
- `lookback=10`, `lookahead=5` → embargo 4 bars after test-end.
- `pct_embargo` override still drops the documented extra train labels on a regular daily index.
- Existing geometry tests with `pct_embargo=0.0` still pass.

## Out of scope

- Rewriting CPCV group combinatorics.
- Walk-forward / NTEST–EXTRA product (parked).

## Answer

Shipped (commit e49c373). Embargo is integer index positions: `embargo_bars` direct, else `min(lookback, lookahead) - 1` (single-sided: that minus 1), else explicit `pct_embargo` floored to bars; default 0 (the silent 0.01T is gone). Median-dt conversion deleted, so calendar gaps never stretch the window; purge geometry and the `n_paths` binomial identity untouched. Module docstring marks D8 as "not covered by the notebook — implemented from Assessing ch. 1 / TTMTS".
