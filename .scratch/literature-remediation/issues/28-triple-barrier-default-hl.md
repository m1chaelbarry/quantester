# Default triple-barrier to high/low

Type: task
Status: open
Part of: [Literature remediation decision map](../map.md)

## Goal

When OHLC high/low exist, triple-barrier labels must **default** to the high/low path. Close-only is an explicit opt-out.

Ruling: notebook D12 (AFML ch. 3). Optional high/low already shipped (#27); this ticket flips the **default** when the frame has those columns.

## Files

- `quantester/strategy/meta_labeling.py` — `triple_barrier_labels`, `fit_secondary` (`high`/`low` default `None` → currently substituted with `close`)
- `tests/test_strategy.py` — `test_triple_barrier_labels` (close-only), `test_triple_barrier_labels_use_high_low_path`, `test_triple_barrier_same_bar_both_hit_is_stop`

## Acceptance criteria

- If the caller passes a DataFrame / dict with `high` and `low`, or `fit_secondary` is given a close series whose index aligns with available high/low from the same bars, **use them** without requiring the caller to thread kwargs.
- Practical API: `triple_barrier_labels(..., high=None, low=None, *, path: str = "auto")` where `path="auto"` uses high/low when both are provided **or** when a new `ohlc: pd.DataFrame` argument is passed; `path="close"` forces today’s close-only labels.
- Same-bar both-hit remains stop (`y=0`) — already shipped, do not regress.
- Close-only tests pass by passing `path="close"` (or omitting high/low with `path="close"`).
- Docstring: AFML ch. 3 path dependence; high/low default when OHLC exists. Notebook-verified intent; first-touch high/low was previously “not covered beyond intent.”

## Tests

- OHLC frame with a wick through the stop and close back inside → label 0 under default auto, 1 (or close-path outcome) under `path="close"` if the close never breached.
- Existing close-only unit tests still pass with explicit close path.
- `fit_secondary` with an OHLC frame uses high/low without extra kwargs.

## Out of scope

- Changing vertical-barrier terminal (still close).
- Meta-label model architecture.
