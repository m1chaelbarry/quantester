# Data

Package: `quantester/data`

The data layer is the engine's **point-in-time market data stream** and the
enforcement point of the look-ahead firewall. Downstream components may only
observe market data through this interface — never through raw DataFrames.

## `DataHandler` (abstract interface)

`quantester/data/base.py`. Any new feed (database, API, parquet, …) subclasses
this and must honor the visibility contract exactly.

| Member | Signature | Contract |
| --- | --- | --- |
| `symbols` | property → `list` | All symbols in the stream. |
| `continue_backtest` | property → `bool` | `False` when the stream is exhausted. |
| `prime_data()` | → `None` | Reset the stream to just before the first bar. |
| `advance()` | → `(timestamp, bars)` | Move to the next timestamp; `bars` maps symbol → OHLCV `pd.Series` **or `None`** (untradeable). |
| `set_phase(phase, timestamp)` | → `None` | Set the firewall context: `"open"` or `"close"`. |
| `get_latest_bars(symbol, n=1)` | → `pd.DataFrame` | Trailing `n` bars visible **under the current phase**: during the open phase the current bar is excluded; during the close phase it is included. |
| `get_current_open(symbol)` | → `float \| None` | Current bar's open print (open phase); `None` if untradeable. |
| `timestamp_at_offset(timestamp, n)` | → `pd.Timestamp \| None` | Timestamp `n` bars later on the master calendar; `None` past the end. Used to stamp `earliest_fill_time`. |
| `current_timestamp` | property | The stream's current timestamp. |
| `bar_at(symbol, timestamp)` | → row or `None` | Execution-side full-bar lookup (used by the execution simulator). |

## `HistoricCSVDataHandler`

`quantester/data/csv_handler.py` — the bundled implementation.

```python
from quantester.data.csv_handler import HistoricCSVDataHandler

handler = HistoricCSVDataHandler({
    "AAPL": "data/AAPL.csv",   # path to CSV ...
    "MSFT": msft_dataframe,    # ... or pre-loaded DataFrame
})
```

- **CSV schema:** `datetime,open,high,low,close,volume` (datetime parsed as
  the index; all OHLCV columns coerced to float).
- Duplicate timestamps are dropped (first wins) and data is sorted by time.
- **Multi-symbol alignment:** the master calendar is the *union* of every
  symbol's timestamps (outer join). A symbol missing a bar at a timestamp is
  served as `None` — untradeable, never erased. Dropping incomplete bars
  would silently delete high-stress/illiquid periods and bias results.

## Information-driven bars

`quantester/data/bars.py` — alternative bar construction for tick/trade data
(AFML ch. 2). Input ticks: DataFrame indexed by datetime with columns
`price, volume`. Output: OHLCV bars indexed by bar-close time, ready to feed
into `HistoricCSVDataHandler`.

```python
from quantester.data.bars import (
    dollar_bars, tick_imbalance_bars,
    dollar_imbalance_bars, volume_imbalance_bars,
)

bars = dollar_bars(ticks, threshold=1_000_000)   # sample per $1M traded
bars = tick_imbalance_bars(ticks, span=10, warmup=3)
```

| Function | Samples a new bar when… |
| --- | --- |
| `dollar_bars(ticks, threshold)` | Cumulative `price × volume` ≥ `threshold`. Robust to price volatility and corporate actions. |
| `tick_imbalance_bars(ticks, span, warmup, initial_expected_len)` | \|Σ bₜ\| exceeds the EWMA-expected tick imbalance, where bₜ = ±1 by the tick rule. |
| `dollar_imbalance_bars(...)` | Same, with flows weighted by dollar volume. |
| `volume_imbalance_bars(...)` | Same, weighted by share/contract volume. |

Imbalance bars capture informed-trading bursts: they sample more frequently
when trade flow is one-sided and less when it is balanced, which tends to
produce return series closer to iid-normal than clock-time bars.

## Writing your own feed

```python
from quantester.data.base import DataHandler

class ParquetDataHandler(DataHandler):
    # implement every member above; the firewall contract is:
    #  - get_latest_bars excludes the current bar during the open phase
    #  - advance() serves None (not a gap in the calendar) for missing bars
    #  - timestamp_at_offset drives order stamping
    ...
```

Keep the three contract bullets exactly — the temporal firewall, the
pending-order ledger, and every regression test depend on them.
