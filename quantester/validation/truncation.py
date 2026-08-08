"""Ernest Chan's historical truncation validation check (Report 1 section 4.1).

Run a full backtest over the complete dataset (File A positions), then truncate
the last N bars and re-run the identical program (File B). Truncate File A to
File B's length: the two position files must agree within ``atol`` on the
common period.

This is a **strong temporal-leakage diagnostic**, not a formal mathematical
proof that all look-ahead is impossible. Any absolute discrepancy beyond
tolerance is evidence the pipeline consumed future information (subject to
documented warm-up behaviour).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TruncationResult:
    passed: bool
    n_truncated: int
    rows_compared: int
    mismatches: list

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"Truncation diagnostic [{status}]: compared {self.rows_compared} rows "
            f"after truncating {self.n_truncated} bars; "
            f"{len(self.mismatches)} mismatch(es)."
        )


def run_truncation_test(run_fn, n_truncated: int = 20,
                        atol: float = 1e-10) -> TruncationResult:
    """run_fn(truncate_last: int | None) -> positions DataFrame (timestamp x symbol).

    Executes the full and truncated backtests and compares overlapping rows
    with an absolute-difference check:

        abs(position_full[t] - position_truncated[t]) <= atol

    Both directions of discrepancy are caught (full > truncated and
    full < truncated).
    """
    full = run_fn(None)
    truncated = run_fn(n_truncated)
    if full.empty or truncated.empty:
        raise ValueError("Backtest produced no positions history to compare.")

    common = full.index.intersection(truncated.index)
    if len(common) == 0:
        return TruncationResult(
            passed=False,
            n_truncated=n_truncate,
            rows_compared=0,
            mismatches=[
                {
                    "error": "no overlapping rows between full and truncated runs",
                }
            ],
        )
    a = full.loc[common].sort_index()
    b = truncated.loc[common].sort_index()
    a, b = a.align(b, join="inner", axis=1)
    a, b = a.fillna(0.0), b.fillna(0.0)

    mismatches = []
    # Absolute difference: directional (full - truncated) > atol misses
    # negative discrepancies and can silently pass look-ahead bugs.
    abs_diff = np.abs(a.to_numpy(dtype=float) - b.to_numpy(dtype=float))
    bad = abs_diff > atol
    if bad.any():
        rows, cols = np.where(bad)
        for r, c in zip(rows[:20], cols[:20]):
            mismatches.append(
                {
                    "timestamp": a.index[r],
                    "symbol": a.columns[c],
                    "full": float(a.iloc[r, c]),
                    "truncated": float(b.iloc[r, c]),
                    "abs_diff": float(abs_diff[r, c]),
                }
            )
    return TruncationResult(
        passed=not mismatches,
        n_truncated=n_truncated,
        rows_compared=len(common),
        mismatches=mismatches,
    )
