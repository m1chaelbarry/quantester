"""Ernest Chan's historical truncation validation check (Report 1 section 4.1).

Run a full backtest over the complete dataset (File A positions), then truncate
the last N bars and re-run the identical program (File B). Truncate File A to
File B's length: the two position files must be mathematically identical. Any
mismatch proves the pipeline is leak-prone and consuming future data.
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
            f"Truncation test [{status}]: compared {self.rows_compared} rows "
            f"after truncating {self.n_truncated} bars; "
            f"{len(self.mismatches)} mismatch(es)."
        )


def run_truncation_test(run_fn, n_truncated: int = 20,
                        atol: float = 1e-10) -> TruncationResult:
    """run_fn(truncate_last: int | None) -> positions DataFrame (timestamp x symbol).

    Executes the full and truncated backtests and compares overlapping rows.
    """
    full = run_fn(None)
    truncated = run_fn(n_truncated)
    if full.empty or truncated.empty:
        raise ValueError("Backtest produced no positions history to compare.")

    common = full.index.intersection(truncated.index)
    a = full.loc[common].sort_index()
    b = truncated.loc[common].sort_index()
    a, b = a.align(b, join="inner", axis=1)
    a, b = a.fillna(0.0), b.fillna(0.0)

    mismatches = []
    diff = (a.to_numpy(dtype=float) - b.to_numpy(dtype=float)) > atol
    if diff.any():
        rows, cols = np.where(diff)
        for r, c in zip(rows[:20], cols[:20]):
            mismatches.append(
                {
                    "timestamp": a.index[r],
                    "symbol": a.columns[c],
                    "full": float(a.iloc[r, c]),
                    "truncated": float(b.iloc[r, c]),
                }
            )
    return TruncationResult(
        passed=not mismatches,
        n_truncated=n_truncated,
        rows_compared=len(common),
        mismatches=mismatches,
    )
