"""Centralized SQLite3 Trials Registry (Cross-Ref section 3.3).

Every backtest/optimization run logs its parameters, Sharpe ratio, and return
moments, providing the N (trial count) and sigma^2_SR (cross-trial Sharpe
variance) that the Deflated Sharpe Ratio mathematically requires (Cross-Ref
section 2.B).

Parallel-safe write path (Cross-Ref-2 section 4.1 -- SQLite rejects concurrent
writers): parallel workers serialize trial records to per-worker JSONL files
(`write_jsonl_record`), followed by a single-threaded `import_jsonl` batch
import after the optimization run completes.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np


class TrialsRegistry:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                params TEXT NOT NULL,
                sharpe REAL NOT NULL,
                mean REAL,
                std REAL,
                skew REAL,
                kurt REAL,
                n_obs INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def log_trial(self, params: dict, sharpe: float, mean: float | None = None,
                  std: float | None = None, skew: float | None = None,
                  kurt: float | None = None, n_obs: int | None = None,
                  run_id: str | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO trials (run_id, params, sharpe, mean, std, skew, kurt, n_obs)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, json.dumps(params, sort_keys=True), float(sharpe),
             mean, std, skew, kurt, n_obs),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def n_trials(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0])

    def sharpe_values(self) -> np.ndarray:
        rows = self._conn.execute("SELECT sharpe FROM trials ORDER BY id").fetchall()
        return np.array([r[0] for r in rows], dtype=float)

    def sharpe_variance(self) -> float:
        values = self.sharpe_values()
        return float(np.var(values, ddof=1)) if len(values) > 1 else 0.0

    def best_trial(self) -> dict | None:
        row = self._conn.execute(
            "SELECT id, params, sharpe, skew, kurt, n_obs FROM trials"
            " ORDER BY sharpe DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "params": json.loads(row[1]),
            "sharpe": row[2],
            "skew": row[3],
            "kurt": row[4],
            "n_obs": row[5],
        }

    # ------------------------------------------------- parallel-safe write path

    @staticmethod
    def write_jsonl_record(path, record: dict) -> None:
        """Per-worker serialization during parallel optimization."""
        with open(Path(path), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def import_jsonl(self, path) -> int:
        """Single-threaded batch import of worker JSONL files post-run."""
        count = 0
        with open(Path(path), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.log_trial(
                    params=rec["params"],
                    sharpe=rec["sharpe"],
                    mean=rec.get("mean"),
                    std=rec.get("std"),
                    skew=rec.get("skew"),
                    kurt=rec.get("kurt"),
                    n_obs=rec.get("n_obs"),
                    run_id=rec.get("run_id"),
                )
                count += 1
        return count

    def close(self) -> None:
        self._conn.close()
