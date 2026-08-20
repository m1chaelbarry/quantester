"""Centralized SQLite3 Trials Registry (Cross-Ref section 3.3).

Every backtest/optimization run logs its parameters, Sharpe ratio, and return
moments, providing the N (trial count) and sigma^2_SR (cross-trial Sharpe
variance) that the Deflated Sharpe Ratio mathematically requires (Cross-Ref
section 2.B).

Automatic experiment registration records strategy identity, parameter set,
data source/range/universe, cost/execution/validation config, random seed,
metrics, timestamp, and a code/version identifier, hashed into a unique
experiment id so researchers cannot silently omit losers from the trial
universe that DSR/PBO consume.

Parallel-safe write path (Cross-Ref-2 section 4.1 -- SQLite rejects concurrent
writers): parallel workers serialize trial records to per-worker JSONL files
(`write_jsonl_record`), followed by a single-threaded `import_jsonl` batch
import after the optimization run completes.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def experiment_hash(payload: dict) -> str:
    """Stable SHA-256 over a canonical JSON payload (sorted keys)."""
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


class TrialsRegistry:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                experiment_hash TEXT,
                strategy_id TEXT,
                params TEXT NOT NULL,
                sharpe REAL NOT NULL,
                mean REAL,
                std REAL,
                skew REAL,
                kurt REAL,
                n_obs INTEGER,
                metrics TEXT,
                data_source TEXT,
                data_range TEXT,
                universe TEXT,
                cost_model TEXT,
                execution_config TEXT,
                validation_config TEXT,
                random_seed TEXT,
                code_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after the original schema (SQLite ALTER)."""
        existing = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(trials)").fetchall()
        }
        for col, decl in (
            ("experiment_hash", "TEXT"),
            ("strategy_id", "TEXT"),
            ("metrics", "TEXT"),
            ("data_source", "TEXT"),
            ("data_range", "TEXT"),
            ("universe", "TEXT"),
            ("cost_model", "TEXT"),
            ("execution_config", "TEXT"),
            ("validation_config", "TEXT"),
            ("random_seed", "TEXT"),
            ("code_version", "TEXT"),
        ):
            if col not in existing:
                self._conn.execute(f"ALTER TABLE trials ADD COLUMN {col} {decl}")

    def log_trial(
        self,
        params: dict,
        sharpe: float,
        mean: float | None = None,
        std: float | None = None,
        skew: float | None = None,
        kurt: float | None = None,
        n_obs: int | None = None,
        run_id: str | None = None,
        *,
        strategy_id: str | None = None,
        metrics: dict | None = None,
        data_source: str | None = None,
        data_range: str | dict | None = None,
        universe: list | str | None = None,
        cost_model: str | dict | None = None,
        execution_config: dict | None = None,
        validation_config: dict | None = None,
        random_seed: int | str | None = None,
        code_version: str | None = None,
        experiment_id: str | None = None,
    ) -> int:
        payload = {
            "strategy_id": strategy_id,
            "params": params,
            "data_source": data_source,
            "data_range": data_range,
            "universe": universe,
            "cost_model": cost_model,
            "execution_config": execution_config,
            "validation_config": validation_config,
            "random_seed": random_seed,
            "code_version": code_version,
        }
        exp_hash = experiment_id or experiment_hash(payload)
        cur = self._conn.execute(
            """
            INSERT INTO trials (
                run_id, experiment_hash, strategy_id, params, sharpe,
                mean, std, skew, kurt, n_obs, metrics,
                data_source, data_range, universe, cost_model,
                execution_config, validation_config, random_seed, code_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                exp_hash,
                strategy_id,
                json.dumps(params, sort_keys=True, default=str),
                float(sharpe),
                mean,
                std,
                skew,
                kurt,
                n_obs,
                json.dumps(metrics, sort_keys=True, default=str) if metrics else None,
                data_source,
                json.dumps(data_range, sort_keys=True, default=str)
                if isinstance(data_range, dict)
                else data_range,
                json.dumps(universe, default=str)
                if isinstance(universe, (list, dict))
                else universe,
                json.dumps(cost_model, sort_keys=True, default=str)
                if isinstance(cost_model, dict)
                else cost_model,
                json.dumps(execution_config, sort_keys=True, default=str)
                if execution_config
                else None,
                json.dumps(validation_config, sort_keys=True, default=str)
                if validation_config
                else None,
                None if random_seed is None else str(random_seed),
                code_version,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def register_experiment(
        self,
        *,
        strategy_id: str,
        params: dict,
        sharpe: float,
        metrics: dict | None = None,
        data_source: str | None = None,
        data_range: str | dict | None = None,
        universe: list | str | None = None,
        cost_model: str | dict | None = None,
        execution_config: dict | None = None,
        validation_config: dict | None = None,
        random_seed: int | str | None = None,
        code_version: str | None = None,
        run_id: str | None = None,
        mean: float | None = None,
        std: float | None = None,
        skew: float | None = None,
        kurt: float | None = None,
        n_obs: int | None = None,
    ) -> dict:
        """Automatic-friendly registration returning the experiment hash + row id."""
        if metrics:
            mean = mean if mean is not None else metrics.get("mean")
            std = std if std is not None else metrics.get("std")
            skew = skew if skew is not None else metrics.get("skew")
            kurt = kurt if kurt is not None else metrics.get("kurt")
            n_obs = n_obs if n_obs is not None else metrics.get("n_obs")
            sharpe = metrics.get("sharpe", sharpe)
        row_id = self.log_trial(
            params=params,
            sharpe=sharpe,
            mean=mean,
            std=std,
            skew=skew,
            kurt=kurt,
            n_obs=n_obs,
            run_id=run_id,
            strategy_id=strategy_id,
            metrics=metrics,
            data_source=data_source,
            data_range=data_range,
            universe=universe,
            cost_model=cost_model,
            execution_config=execution_config,
            validation_config=validation_config,
            random_seed=random_seed,
            code_version=code_version,
        )
        row = self._conn.execute(
            "SELECT experiment_hash, created_at FROM trials WHERE id = ?",
            (row_id,),
        ).fetchone()
        return {
            "id": row_id,
            "experiment_hash": row[0],
            "created_at": row[1],
            "n_trials": self.n_trials(),
        }

    def n_trials(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0])

    def sharpe_values(self) -> np.ndarray:
        rows = self._conn.execute("SELECT sharpe FROM trials ORDER BY id").fetchall()
        return np.array([r[0] for r in rows], dtype=float)

    def sharpe_variance(self) -> float:
        values = self.sharpe_values()
        return float(np.var(values, ddof=1)) if len(values) > 1 else 0.0

    def experiment_ids(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT experiment_hash FROM trials ORDER BY id"
        ).fetchall()
        return [r[0] for r in rows if r[0]]

    def best_trial(self) -> dict | None:
        row = self._conn.execute(
            "SELECT id, params, sharpe, mean, std, skew, kurt, n_obs, "
            "experiment_hash, strategy_id FROM trials ORDER BY sharpe DESC "
            "LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "params": json.loads(row[1]),
            "sharpe": row[2],
            "mean": row[3],
            "std": row[4],
            "skew": row[5],
            "kurt": row[6],
            "n_obs": row[7],
            "experiment_hash": row[8],
            "strategy_id": row[9],
        }

    # ------------------------------------------------- parallel-safe write path

    @staticmethod
    def write_jsonl_record(path, record: dict) -> None:
        """Per-worker serialization during parallel optimization."""
        with open(Path(path), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

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
                    strategy_id=rec.get("strategy_id"),
                    metrics=rec.get("metrics"),
                    data_source=rec.get("data_source"),
                    data_range=rec.get("data_range"),
                    universe=rec.get("universe"),
                    cost_model=rec.get("cost_model"),
                    execution_config=rec.get("execution_config"),
                    validation_config=rec.get("validation_config"),
                    random_seed=rec.get("random_seed"),
                    code_version=rec.get("code_version"),
                    experiment_id=rec.get("experiment_hash"),
                )
                count += 1
        return count

    def close(self) -> None:
        self._conn.close()


def auto_register_from_equity(
    registry: TrialsRegistry,
    equity,
    *,
    strategy_id: str,
    params: dict,
    **kwargs: Any,
) -> dict:
    """Compute return moments from an equity curve and register the trial.

    Moments are taken from **simple** returns to match the D1 canonical
    ``analytics.performance`` Sharpe convention (Carver cost drag is linear
    in simple-return Sharpe). Log returns remain available via
    ``analytics.returns`` for the documented Masters MCPT exception.
    """
    from ..analytics.performance import annualized_sharpe
    from ..analytics.returns import simple_returns_from_equity
    from scipy.stats import kurtosis, skew

    import pandas as pd

    if not isinstance(equity, pd.Series):
        equity = pd.Series(equity)
    rets = simple_returns_from_equity(equity)
    metrics = {
        "sharpe": float(annualized_sharpe(equity)),
        "mean": float(rets.mean()) if len(rets) else None,
        "std": float(rets.std()) if len(rets) else None,
        "skew": float(skew(rets)) if len(rets) > 2 else None,
        # Pearson kurtosis (normal = 3) — required by Bailey DSR formula.
        "kurt": float(kurtosis(rets, fisher=False)) if len(rets) > 3 else None,
        "n_obs": int(len(rets)),
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    return registry.register_experiment(
        strategy_id=strategy_id,
        params=params,
        sharpe=metrics["sharpe"],
        metrics=metrics,
        mean=metrics["mean"],
        std=metrics["std"],
        skew=metrics["skew"],
        kurt=metrics["kurt"],
        n_obs=metrics["n_obs"],
        **kwargs,
    )
