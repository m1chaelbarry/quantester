"""Performance metrics, trials registry, DSR, tearsheet."""

import numpy as np
import pandas as pd
import pytest

from quantester.analytics.dsr import (
    deflated_sharpe_ratio,
    dsr_from_registry,
    expected_max_sharpe,
)
from quantester.analytics.performance import (
    SPEED_LIMIT_SR,
    annualized_sharpe,
    calmar_ratio,
    carver_cost_drag_sr,
    max_drawdown,
    speed_limit_warning,
)
from quantester.analytics.tearsheet import generate_tearsheet
from quantester.analytics.trials_registry import TrialsRegistry


def _equity_from_log_returns(log_rets, start=100.0):
    idx = pd.bdate_range("2024-01-01", periods=len(log_rets) + 1)
    return pd.Series(start * np.exp(np.concatenate([[0.0], np.cumsum(log_rets)])), index=idx)


def test_sharpe_manual_annualization():
    log_rets = np.tile([0.002, -0.001], 50)
    equity = _equity_from_log_returns(log_rets)
    rets = pd.Series(log_rets)
    expected = (rets.mean() / rets.std(ddof=1)) * np.sqrt(252)
    assert annualized_sharpe(equity) == pytest.approx(float(expected), rel=1e-9)


def test_max_drawdown_known_path():
    equity = pd.Series(
        [100, 120, 90, 110, 130],
        index=pd.bdate_range("2024-01-01", periods=5),
        dtype=float,
    )
    result = max_drawdown(equity)
    assert result["max_drawdown"] == pytest.approx(90 / 120 - 1)
    assert result["peak"] == equity.index[1]
    assert result["trough"] == equity.index[2]
    assert result["duration"] >= 0


def test_calmar_positive_for_growth():
    equity = _equity_from_log_returns(np.tile([0.003, -0.001], 60))
    assert calmar_ratio(equity) > 0


def test_carver_drag_and_speed_limit():
    drag = carver_cost_drag_sr(annual_turnover=4.0, standardized_cost_sr=0.01)
    assert drag == pytest.approx(0.04)
    assert speed_limit_warning(drag) is None
    assert speed_limit_warning(10 * 0.01) is not None  # 0.10 > 0.08 limit
    assert SPEED_LIMIT_SR == 0.08


def test_registry_round_trip_and_batch_import(tmp_path):
    registry = TrialsRegistry()
    for i, sharpe in enumerate([0.5, 1.2, 0.8]):
        registry.log_trial(params={"p": i}, sharpe=sharpe, n_obs=100, run_id="r1")
    assert registry.n_trials() == 3
    assert registry.sharpe_variance() > 0
    assert registry.best_trial()["sharpe"] == 1.2

    jsonl = tmp_path / "worker_0.jsonl"
    for sharpe in (0.9, 1.1):
        TrialsRegistry.write_jsonl_record(
            jsonl, {"params": {"worker": 0}, "sharpe": sharpe, "n_obs": 50}
        )
    imported = registry.import_jsonl(jsonl)
    assert imported == 2
    assert registry.n_trials() == 5
    registry.close()


def test_dsr_properties():
    # More trials -> higher bar -> lower DSR for the same observed Sharpe.
    dsr_few = deflated_sharpe_ratio(1.0, n_trials=2, trial_variance=0.05, n_obs=250)
    dsr_many = deflated_sharpe_ratio(1.0, n_trials=200, trial_variance=0.05, n_obs=250)
    assert 0.0 <= dsr_many < dsr_few <= 1.0
    assert expected_max_sharpe(200, 0.05) > expected_max_sharpe(2, 0.05)


def test_dsr_from_registry():
    registry = TrialsRegistry()
    for sharpe in (0.4, 0.9, 0.6):
        registry.log_trial(params={}, sharpe=sharpe, n_obs=100)
    value = dsr_from_registry(registry, sr_hat=0.9, n_obs=100)
    assert 0.0 <= value <= 1.0
    registry.close()


def test_tearsheet_creates_file(tmp_path):
    equity = _equity_from_log_returns(np.tile([0.002, -0.001], 40))
    path = tmp_path / "sheet.png"
    stats = generate_tearsheet(equity, path, extra_stats={"DSR": "0.91"})
    assert path.exists() and path.stat().st_size > 0
    assert stats["sharpe"] > 0
