"""Strategy signal logic and meta-labeling scaffold."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from quantester.events import LONG, SHORT, SignalEvent
from quantester.strategy.examples import (
    BuyAndHoldStrategy,
    MovingAverageCrossStrategy,
    crossover_positions,
)
from quantester.strategy.meta_labeling import (
    MetaLabelingStrategy,
    triple_barrier_labels,
    zscore_size_transform,
)


def test_crossover_positions_known_sequence():
    close = pd.Series(
        [10, 10, 10, 10, 10, 11, 12, 13, 14, 13, 12, 11, 10, 9, 8, 7, 6],
        index=pd.bdate_range("2024-01-01", periods=17),
        dtype=float,
    )
    positions = crossover_positions(close, fast=2, slow=4)
    assert (positions.iloc[:5] == 0).all()          # not enough data
    assert positions.iloc[5:].isin([1.0, -1.0]).all()
    assert positions.iloc[5] == 1.0                  # first cross up
    assert positions.iloc[-1] == -1.0                # ends short after decline
    assert (positions.diff().fillna(0) != 0).sum() == 2  # exactly two crosses


def test_ma_cross_event_matches_vectorized(ohlc):
    from quantester.data.csv_handler import HistoricCSVDataHandler

    handler = HistoricCSVDataHandler({"AAA": ohlc})
    strategy = MovingAverageCrossStrategy(handler, "AAA", fast=3, slow=8)
    handler.prime_data()
    emitted = []
    while handler.continue_backtest:
        ts, bars = handler.advance()
        handler.set_phase("close", ts)
        if bars["AAA"] is not None:

            class _Q:
                def put(self, item):
                    emitted.append(item)

            strategy.calculate_signals(type("E", (), {"timestamp": ts, "bars": bars})(), _Q())

    twin = strategy.vectorized_signals({"AAA": ohlc})["AAA"]
    for signal in emitted:
        expected = twin.loc[: signal.timestamp].iloc[-1]
        if signal.signal_type == LONG:
            assert expected == 1.0
        elif signal.signal_type == SHORT:
            assert expected == -1.0
        else:
            assert expected == 0.0


def test_buy_and_hold_emits_once(ohlc):
    from quantester.data.csv_handler import HistoricCSVDataHandler

    handler = HistoricCSVDataHandler({"AAA": ohlc})
    strategy = BuyAndHoldStrategy(handler)
    handler.prime_data()
    ts, bars = handler.advance()
    handler.set_phase("close", ts)
    emitted = []

    class _Q:
        def put(self, item):
            emitted.append(item)

    event = type("E", (), {"timestamp": ts, "bars": bars})()
    strategy.calculate_signals(event, _Q())
    strategy.calculate_signals(event, _Q())
    assert len(emitted) == 1
    assert emitted[0].signal_type == LONG


def test_triple_barrier_labels():
    idx = pd.bdate_range("2024-01-01", periods=30)
    rising = pd.Series(np.linspace(100, 120, 30), index=idx)
    falling = pd.Series(np.linspace(100, 80, 30), index=idx)
    events = pd.DataFrame({"t0": [idx[0]], "side": [1]})
    assert triple_barrier_labels(rising, events, 0.05, 0.05, 20).iloc[0] == 1
    assert triple_barrier_labels(falling, events, 0.05, 0.05, 20).iloc[0] == 0
    short_events = pd.DataFrame({"t0": [idx[0]], "side": [-1]})
    assert triple_barrier_labels(falling, short_events, 0.05, 0.05, 20).iloc[0] == 1


def test_meta_labeling_scales_strength(ohlc):
    from quantester.data.csv_handler import HistoricCSVDataHandler

    handler = HistoricCSVDataHandler({"AAA": ohlc})
    primary = BuyAndHoldStrategy(handler)

    X = pd.DataFrame({"f": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0, 0, 1, 1])
    model = LogisticRegression().fit(X, y)

    wrapper = MetaLabelingStrategy(primary, handler, model=model,
                                   feature_fn=lambda dh, s: [3.0])
    handler.prime_data()
    ts, bars = handler.advance()
    handler.set_phase("close", ts)
    emitted = []

    class _Q:
        def put(self, item):
            emitted.append(item)

    wrapper.calculate_signals(type("E", (), {"timestamp": ts, "bars": bars})(), _Q())
    assert len(emitted) == 1
    expected_prob = float(model.predict_proba([[3.0]])[0][1])
    assert emitted[0].strength == pytest.approx(expected_prob)

    passthrough = MetaLabelingStrategy(BuyAndHoldStrategy(handler), handler)
    emitted2 = []

    class _Q2:
        def put(self, item):
            emitted2.append(item)

    passthrough.calculate_signals(type("E", (), {"timestamp": ts, "bars": bars})(), _Q2())
    assert emitted2[0].strength == 1.0


def test_zscore_transform_bounds():
    assert -1.0 < zscore_size_transform(0.7) < 1.0
    assert zscore_size_transform(0.5) == pytest.approx(0.0)
    assert zscore_size_transform(0.0) == 0.0
