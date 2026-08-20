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


def test_triple_barrier_labels_use_high_low_path():
    """AFML ch.3: first touch is on the intra-bar high/low path, not close."""
    idx = pd.bdate_range("2024-01-01", periods=10)
    close = pd.Series([100.0] * 10, index=idx)
    high = close.copy()
    low = close.copy()
    high.iloc[2] = 110.0  # TP at 105; close never leaves 100
    events = pd.DataFrame({"t0": [idx[0]], "side": [1]})
    assert triple_barrier_labels(close, events, 0.05, 0.05, 8).iloc[0] == 0
    assert triple_barrier_labels(
        close, events, 0.05, 0.05, 8, high=high, low=low,
    ).iloc[0] == 1


def test_triple_barrier_same_bar_both_hit_is_stop():
    """When high and low breach TP and SL on the same bar, label the stop."""
    idx = pd.bdate_range("2024-01-01", periods=6)
    close = pd.Series([100.0] * 6, index=idx)
    high = close.copy()
    low = close.copy()
    high.iloc[1] = 110.0
    low.iloc[1] = 90.0
    events = pd.DataFrame({"t0": [idx[0]], "side": [1]})
    assert triple_barrier_labels(
        close, events, 0.05, 0.05, 4, high=high, low=low,
    ).iloc[0] == 0


# --------------------------------------------------------------------------
# D12 (ticket 28): high/low is the DEFAULT path when OHLC exists
# --------------------------------------------------------------------------


def _ohlc_stop_wick_frame():
    """Long entry: stop-loss wicked intra-bar, close never breaches."""
    idx = pd.bdate_range("2024-01-01", periods=5, tz="UTC")
    close = pd.Series([100.0, 99.5, 99.2, 99.8, 100.3], index=idx)
    return pd.DataFrame(
        {
            "high": close + 0.2,  # tp = 101.1 never touched
            "low": close - 0.5,   # sl = 98.9 wicked at idx[2] (98.7)
            "close": close,
        },
        index=idx,
    )


def test_triple_barrier_auto_path_defaults_to_high_low():
    """path="auto" (the default) uses high/low when an OHLC frame is passed;
    path="close" is the explicit close-only opt-out."""
    frame = _ohlc_stop_wick_frame()
    events = pd.DataFrame({"t0": [frame.index[0]], "side": [1]})
    assert triple_barrier_labels(
        None, events, 0.011, 0.011, 4, ohlc=frame
    ).iloc[0] == 0  # stop wick labels the stop by default
    assert triple_barrier_labels(
        None, events, 0.011, 0.011, 4, ohlc=frame, path="close"
    ).iloc[0] == 1  # close path: vertical barrier, final close above entry
    # No high/low anywhere -> close-only even under auto (legacy callers).
    assert triple_barrier_labels(
        frame["close"], events, 0.011, 0.011, 4
    ).iloc[0] == 1


def test_triple_barrier_path_and_input_validation():
    frame = _ohlc_stop_wick_frame()
    events = pd.DataFrame({"t0": [frame.index[0]], "side": [1]})
    with pytest.raises(ValueError, match="path"):
        triple_barrier_labels(frame["close"], events, 0.011, 0.011, 4,
                              path="wick")
    with pytest.raises(TypeError, match="close"):
        triple_barrier_labels(None, events, 0.011, 0.011, 4)


def test_fit_secondary_uses_ohlc_high_low_without_kwargs():
    """fit_secondary given an OHLC frame labels on the high/low path (D12)."""
    from sklearn.linear_model import LogisticRegression

    from quantester.strategy.meta_labeling import MetaLabelingStrategy

    frame = _ohlc_stop_wick_frame()
    # Same entry bar, both sides: the long gets stop-wicked (0), the short
    # takes profit through the same low wick (1). Close-path labels would be
    # exactly flipped — a crisp contrast.
    events = pd.DataFrame(
        {"t0": [frame.index[0], frame.index[0]], "side": [1, -1]},
        index=["long", "short"],
    )
    features = pd.DataFrame({"f": [0.0, 1.0]}, index=["long", "short"])
    wrapper = MetaLabelingStrategy(
        primary=BuyAndHoldStrategy(None), data_handler=None,
        model=LogisticRegression(),
    )
    y = wrapper.fit_secondary(features, frame, events, 0.011, 0.011, 4)
    assert list(y) == [0, 1]



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
