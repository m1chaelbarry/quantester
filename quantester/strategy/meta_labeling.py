"""Meta-labeling scaffold (Report 2 section 2; AFML ch.3 notebook-verified).

The PRIMARY model decides the trade SIDE (direction). The SECONDARY model is a
binary classifier that predicts the probability the primary model is correct,
and that probability scales the trade SIZE. This filters false positives:
precision rises at the cost of recall.

Labels come from the triple-barrier method: y=1 if the primary signal was
correct (take-profit barrier touched first on the high/low path / positive
realization at the vertical barrier), y=0 otherwise. Same-bar TP and SL
touches label the stop. Close-only callers omit high/low and recover the
legacy path.

The z-score size transform m = 2*Phi(z) - 1 (AFML ch.10) was NOT covered by the
user's notebook; it is offered as an option and flagged as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..events import SignalEvent
from .base import Strategy


def triple_barrier_labels(close: pd.Series, events: pd.DataFrame,
                          tp_pct: float, sl_pct: float, max_holding: int,
                          high: pd.Series | None = None,
                          low: pd.Series | None = None) -> pd.Series:
    """Binary meta-labels for primary-model events.

    events: DataFrame with columns [t0 (entry timestamp), side (+1/-1)].
    Returns Series of y in {0, 1} indexed like events: 1 if the primary signal
    was correct under the triple-barrier outcome.

    First touch walks the intra-bar high/low path (AFML ch. 3): long TP if
    high ≥ tp, long SL if low ≤ sl; short is the opposite. When both barriers
    are touched on the same bar, the label is the stop (y=0). ``high``/``low``
    default to ``close`` so existing close-only callers keep their labels.
    The vertical-barrier terminal still uses close.

    Notebook-verified: AFML ch. 3 triple-barrier intent (close-path labels).
    High/low first-touch is AFML ch. 3 path dependence — not covered by a
    notebook page beyond that intent; implemented from AFML ch. 3.
    """
    high = close if high is None else high
    low = close if low is None else low
    labels = {}
    idx = close.index
    pos_of = {ts: i for i, ts in enumerate(idx)}
    for event_id, row in events.iterrows():
        i0 = pos_of[row["t0"]]
        side = float(row["side"])
        entry = float(close.iloc[i0])
        tp = entry * (1 + side * tp_pct)
        sl = entry * (1 - side * sl_pct)
        end = min(i0 + max_holding, len(idx) - 1)
        path_high = high.iloc[i0 + 1 : end + 1]
        path_low = low.iloc[i0 + 1 : end + 1]
        path_close = close.iloc[i0 + 1 : end + 1]
        if side > 0:
            hit_tp = path_high >= tp
            hit_sl = path_low <= sl
        else:
            hit_tp = path_low <= tp
            hit_sl = path_high >= sl
        y = 0
        if hit_tp.any() or hit_sl.any():
            tp_t = hit_tp.idxmax() if hit_tp.any() else None
            sl_t = hit_sl.idxmax() if hit_sl.any() else None
            if sl_t is not None and (tp_t is None or sl_t <= tp_t):
                y = 0  # stop first, or same-bar both-hit
            else:
                y = 1
        elif len(path_close):
            final = float(path_close.iloc[-1])
            y = 1 if (final - entry) * side > 0 else 0
        labels[event_id] = y
    return pd.Series(labels)


def zscore_size_transform(prob: float) -> float:
    """m = 2*Phi(z) - 1 with z = (p - 0.5) / sqrt(p(1-p)).

    AFML ch.10; NOT covered by the user's notebook -- optional, flagged.
    """
    if not 0.0 < prob < 1.0:
        return 0.0
    z = (prob - 0.5) / np.sqrt(prob * (1 - prob))
    return float(2 * norm.cdf(z) - 1)


class MetaLabelingStrategy(Strategy):
    """Wraps a primary Strategy; secondary model probability scales signal strength.

    `model` is any sklearn-style estimator with fit/predict_proba. When model is
    None the wrapper passes primary signals through unchanged (strength 1).
    """

    def __init__(self, primary: Strategy, data_handler, model=None,
                 feature_fn=None, size_transform: str = "linear"):
        if size_transform not in ("linear", "zscore"):
            raise ValueError("size_transform must be 'linear' or 'zscore'")
        self.primary = primary
        self.data_handler = data_handler
        self.model = model
        self.feature_fn = feature_fn
        self.size_transform = size_transform
        self.delay = primary.delay

    def matches_phase(self, phase: str) -> bool:
        return self.primary.matches_phase(phase)

    def calculate_signals(self, event, events_queue):
        class _Interceptor:
            def __init__(self, outer):
                self.outer = outer
                self.captured = []

            def put(self, signal):
                self.captured.append(signal)

        interceptor = _Interceptor(self)
        self.primary.calculate_signals(event, interceptor)
        for signal in interceptor.captured:
            strength = signal.strength
            if self.model is not None and self.feature_fn is not None:
                features = self.feature_fn(self.data_handler, signal)
                names = getattr(self.model, "feature_names_in_", None)
                if names is not None:
                    features = pd.DataFrame([features], columns=names)
                    prob = float(self.model.predict_proba(features)[0][1])
                else:
                    prob = float(self.model.predict_proba([features])[0][1])
                if self.size_transform == "zscore":
                    # Confidence only — never invert declared side via negative strength.
                    strength = max(0.0, zscore_size_transform(prob))
                else:
                    strength = prob
            events_queue.put(
                SignalEvent(
                    signal.timestamp,
                    signal.symbol,
                    signal.signal_type,
                    strength=strength,
                    delay=signal.delay,
                    fill_at=signal.fill_at,
                    limit_price=getattr(signal, "limit_price", None),
                    cancel_orders=getattr(signal, "cancel_orders", False),
                    stop_distance=getattr(signal, "stop_distance", None),
                )
            )

    def fit_secondary(self, features: pd.DataFrame, close: pd.Series,
                      events: pd.DataFrame, tp_pct: float, sl_pct: float,
                      max_holding: int):
        """Build triple-barrier labels for the primary events and fit the model."""
        if self.model is None:
            raise ValueError("No secondary model configured.")
        y = triple_barrier_labels(close, events, tp_pct, sl_pct, max_holding)
        self.model.fit(features.loc[y.index], y)
        return y

    def vectorized_signals(self, data: dict):
        return self.primary.vectorized_signals(data)
