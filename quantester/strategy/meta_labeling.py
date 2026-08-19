"""Meta-labeling scaffold (Report 2 section 2; AFML ch.3 notebook-verified).

The PRIMARY model decides the trade SIDE (direction). The SECONDARY model is a
binary classifier that predicts the probability the primary model is correct,
and that probability scales the trade SIZE. This filters false positives:
precision rises at the cost of recall.

Labels come from the triple-barrier method: y=1 if the primary signal was
correct (take-profit barrier touched first on the high/low path / positive
realization at the vertical barrier), y=0 otherwise. Same-bar TP and SL
touches label the stop. When OHLC high/low exist, the high/low path is the
DEFAULT (ruling D12, ticket 28); ``path="close"`` is the explicit close-only
opt-out.

The z-score size transform m = 2*Phi(z) - 1 (AFML ch.10) was NOT covered by the
user's notebook; it is offered as an option and flagged as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..events import SignalEvent
from .base import Strategy


def triple_barrier_labels(close: pd.Series | None, events: pd.DataFrame,
                          tp_pct: float, sl_pct: float, max_holding: int,
                          high: pd.Series | None = None,
                          low: pd.Series | None = None, *,
                          path: str = "auto",
                          ohlc: pd.DataFrame | None = None) -> pd.Series:
    """Binary meta-labels for primary-model events.

    events: DataFrame with columns [t0 (entry timestamp), side (+1/-1)].
    Returns Series of y in {0, 1} indexed like events: 1 if the primary signal
    was correct under the triple-barrier outcome.

    First touch walks the intra-bar high/low path (AFML ch. 3): long TP if
    high ≥ tp, long SL if low ≤ sl; short is the opposite. When both barriers
    are touched on the same bar, the label is the stop (y=0). The
    vertical-barrier terminal still uses close.

    Path selection (ruling D12): ``path="auto"`` (default) labels on
    high/low whenever both are available — via the ``high``/``low`` series
    or an ``ohlc`` DataFrame (whose close/high/low replace the positional
    ``close``). ``path="close"`` forces the legacy close-only labels even
    when high/low are passed. Callers holding only closes keep their labels
    unchanged (high/low fall back to close).

    Notebook-verified: AFML ch. 3 triple-barrier intent. High/low first-touch
    is AFML ch. 3 path dependence — previously "not covered beyond intent";
    the default flip itself is notebook ruling D12.
    """
    if path not in ("auto", "close"):
        raise ValueError(f"path must be 'auto' or 'close'; got {path!r}")
    if ohlc is not None:
        close = ohlc["close"]
        if "high" in ohlc.columns and "low" in ohlc.columns:
            high, low = ohlc["high"], ohlc["low"]
    if close is None:
        raise TypeError(
            "triple_barrier_labels needs a close Series (positional) or an "
            "ohlc DataFrame with a 'close' column."
        )
    if path == "close" or high is None or low is None:
        high = close
        low = close
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

    def fit_secondary(self, features: pd.DataFrame, bars,
                      events: pd.DataFrame, tp_pct: float, sl_pct: float,
                      max_holding: int, high: pd.Series | None = None,
                      low: pd.Series | None = None, *, path: str = "auto"):
        """Build triple-barrier labels for the primary events and fit the model.

        ``bars`` is an OHLC DataFrame (high/low path by default, D12) or a
        bare close Series (close-only legacy labeling; ``high``/``low``
        kwargs aligned with it opt into the path). ``path="close"`` forces
        close-only labels either way.
        """
        if self.model is None:
            raise ValueError("No secondary model configured.")
        if isinstance(bars, pd.DataFrame):
            y = triple_barrier_labels(
                None, events, tp_pct, sl_pct, max_holding, path=path, ohlc=bars,
            )
        else:
            y = triple_barrier_labels(
                bars, events, tp_pct, sl_pct, max_holding,
                high=high, low=low, path=path,
            )
        self.model.fit(features.loc[y.index], y)
        return y

    def vectorized_signals(self, data: dict):
        return self.primary.vectorized_signals(data)
