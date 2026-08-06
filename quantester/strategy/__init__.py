from .base import Strategy
from .examples import BuyAndHoldStrategy, MovingAverageCrossStrategy
from .pairs_trading import PairsTradingStrategy

__all__ = [
    "Strategy",
    "BuyAndHoldStrategy",
    "MovingAverageCrossStrategy",
    "PairsTradingStrategy",
]
