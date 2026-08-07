from .base import Strategy
from .examples import BuyAndHoldStrategy, MovingAverageCrossStrategy
from .pairs_trading import PairsTradingStrategy
from .tranche_pullback import TranchePullbackStrategy

__all__ = [
    "Strategy",
    "BuyAndHoldStrategy",
    "MovingAverageCrossStrategy",
    "PairsTradingStrategy",
    "TranchePullbackStrategy",
]
