from .base import Strategy
from .donchian_breakout import DonchianBreakoutStrategy
from .ewmac_carry import EWMACCarryStrategy
from .examples import BuyAndHoldStrategy, MovingAverageCrossStrategy
from .pairs_trading import PairsTradingStrategy
from .tranche_pullback import TranchePullbackStrategy

__all__ = [
    "Strategy",
    "BuyAndHoldStrategy",
    "MovingAverageCrossStrategy",
    "PairsTradingStrategy",
    "TranchePullbackStrategy",
    "DonchianBreakoutStrategy",
    "EWMACCarryStrategy",
]
