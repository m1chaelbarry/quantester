from .base import Strategy
from .donchian_breakout import DonchianBreakoutStrategy
from .examples import BuyAndHoldStrategy, MovingAverageCrossStrategy
from .letf_dual_ema import LetfDualEmaDeltaStrategy, dual_ema_delta_positions
from .pairs_trading import PairsTradingStrategy
from .tranche_pullback import TranchePullbackStrategy

__all__ = [
    "Strategy",
    "BuyAndHoldStrategy",
    "MovingAverageCrossStrategy",
    "PairsTradingStrategy",
    "TranchePullbackStrategy",
    "DonchianBreakoutStrategy",
    "LetfDualEmaDeltaStrategy",
    "dual_ema_delta_positions",
]
