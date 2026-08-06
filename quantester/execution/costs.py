"""Transaction cost models (Report 1 section 2.4).

All models are deterministic functions of the order and the bar, so the event
engine and the Monte Carlo vectorized fast-track share identical cost semantics
(parity is testable).

- Commissions: fixed + per-share variable.
- Spread crossing: taking immediate liquidity costs half the bid-ask spread.
- Kaufman volatility-adjusted slippage: scales with the bar's range
  ((high - low) / close). The exact formula was not covered by the user's
  notebook; implemented as the standard volatility-proportional form.
- Kyle's lambda market impact as dp = lambda * dx ONLY (Cross-Ref section 1.C):
  noise-trader flow dy is unobservable from historical bars; simulating it would
  inject non-deterministic noise and break reproducibility. Lambda is an
  Amihud-style illiquidity coefficient from bar volume/volatility (estimation
  method not covered by the notebook; design choice flagged).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    fixed_commission: float = 1.0        # currency per order
    per_share_commission: float = 0.005  # currency per share
    spread_pct: float = 0.0005           # full bid-ask spread as fraction of price
    slippage_vol_coef: float = 0.1       # Kaufman coefficient on bar range pct
    impact_coef: float = 0.1             # Kyle lambda scale (Amihud-style)

    def commission(self, quantity: float) -> float:
        """c_t: proportional + fixed exchange/clearing/regulatory costs."""
        if quantity <= 0:
            return 0.0
        return self.fixed_commission + self.per_share_commission * quantity

    def half_spread(self, price: float) -> float:
        """Penalty for crossing the bid-ask spread to take liquidity."""
        return price * self.spread_pct / 2.0

    def kaufman_slippage(self, price: float, bar_high: float, bar_low: float) -> float:
        """Volatility-adjusted slippage from the bar's range (deterministic)."""
        if price <= 0:
            return 0.0
        range_pct = max(bar_high - bar_low, 0.0) / price
        return price * self.slippage_vol_coef * range_pct

    def kyle_lambda(self, price: float, quantity: float, bar_volume: float,
                    bar_high: float, bar_low: float) -> float:
        """Market impact dp = lambda * dx, lambda from volume/volatility.

        lambda = impact_coef * volatility_pct / volume (illiquidity rises with
        volatility and falls with depth); dx is the trader's own order size.
        """
        if bar_volume <= 0 or price <= 0:
            return 0.0
        volatility_pct = max(bar_high - bar_low, 0.0) / price
        lam = self.impact_coef * volatility_pct / bar_volume
        return price * lam * quantity

    def adverse_adjustment(self, price: float, quantity: float, bar) -> float:
        """Total per-share adverse price adjustment for a liquidity-taking fill."""
        return (
            self.half_spread(price)
            + self.kaufman_slippage(price, float(bar["high"]), float(bar["low"]))
            + self.kyle_lambda(price, quantity, float(bar["volume"]),
                               float(bar["high"]), float(bar["low"]))
        )
