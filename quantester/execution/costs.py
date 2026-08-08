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

    def __post_init__(self):
        for name in (
            "fixed_commission",
            "per_share_commission",
            "spread_pct",
            "slippage_vol_coef",
            "impact_coef",
        ):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(
                    f"CostModel.{name} must be >= 0 (costs cannot be negative). "
                    f"Got {value!r}."
                )
            setattr(self, name, value)
        if self.spread_pct >= 0.05:
            raise ValueError(
                f"CostModel.spread_pct={self.spread_pct!r} looks like a percentage "
                "instead of a fraction. Use 0.0005 for 5 bps, not 0.05 or 5."
            )

    def commission(self, quantity: float, price: float | None = None) -> float:
        """c_t: proportional + fixed exchange/clearing/regulatory costs.

        `price` is the fill reference price; the base model is price-independent
        (per-share schedule), but notional-based models (e.g. bps-of-notional
        exchange fees) require it.
        """
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


@dataclass
class ConservativeFrictionCostModel(CostModel):
    """Stressed exchange friction: C_trade = 2 * (S_bid-ask / 2 + mu_fee).

    Verification status: not covered by the notebook — implemented from the
    user's strategy specification (two times the standard bid-ask spread
    penalty plus explicit maker/taker fees, both doubled as a conservatism
    stress for prop-evaluation / small-account risk rules).

    - Adverse adjustment (phi_t): friction_multiplier * half-spread, i.e. the
      FULL bid-ask spread at the default 2x multiplier. Kaufman range slippage
      and Kyle impact are zeroed here: the spec's friction term subsumes them.
    - Commission (c_t): friction_multiplier * fee_rate * notional, the doubled
      maker/taker fee schedule.

    Applied uniformly to every fill (market and resting limit alike): charging
    taker-grade friction on post-only maker fills is deliberately pessimistic.
    Deterministic in (order, bar) so event engine and MC fast-track stay in
    parity.
    """

    fee_rate: float = 0.0004          # mu_fee: per-side fee as fraction of notional
    friction_multiplier: float = 2.0  # the spec's 2x conservatism stress

    def adverse_adjustment(self, price: float, quantity: float, bar) -> float:
        return self.friction_multiplier * self.half_spread(price)

    def commission(self, quantity: float, price: float | None = None) -> float:
        if quantity <= 0:
            return 0.0
        if price is None or price <= 0:
            raise ValueError(
                "ConservativeFrictionCostModel.commission requires the fill "
                "reference price to charge notional-based fees."
            )
        return self.friction_multiplier * self.fee_rate * quantity * price


@dataclass
class RetailCostModel(CostModel):
    """OHLCV-only retail execution cost model (no Level-2 / order book).

    Decomposition of the per-share adverse adjustment:

        fill = reference ± (spread + volatility_slippage + participation_impact)

    where:

    - ``spread_bps``: full bid-ask assumption in basis points (instrument-
      specific; do not hard-code one universal spread).
    - ``volatility_slippage_factor × range_bps``: Kaufman-style uncertainty
      from the bar's high-low range.
    - ``impact_factor × volatility_bps × participation ** impact_exponent``:
      nonlinear participation-aware market impact. Tiny orders against deep
      volume incur negligible impact; large orders against thin bars do not.

    ``max_participation_rate`` is a liquidity constraint (not a cost): the
    execution simulator clips fill quantity to
    ``bar_volume × max_participation_rate`` and may partially fill / reject /
    split across subsequent bars.

    Verification status: not covered by the notebook — implemented from the
    Quantester retail-execution hardening specification (OHLCV friction with
    configurable spread / vol slippage / square-root-style impact).

    Deterministic in (order, bar) so the event engine and MC fast-track stay
    in parity when the same model instance is shared.
    """

    spread_bps: float = 5.0
    volatility_slippage_factor: float = 0.1
    impact_factor: float = 0.1
    impact_exponent: float = 0.5
    max_participation_rate: float = 0.05
    # Keep legacy CostModel knobs inert so accidental inheritance of defaults
    # cannot double-count spread/slippage/impact under the retail formulas.
    fixed_commission: float = 0.0
    per_share_commission: float = 0.0
    spread_pct: float = 0.0
    slippage_vol_coef: float = 0.0
    impact_coef: float = 0.0

    def __post_init__(self) -> None:
        if self.spread_bps < 0:
            raise ValueError("spread_bps must be non-negative")
        if self.volatility_slippage_factor < 0:
            raise ValueError("volatility_slippage_factor must be non-negative")
        if self.impact_factor < 0:
            raise ValueError("impact_factor must be non-negative")
        if not 0.0 < self.impact_exponent <= 2.0:
            raise ValueError("impact_exponent must lie in (0, 2]")
        if not 0.0 < self.max_participation_rate <= 1.0:
            raise ValueError("max_participation_rate must lie in (0, 1]")

    @staticmethod
    def _bps_to_fraction(bps: float) -> float:
        return float(bps) / 10_000.0

    def half_spread(self, price: float) -> float:
        return price * self._bps_to_fraction(self.spread_bps) / 2.0

    def volatility_bps(self, price: float, bar_high: float, bar_low: float) -> float:
        if price <= 0:
            return 0.0
        return max(bar_high - bar_low, 0.0) / price * 10_000.0

    def volatility_slippage(self, price: float, bar_high: float, bar_low: float) -> float:
        vol_bps = self.volatility_bps(price, bar_high, bar_low)
        return price * self._bps_to_fraction(
            self.volatility_slippage_factor * vol_bps
        )

    def participation(self, quantity: float, bar_volume: float) -> float:
        if bar_volume <= 0 or quantity <= 0:
            return 0.0
        return float(quantity) / float(bar_volume)

    def participation_impact(
        self, price: float, quantity: float, bar_volume: float,
        bar_high: float, bar_low: float,
    ) -> float:
        if price <= 0 or quantity <= 0 or bar_volume <= 0:
            return 0.0
        vol_bps = self.volatility_bps(price, bar_high, bar_low)
        part = self.participation(quantity, bar_volume)
        impact_bps = (
            self.impact_factor * vol_bps * (part ** self.impact_exponent)
        )
        return price * self._bps_to_fraction(impact_bps)

    def max_fill_quantity(self, bar_volume: float) -> float:
        return max(float(bar_volume), 0.0) * self.max_participation_rate

    def cost_components(self, price: float, quantity: float, bar) -> dict:
        """Breakdown of the per-share adverse adjustment (for diagnostics)."""
        high = float(bar["high"])
        low = float(bar["low"])
        volume = float(bar["volume"])
        spread = self.half_spread(price)
        slip = self.volatility_slippage(price, high, low)
        impact = self.participation_impact(price, quantity, volume, high, low)
        return {
            "half_spread": spread,
            "volatility_slippage": slip,
            "participation_impact": impact,
            "participation": self.participation(quantity, volume),
            "volatility_bps": self.volatility_bps(price, high, low),
            "total_adjustment": spread + slip + impact,
        }

    def adverse_adjustment(self, price: float, quantity: float, bar) -> float:
        return self.cost_components(price, quantity, bar)["total_adjustment"]

    def kaufman_slippage(self, price: float, bar_high: float, bar_low: float) -> float:
        return self.volatility_slippage(price, bar_high, bar_low)

    def kyle_lambda(self, price: float, quantity: float, bar_volume: float,
                    bar_high: float, bar_low: float) -> float:
        return self.participation_impact(
            price, quantity, bar_volume, bar_high, bar_low
        )


# ---------------------------------------------------------------------------
# Cost-stress scenarios (BASE / CONSERVATIVE / STRESS)
# ---------------------------------------------------------------------------

def retail_cost_scenario(name: str = "BASE", **overrides) -> RetailCostModel:
    """Named retail friction presets for cost-stress testing.

    Assumptions are configurable modelling choices, not universal truths.
    Stress levels are severe-but-plausible — not arbitrary giant costs.
    """
    presets = {
        "BASE": dict(
            spread_bps=5.0,
            volatility_slippage_factor=0.10,
            impact_factor=0.10,
            impact_exponent=0.5,
            max_participation_rate=0.05,
        ),
        "CONSERVATIVE": dict(
            spread_bps=10.0,
            volatility_slippage_factor=0.20,
            impact_factor=0.20,
            impact_exponent=0.6,
            max_participation_rate=0.03,
        ),
        "STRESS": dict(
            spread_bps=25.0,
            volatility_slippage_factor=0.40,
            impact_factor=0.40,
            impact_exponent=0.7,
            max_participation_rate=0.01,
        ),
    }
    key = name.upper()
    if key not in presets:
        raise ValueError(
            f"Unknown retail cost scenario {name!r}; "
            f"expected one of {sorted(presets)}"
        )
    params = {**presets[key], **overrides}
    return RetailCostModel(**params)
