import numpy as np
import pandas as pd
import pytest

from quantester.execution.costs import CostModel
from quantester.utils.synthetic import make_synthetic_ohlcv


@pytest.fixture
def ohlc():
    return make_synthetic_ohlcv("AAA", n_bars=120, seed=11)


@pytest.fixture
def ohlc_with_missing():
    complete = make_synthetic_ohlcv("AAA", n_bars=120, seed=11)
    gappy = make_synthetic_ohlcv("BBB", n_bars=120, seed=22, missing_every=17)
    return {"AAA": complete, "BBB": gappy}


@pytest.fixture
def zero_costs():
    return CostModel(
        fixed_commission=0.0,
        per_share_commission=0.0,
        spread_pct=0.0,
        slippage_vol_coef=0.0,
        impact_coef=0.0,
    )


@pytest.fixture
def toy_returns():
    rng = np.random.default_rng(5)
    return rng.normal(0.0005, 0.01, size=400)
