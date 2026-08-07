"""Synthetic path modeling (MC Report section 5).

- Ornstein-Uhlenbeck process for mean-reverting synthetic prices:
      dP_t = theta (mu - P_{t-1}) dt + sigma dW_t
  Parameters {theta, mu, sigma} estimated from history via OLS of dP on P_{t-1}.
  Generating 100,000-path ensembles and sweeping stop-loss/take-profit grids
  across them yields Optimal Trading Rules (OTR) calibrated over the whole
  stochastic space rather than one realized path.
- Correlated Gaussian multi-asset return generator with common/idiosyncratic
  shock injection (section 5.1; the HRP/CLA/IVP allocator bake-off is out of
  scope for this build).
- Stationary block bootstrap over OHLCV bars (Politis-Romano): resamples
  contiguous blocks of bars with geometric random lengths, preserving each
  bar's intra-bar OHLC shape and the short-range serial correlation that iid
  shuffling destroys, while scrambling the long-run regime sequence. This is
  the MC null for path-dependent strategies that have no closed-form
  vectorized twin (the event engine re-runs on each synthetic path).
  Verification status: protocol endorsed by the notebook cross-reference
  (Masters' stationary/tapered block bootstrap; de Prado's sequential
  bootstrap alternative noted); implemented from the canonical
  Politis-Romano (1994) stationary bootstrap — the OHLC shape-preserving
  reconstruction is not covered by the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OUParams:
    theta: float
    mu: float
    sigma: float


def estimate_ou_params(prices, dt: float = 1.0) -> OUParams:
    """OLS of dP_t on P_{t-1}: dP = a + b P + eps; theta=-b/dt, mu=-a/b,
    sigma=std(eps)/sqrt(dt). A non-mean-reverting fit (b >= 0) falls back to
    theta=0 (random walk around the sample mean)."""
    p = np.asarray(prices, dtype=float)
    dp = np.diff(p)
    b, a = np.polyfit(p[:-1], dp, 1)
    resid = dp - (a + b * p[:-1])
    sigma = float(resid.std(ddof=1) / np.sqrt(dt))
    if b >= 0:
        return OUParams(theta=0.0, mu=float(p.mean()), sigma=sigma)
    return OUParams(theta=float(-b / dt), mu=float(-a / b), sigma=sigma)


def generate_ou_paths(params: OUParams, p0: float, n_steps: int, n_paths: int,
                      dt: float = 1.0, seed: int | None = None) -> np.ndarray:
    """Euler-Maruyama simulation; returns (n_paths, n_steps + 1)."""
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = p0
    z = rng.standard_normal((n_paths, n_steps))
    for t in range(n_steps):
        drift = params.theta * (params.mu - paths[:, t]) * dt
        diffusion = params.sigma * np.sqrt(dt) * z[:, t]
        paths[:, t + 1] = paths[:, t] + drift + diffusion
    return paths


def correlated_gaussian_returns(n_assets: int, n_obs: int, cov: np.ndarray | None = None,
                                common_shock_scale: float = 0.0,
                                idio_shock_scale: float = 0.0,
                                n_common_shocks: int = 5,
                                n_idio_shocks: int = 20,
                                seed: int | None = None) -> np.ndarray:
    """(n_obs, n_assets) correlated Gaussian returns with injected shocks
    replicating fat tails and regime shifts (MC Report section 5.1)."""
    rng = np.random.default_rng(seed)
    if cov is None:
        a = rng.standard_normal((n_assets, n_assets))
        cov = a @ a.T / n_assets + np.eye(n_assets)
    chol = np.linalg.cholesky(cov)
    returns = rng.standard_normal((n_obs, n_assets)) @ chol.T

    if common_shock_scale > 0:
        times = rng.integers(0, n_obs, size=n_common_shocks)
        signs = rng.choice([-1.0, 1.0], size=n_common_shocks)
        for t, s in zip(times, signs):
            returns[t] += s * common_shock_scale
    if idio_shock_scale > 0:
        times = rng.integers(0, n_obs, size=n_idio_shocks)
        assets = rng.integers(0, n_assets, size=n_idio_shocks)
        signs = rng.choice([-1.0, 1.0], size=n_idio_shocks)
        returns[times, assets] += signs * idio_shock_scale
    return returns


def _stationary_bootstrap_indices(n: int, mean_block: float, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap index sequence of length n:
    start uniform; at each step jump to a fresh uniform start with
    probability 1/mean_block, else continue to the next bar (wrap-around)."""
    if n < 2:
        raise ValueError("need at least 2 bars to bootstrap")
    if mean_block < 1.0:
        raise ValueError("mean_block must be >= 1 (1 = iid shuffle)")
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(0, n)
    jump = rng.random(n - 1) < (1.0 / mean_block)
    fresh = rng.integers(0, n, size=n - 1)
    for t in range(1, n):
        idx[t] = fresh[t - 1] if jump[t - 1] else (idx[t - 1] + 1) % n
    return idx


def bootstrap_ohlcv(frame: pd.DataFrame, mean_block: float = 20.0,
                    seed: int | None = None) -> pd.DataFrame:
    """Stationary block bootstrap of an OHLCV frame, shape-preserving.

    Each original bar j contributes its close-to-close return, open gap
    (open_j / close_{j-1}), intra-bar up/down wick fractions and volume; the
    synthetic bar rebuilds OHLC multiplicatively off the running synthetic
    close, so high >= max(open, close) and low <= min(open, close) hold by
    construction. Blocks of consecutive bars are resampled jointly (mean
    length `mean_block`), preserving short-range dependence and volatility
    clustering inside blocks while destroying the long-run regime ordering.
    Deterministic under a fixed seed; the index/calendar is reused as-is.
    """
    df = frame[["open", "high", "low", "close", "volume"]].astype(float)
    n = len(df)
    close = df["close"].to_numpy()
    open_ = df["open"].to_numpy()
    prev_close = np.concatenate([[close[0]], close[:-1]])
    ret = close / prev_close - 1.0
    ret[0] = 0.0
    gap = open_ / prev_close
    up = df["high"].to_numpy() / np.maximum(open_, close)   # >= 1
    dn = df["low"].to_numpy() / np.minimum(open_, close)    # <= 1
    volume = df["volume"].to_numpy()

    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_indices(n, mean_block, rng)

    s_open = np.empty(n)
    s_close = np.empty(n)
    c_prev = close[0]
    for t in range(n):
        j = idx[t]
        o = c_prev * gap[j]
        c = c_prev * (1.0 + ret[j])
        s_open[t] = o
        s_close[t] = c
        c_prev = c
    oc_max = np.maximum(s_open, s_close)
    oc_min = np.minimum(s_open, s_close)
    out = pd.DataFrame(
        {
            "open": s_open,
            "high": oc_max * up[idx],
            "low": oc_min * dn[idx],
            "close": s_close,
            "volume": volume[idx],
        },
        index=frame.index,
    )
    return out


def otr_sweep(paths: np.ndarray, stop_losses, take_profits,
              entry_price: float | None = None) -> pd.DataFrame:
    """Optimal Trading Rules sweep: mean long PnL per (stop, take-profit) cell.

    Entry at each path's first price (or `entry_price`); exit at the first
    barrier touched (stop checked first on ties), else at the path's end.
    """
    paths = np.asarray(paths, dtype=float)
    entry = float(paths[0, 0]) if entry_price is None else float(entry_price)
    rows = []
    for sl in stop_losses:
        for tp in take_profits:
            stop_level = entry * (1 - sl)
            tp_level = entry * (1 + tp)
            pnls = np.empty(paths.shape[0])
            for i, path in enumerate(paths):
                exit_price = path[-1]
                for price in path[1:]:
                    if price <= stop_level:
                        exit_price = stop_level
                        break
                    if price >= tp_level:
                        exit_price = tp_level
                        break
                pnls[i] = exit_price - entry
            rows.append({"stop_loss": sl, "take_profit": tp,
                         "mean_pnl": float(pnls.mean()),
                         "win_rate": float((pnls > 0).mean())})
    return pd.DataFrame(rows)
