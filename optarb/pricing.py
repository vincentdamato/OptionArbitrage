"""Option pricing: Black-Scholes-Merton, Black-76, a CRR binomial for American options, and Greeks.

Conventions (all Greeks are per one option on one share):

* ``delta``, ``gamma``: per $1 move in spot.
* ``theta``: per calendar day.
* ``vega``: per 1 volatility point (0.01 in sigma).
* ``rho``: per 1 percentage point in the rate.
* ``vanna``: change in delta per 1 vol point.
* ``charm``: change in delta per calendar day.
* ``vomma``: change in vega (per vol point) per 1 vol point.
* ``speed``: change in gamma per $1 move in spot.

Rates and dividend yields are continuously compounded and annualized. Every analytic
Greek is checked against finite differences in ``tests/test_pricing.py``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from optarb.models import Right

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _right(right: Right | str) -> Right:
    return right if isinstance(right, Right) else Right(str(right).upper()[0])


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def intrinsic(S: float, K: float, right: Right | str) -> float:
    return max(S - K, 0.0) if _right(right) is Right.CALL else max(K - S, 0.0)


def bsm_price(S: float, K: float, T: float, r: float, sigma: float, right: Right | str, q: float = 0.0) -> float:
    """European option price under Black-Scholes-Merton with a continuous dividend yield."""
    right = _right(right)
    if T <= 0 or sigma <= 0:
        t = max(T, 0.0)
        fwd = S * math.exp(-q * t) - K * math.exp(-r * t)
        return max(fwd, 0.0) if right is Right.CALL else max(-fwd, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    df_q, df_r = math.exp(-q * T), math.exp(-r * T)
    if right is Right.CALL:
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)


def black76_price(F: float, K: float, T: float, df: float, sigma: float, right: Right | str) -> float:
    """European option price on a forward ``F`` with discount factor ``df`` to expiry."""
    right = _right(right)
    if T <= 0 or sigma <= 0:
        return df * (max(F - K, 0.0) if right is Right.CALL else max(K - F, 0.0))
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol_sqrt_t * vol_sqrt_t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    if right is Right.CALL:
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_implied_vol(
    price: float, F: float, K: float, T: float, df: float, right: Right | str, lo: float = 1e-4, hi: float = 5.0
) -> float | None:
    """Implied volatility under Black-76. Returns None when the price is outside the no-arbitrage range."""
    right = _right(right)
    if T <= 0 or price <= 0:
        return None
    lower = df * (max(F - K, 0.0) if right is Right.CALL else max(K - F, 0.0))
    upper = df * (F if right is Right.CALL else K)
    if not lower < price < upper:
        return None

    def objective(sigma: float) -> float:
        return black76_price(F, K, T, df, sigma, right) - price

    if objective(lo) > 0 or objective(hi) < 0:
        return None
    return float(brentq(objective, lo, hi, xtol=1e-10))


def implied_vol(
    price: float, S: float, K: float, T: float, r: float, right: Right | str, q: float = 0.0
) -> float | None:
    """Black-Scholes-Merton implied volatility (European). Returns None outside the no-arbitrage range."""
    return black76_implied_vol(price, S * math.exp((r - q) * T), K, T, math.exp(-r * T), right)


def _crr_tree(
    S: float, K: float, T: float, r: float, sigma: float, right: Right, q: float, steps: int
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    """Backward induction on a CRR tree. Returns (value, values at step 1, values at step 2, dt, u)."""
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("CRR probability outside (0, 1); increase steps")
    j = np.arange(steps + 1)
    spots = S * u ** (steps - j) * d**j
    sign = 1.0 if right is Right.CALL else -1.0
    values = np.maximum(sign * (spots - K), 0.0)
    step1 = step2 = values
    for n in range(steps, 0, -1):
        spots = spots[:-1] * d  # one step back: S u^(n-1-j) d^j
        continuation = disc * (p * values[:-1] + (1.0 - p) * values[1:])
        values = np.maximum(continuation, sign * (spots - K))
        if n == 3:
            step2 = values.copy()
        elif n == 2:
            step1 = values.copy()
    return float(values[0]), step1, step2, dt, u


def crr_american_price(
    S: float, K: float, T: float, r: float, sigma: float, right: Right | str, q: float = 0.0, steps: int = 400
) -> float:
    """American option price on a Cox-Ross-Rubinstein binomial tree with a continuous dividend yield."""
    right = _right(right)
    if T <= 0 or sigma <= 0:
        return intrinsic(S, K, right)
    return _crr_tree(S, K, T, r, sigma, right, q, steps)[0]


def crr_implied_vol(
    price: float, S: float, K: float, T: float, r: float, right: Right | str, q: float = 0.0, steps: int = 150
) -> float | None:
    """Implied volatility of an American option on the CRR tree. None outside the no-arbitrage range."""
    right = _right(right)
    if T <= 0 or price <= 0 or price < intrinsic(S, K, right) + 1e-6:
        return None

    def objective(sigma: float) -> float:
        return crr_american_price(S, K, T, r, sigma, right, q, steps) - price

    lo = max(1e-3, 1.5 * abs(r - q) * math.sqrt(T / steps) + 1e-4)  # keeps the tree probability inside (0, 1)
    hi = 5.0
    try:
        if objective(lo) > 0 or objective(hi) < 0:
            return None
        return float(brentq(objective, lo, hi, xtol=1e-7))
    except ValueError:
        return None


@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    vanna: float
    charm: float
    vomma: float
    speed: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def bsm_greeks(S: float, K: float, T: float, r: float, sigma: float, right: Right | str, q: float = 0.0) -> Greeks:
    """Closed-form Black-Scholes-Merton price and Greeks, in the units listed in the module docstring."""
    right = _right(right)
    if T <= 0 or sigma <= 0 or S <= 0:
        price = bsm_price(S, K, T, r, sigma, right, q)
        itm = (S > K) if right is Right.CALL else (S < K)
        delta = (1.0 if right is Right.CALL else -1.0) if itm else 0.0
        return Greeks(price, delta, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    sqrt_t = math.sqrt(T)
    df_q, df_r = math.exp(-q * T), math.exp(-r * T)
    pdf_d1 = math.exp(-0.5 * d1 * d1) / _SQRT_2PI
    is_call = right is Right.CALL

    price = bsm_price(S, K, T, r, sigma, right, q)
    delta = df_q * norm.cdf(d1) if is_call else df_q * (norm.cdf(d1) - 1.0)
    gamma = df_q * pdf_d1 / (S * sigma * sqrt_t)
    vega_full = S * df_q * pdf_d1 * sqrt_t  # per 1.00 of sigma

    decay = -S * df_q * pdf_d1 * sigma / (2.0 * sqrt_t)
    if is_call:
        theta_year = decay + q * S * df_q * norm.cdf(d1) - r * K * df_r * norm.cdf(d2)
        rho_full = K * T * df_r * norm.cdf(d2)
    else:
        theta_year = decay - q * S * df_q * norm.cdf(-d1) + r * K * df_r * norm.cdf(-d2)
        rho_full = -K * T * df_r * norm.cdf(-d2)

    vanna_full = -df_q * pdf_d1 * d2 / sigma
    charm_common = df_q * pdf_d1 * (2.0 * (r - q) * T - d2 * sigma * sqrt_t) / (2.0 * T * sigma * sqrt_t)
    if is_call:
        charm_year = q * df_q * norm.cdf(d1) - charm_common
    else:
        charm_year = -q * df_q * norm.cdf(-d1) - charm_common
    vomma_full = vega_full * d1 * d2 / sigma
    speed = -gamma / S * (d1 / (sigma * sqrt_t) + 1.0)

    return Greeks(
        price=price,
        delta=delta,
        gamma=gamma,
        theta=theta_year / 365.0,
        vega=vega_full / 100.0,
        rho=rho_full / 100.0,
        vanna=vanna_full / 100.0,
        charm=charm_year / 365.0,
        vomma=vomma_full / 10_000.0,
        speed=speed,
    )


def crr_american_greeks(
    S: float, K: float, T: float, r: float, sigma: float, right: Right | str, q: float = 0.0, steps: int = 300
) -> Greeks:
    """Price and Greeks of an American option on the CRR tree, in the module's units.

    Delta, gamma and theta come from the tree's own early nodes (Hull, chapter 21); vega and rho come
    from central bumps of one vol point and one basis point. Second-order cross Greeks are not computed
    and are NaN.
    """
    right = _right(right)
    if T <= 0 or sigma <= 0 or S <= 0:
        return bsm_greeks(S, K, T, r, sigma, right, q)
    price, v1, v2, dt, u = _crr_tree(S, K, T, r, sigma, right, q, steps)
    d = 1.0 / u
    delta = (v1[0] - v1[1]) / (S * u - S * d)
    delta_up = (v2[0] - v2[1]) / (S * u * u - S)
    delta_dn = (v2[1] - v2[2]) / (S - S * d * d)
    gamma = (delta_up - delta_dn) / (0.5 * (S * u * u - S * d * d))
    theta_year = (v2[1] - price) / (2.0 * dt)
    hv, hr = 0.01, 1e-4
    vega = (
        crr_american_price(S, K, T, r, sigma + hv, right, q, steps)
        - crr_american_price(S, K, T, r, max(sigma - hv, 1e-4), right, q, steps)
    ) / 2.0
    rho = (
        (
            crr_american_price(S, K, T, r + hr, sigma, right, q, steps)
            - crr_american_price(S, K, T, r - hr, sigma, right, q, steps)
        )
        / (2.0 * hr)
        / 100.0
    )
    nan = float("nan")
    return Greeks(price, float(delta), float(gamma), theta_year / 365.0, vega, rho, nan, nan, nan, nan)
