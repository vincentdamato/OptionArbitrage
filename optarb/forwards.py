"""Implied forward and discount factor per expiry, backed out of put-call parity.

For European options, parity says ``C(K) - P(K) = DF * (F - K)`` at every strike, so a
straight-line fit of mid ``C - P`` against ``K`` gives slope ``-DF`` and intercept ``DF * F``.
That replaces a hard-coded risk-free rate and a guessed dividend yield with what the market
itself implies. For American options early exercise bends the line, so the result is an
approximation and is labeled as such.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from optarb.models import ExerciseStyle, OptionsChain


@dataclass(frozen=True)
class ImpliedForward:
    expiration: date
    T: float
    discount_factor: float
    forward: float
    n_strikes: int
    approximate: bool  # True for American options

    @property
    def implied_rate(self) -> float:
        """Continuously compounded rate implied by the discount factor."""
        return -math.log(self.discount_factor) / self.T

    def implied_carry(self, spot: float) -> float:
        """Annualized r - q implied by the forward relative to spot."""
        return math.log(self.forward / spot) / self.T


def _wls(K: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([np.ones_like(K), K])
    sw = np.sqrt(w)
    (intercept, slope), *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return float(intercept), float(slope)


def fit_forward(
    chain: OptionsChain, expiration: date, moneyness_band: float = 0.10, min_strikes: int = 4
) -> ImpliedForward | None:
    """Weighted least-squares fit of parity on strikes within ``moneyness_band`` of spot.

    Each strike is weighted by the inverse of its combined bid/ask spread, so tight markets dominate,
    and strikes far off the line (a stale or mispriced quote) are dropped before the final fit.
    Returns None when there are too few two-sided pairs or the fit is not economically sensible.
    """
    T = chain.year_fraction(expiration)
    spot = chain.underlying.mid
    if T <= 0 or spot <= 0:
        return None

    rows = []
    for call, put in chain.pairs(expiration):
        if min(call.bid, put.bid) <= 0:
            continue
        if abs(call.strike / spot - 1.0) > moneyness_band:
            continue
        rows.append((call.strike, call.mid - put.mid, call.spread + put.spread))
    if len(rows) < min_strikes:
        return None

    K, y, spread = (np.array(col, dtype=float) for col in zip(*rows, strict=True))
    w = 1.0 / np.maximum(spread, 0.01)
    keep = np.ones_like(K, dtype=bool)
    for _ in range(2):  # fit, drop strikes more than 4 robust sigmas off the line, refit
        intercept, slope = _wls(K[keep], y[keep], w[keep])
        resid = y - (intercept + slope * K)
        mad = np.median(np.abs(resid[keep] - np.median(resid[keep])))
        new_keep = np.abs(resid) <= max(4.0 * 1.4826 * mad, 0.02)
        if new_keep.sum() < min_strikes or (new_keep == keep).all():
            break
        keep = new_keep
    intercept, slope = _wls(K[keep], y[keep], w[keep])
    df = -slope
    if not 0.5 < df <= 1.05:
        return None
    return ImpliedForward(
        expiration=expiration,
        T=T,
        discount_factor=float(df),
        forward=float(intercept / df),
        n_strikes=int(keep.sum()),
        approximate=chain.exercise_style is ExerciseStyle.AMERICAN,
    )


def fit_all_forwards(chain: OptionsChain, **kwargs) -> dict[date, ImpliedForward]:
    fits = {exp: fit_forward(chain, exp, **kwargs) for exp in chain.expirations}
    return {exp: fit for exp, fit in fits.items() if fit is not None}
