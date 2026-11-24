"""Implied volatility smile per expiry, and each quote's distance from it.

This is relative value, not arbitrage: a quote far from a fitted smile can stay there. It is
reported in volatility points so it reads the way a vol trader would read it.

Implied vols come from mid prices with Black-76 on the parity-implied forward and discount
factor, so they don't depend on a guessed rate or dividend. American chains are approximate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from optarb.forwards import fit_forward
from optarb.models import OptionsChain, Right
from optarb.pricing import black76_implied_vol


@dataclass(frozen=True)
class SmilePoint:
    expiration: date
    strike: float
    right: Right
    log_moneyness: float
    iv: float
    fitted_iv: float

    @property
    def residual_vol_points(self) -> float:
        return 100.0 * (self.iv - self.fitted_iv)


def smile(chain: OptionsChain, expiration: date, otm_only: bool = True) -> list[SmilePoint]:
    """Quadratic fit of IV in log-moneyness ``ln(K / F)``, using out-of-the-money options by default."""
    fwd = fit_forward(chain, expiration)
    if fwd is None:
        return []
    F, df, T = fwd.forward, fwd.discount_factor, fwd.T
    rows = []
    for right in (Right.CALL, Right.PUT):
        for q in chain.side(expiration, right):
            if q.bid <= 0 or q.ask <= 0:
                continue
            if otm_only and ((right is Right.CALL and q.strike < F) or (right is Right.PUT and q.strike > F)):
                continue
            iv = black76_implied_vol(q.mid, F, q.strike, T, df, right)
            if iv is not None:
                rows.append((q.strike, right, math.log(q.strike / F), iv))
    if len(rows) < 4:
        return []
    k = np.array([r[2] for r in rows])
    iv = np.array([r[3] for r in rows])
    coef = np.polyfit(k, iv, 2)
    fitted = np.polyval(coef, k)
    return [SmilePoint(expiration, s, r, km, v, float(f)) for (s, r, km, v), f in zip(rows, fitted, strict=True)]
