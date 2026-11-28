"""Implied volatility and Greeks for every quote in a snapshot, computed from the snapshot itself.

European chains (SPX, XSP) are priced with Black-76 on the forward and discount factor implied by
put-call parity at that expiry, so no rate or dividend has to be assumed. Greeks are then taken
with respect to spot, using the rate and carry those imply.

American chains (equities, ETFs) are priced on a CRR tree. Parity doesn't pin down the rate for
American options, so the rate comes from the caller, or else the rate the data source reported,
and the dividend yield comes from the underlying.

When the source reports its own IV and Greeks (Schwab does), ``vendor_comparison`` shows how far
apart they are. A ratio near 100 or 0.01 on any Greek means a unit mismatch, not a pricing gap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from optarb.forwards import fit_forward
from optarb.models import ExerciseStyle, OptionsChain, Right
from optarb.pricing import (
    Greeks,
    black76_implied_vol,
    bsm_greeks,
    crr_american_greeks,
    crr_implied_vol,
)

DEFAULT_RATE = 0.045
GREEKS = ("delta", "gamma", "theta", "vega", "rho")


@dataclass(frozen=True)
class Inputs:
    """What the IV and Greeks for one expiry were computed from."""

    T: float
    rate: float
    carry_yield: float  # dividend yield q, or the parity-implied equivalent
    forward: float
    discount_factor: float
    rate_source: str  # "parity", "caller", "vendor" or "default"


def expiry_inputs(chain: OptionsChain, rate: float | None = None) -> dict:
    """Per-expiry rate, carry and forward, with where the rate came from."""
    spot = chain.underlying.mid
    out = {}
    for exp in chain.expirations:
        T = chain.year_fraction(exp)
        if T <= 0:
            continue
        fit = fit_forward(chain, exp) if chain.exercise_style is ExerciseStyle.EUROPEAN else None
        if fit is not None:
            r = fit.implied_rate
            q = r - fit.implied_carry(spot)
            out[exp] = Inputs(T, r, q, fit.forward, fit.discount_factor, "parity")
            continue
        if rate is not None:
            r, src = rate, "caller"
        elif chain.vendor_rate is not None:
            r, src = chain.vendor_rate, "vendor"
        else:
            r, src = DEFAULT_RATE, "default"
        q = chain.underlying.dividend_yield
        out[exp] = Inputs(T, r, q, spot * math.exp((r - q) * T), math.exp(-r * T), src)
    return out


def analyze(
    chain: OptionsChain,
    rate: float | None = None,
    otm_only: bool = False,
    spread_iv: bool | None = None,
    tree_steps: int = 150,
) -> pd.DataFrame:
    """One row per two-sided quote: mid, bid and ask IV, our Greeks, and the vendor's where reported.

    ``spread_iv`` (IV at the bid and at the ask) defaults to on for European chains, where it is cheap,
    and off for American chains, where each IV is a tree solve.
    """
    american = chain.exercise_style is ExerciseStyle.AMERICAN
    spread_iv = (not american) if spread_iv is None else spread_iv
    inputs = expiry_inputs(chain, rate)
    S = chain.underlying.mid
    rows = []
    for q in chain.quotes:
        inp = inputs.get(q.expiration)
        if inp is None or q.bid <= 0 or q.ask <= 0 or q.ask < q.bid:
            continue
        if otm_only and (
            (q.right is Right.CALL and q.strike < inp.forward) or (q.right is Right.PUT and q.strike > inp.forward)
        ):
            continue

        def iv_of(price: float, right=q.right, K=q.strike, inp=inp) -> float | None:
            if american:
                return crr_implied_vol(price, S, K, inp.T, inp.rate, right, inp.carry_yield, tree_steps)
            return black76_implied_vol(price, inp.forward, K, inp.T, inp.discount_factor, right)

        iv = iv_of(q.mid)
        g: Greeks | None = None
        if iv is not None:
            if american:
                g = crr_american_greeks(S, q.strike, inp.T, inp.rate, iv, q.right, inp.carry_yield, tree_steps)
            else:
                g = bsm_greeks(S, q.strike, inp.T, inp.rate, iv, q.right, inp.carry_yield)
        row = {
            "expiration": q.expiration,
            "strike": q.strike,
            "right": q.right.value,
            "bid": q.bid,
            "ask": q.ask,
            "mid": q.mid,
            "T": inp.T,
            "forward": inp.forward,
            "rate": inp.rate,
            "rate_source": inp.rate_source,
            "model": "crr" if american else "black76",
            "iv": iv,
            "iv_bid": iv_of(q.bid) if spread_iv else None,
            "iv_ask": iv_of(q.ask) if spread_iv else None,
            "vendor_iv": q.vendor_iv,
        }
        for name in GREEKS:
            row[name] = getattr(g, name) if g is not None else None
            row[f"vendor_{name}"] = getattr(q, f"vendor_{name}")
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["expiration", "right", "strike"], ignore_index=True)
    return df


def vendor_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """For IV and each Greek: how many quotes both sides priced, the median and 90th percentile absolute gap,
    and the median ratio of ours to the vendor's (to catch unit mismatches)."""
    rows = []
    for name in ("iv", *GREEKS):
        vendor = f"vendor_{name}"
        if df.empty or vendor not in df:
            continue
        both = df[[name, vendor]].apply(pd.to_numeric, errors="coerce").dropna()
        if both.empty:
            continue
        gap = (both[name] - both[vendor]).abs()
        nonzero = both[both[vendor].abs() > 1e-9]
        ratio = (nonzero[name] / nonzero[vendor]).median() if not nonzero.empty else np.nan
        rows.append(
            {
                "measure": name,
                "quotes": len(both),
                "median_abs_gap": gap.median(),
                "p90_abs_gap": gap.quantile(0.9),
                "median_ratio": ratio,
            }
        )
    return pd.DataFrame(rows, columns=["measure", "quotes", "median_abs_gap", "p90_abs_gap", "median_ratio"])


def surface_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per expiry: at-the-money IV, 25-delta put and call IV, risk reversal and butterfly, in vol points.

    Built from out-of-the-money quotes only. ATM is interpolated at the forward; the 25-delta vols are
    interpolated in delta. A blank means the expiry has no quotes on that side of the delta.
    """
    cols = ["expiration", "T", "atm_iv", "put25_iv", "call25_iv", "rr25", "bf25", "quotes"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for exp, g in df.dropna(subset=["iv", "delta"]).groupby("expiration"):
        F = g["forward"].iloc[0]
        calls = g[(g["right"] == "C") & (g["strike"] >= F)].sort_values("strike")
        puts = g[(g["right"] == "P") & (g["strike"] <= F)].sort_values("strike")
        otm = pd.concat([puts, calls]).sort_values("strike")
        if len(otm) < 2:
            continue
        atm = float(np.interp(F, otm["strike"], otm["iv"]))

        def at_delta(side: pd.DataFrame, target: float) -> float | None:
            side = side.sort_values("delta")
            d, v = side["delta"].to_numpy(float), side["iv"].to_numpy(float)
            if len(side) < 2 or not d.min() <= target <= d.max():
                return None
            return float(np.interp(target, d, v))

        p25, c25 = at_delta(puts, -0.25), at_delta(calls, 0.25)
        rr = (c25 - p25) if p25 is not None and c25 is not None else None
        bf = (0.5 * (c25 + p25) - atm) if p25 is not None and c25 is not None else None
        pts = lambda x: None if x is None else 100.0 * x  # noqa: E731
        out.append(
            {
                "expiration": exp,
                "T": g["T"].iloc[0],
                "atm_iv": pts(atm),
                "put25_iv": pts(p25),
                "call25_iv": pts(c25),
                "rr25": pts(rr),
                "bf25": pts(bf),
                "quotes": len(g),
            }
        )
    return pd.DataFrame(out, columns=cols)
