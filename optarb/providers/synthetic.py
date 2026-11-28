"""Seeded synthetic option chains for demos and tests. Every chain it makes is labeled ``synthetic``.

Prices come from a model with a realistic skew and bid/ask spreads that widen in the wings and
for short-dated options. European chains use Black-Scholes-Merton; American chains use the CRR
binomial tree, so a clean chain is free of model-free arbitrage at executable prices.
``plant`` adds known violations so tests (and the demo) can show the detectors working.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import numpy as np

from optarb.models import ExerciseStyle, OptionQuote, OptionsChain, Right, Underlying
from optarb.pricing import bsm_price, crr_american_price


def _skew_vol(base: float, moneyness: float, T: float) -> float:
    """Downward-sloping smile in log-moneyness, flatter for longer expiries."""
    k = math.log(moneyness)
    slope = -0.35 / math.sqrt(max(T, 1 / 52))
    return max(0.05, base + slope * k * 0.25 + 0.6 * k * k)


def _half_spread(price: float, moneyness: float, T: float, tick: float) -> float:
    rel = 0.005 + 0.02 * abs(math.log(moneyness)) / 0.1 + 0.002 / math.sqrt(max(T, 1 / 365))
    return max(tick, min(price * rel, 0.03 + 0.003 * price))


def _round(x: float, tick: float, up: bool) -> float:
    n = x / tick
    return round((math.ceil(n - 1e-9) if up else math.floor(n + 1e-9)) * tick, 2)


def make_chain(
    symbol: str = "XSP",
    spot: float = 100.0,
    rate: float = 0.045,
    dividend_yield: float = 0.012,
    base_vol: float = 0.18,
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN,
    asof: datetime | None = None,
    expiries_days: tuple[int, ...] = (14, 35, 63),
    strike_step: float = 1.0,
    n_strikes: int = 21,
    seed: int = 7,
    tick: float = 0.01,
    crr_steps: int = 200,
) -> OptionsChain:
    rng = np.random.default_rng(seed)
    asof = asof or datetime(2026, 11, 20, 15, 0, tzinfo=UTC)
    start = asof.date()
    center = round(spot / strike_step) * strike_step
    strikes = [center + strike_step * (i - n_strikes // 2) for i in range(n_strikes)]
    probe = OptionsChain(Underlying(symbol, spot, spot), asof, exercise_style)
    quotes: list[OptionQuote] = []
    for days in expiries_days:
        exp = start + timedelta(days=days)
        T = probe.year_fraction(exp)
        for K in strikes:
            m = K / spot
            sigma = _skew_vol(base_vol, m, T)
            for right in (Right.CALL, Right.PUT):
                if exercise_style is ExerciseStyle.EUROPEAN:
                    fair = bsm_price(spot, K, T, rate, sigma, right, dividend_yield)
                else:
                    fair = crr_american_price(spot, K, T, rate, sigma, right, dividend_yield, steps=crr_steps)
                half = _half_spread(fair, m, T, tick)
                bid = _round(max(fair - half, 0.0), tick, up=False)
                ask = _round(fair + half, tick, up=True)
                if ask <= bid:
                    ask = round(bid + tick, 2)
                oi = int(rng.integers(50, 5000) * math.exp(-8 * abs(math.log(m))))
                quotes.append(
                    OptionQuote(
                        strike=K,
                        expiration=exp,
                        right=right,
                        bid=bid,
                        ask=ask,
                        last=round(fair, 2),
                        volume=int(oi * rng.uniform(0.02, 0.3)),
                        open_interest=oi,
                        vendor_iv=round(sigma, 4),
                        contract_symbol=f"{symbol}{exp:%y%m%d}{right.value}{int(K * 1000):08d}",
                    )
                )
    stock_half = max(0.01, round(spot * 0.0001, 2))
    underlying = Underlying(symbol, round(spot - stock_half, 2), round(spot + stock_half, 2), spot, dividend_yield)
    return OptionsChain(underlying, asof, exercise_style, quotes, source="synthetic")


def _find(chain: OptionsChain, exp: date, right: Right, strike: float) -> int:
    for i, q in enumerate(chain.quotes):
        if q.expiration == exp and q.right is right and q.strike == strike:
            return i
    raise KeyError((exp, right, strike))


def _set(chain: OptionsChain, exp: date, right: Right, strike: float, **changes) -> None:
    i = _find(chain, exp, right, strike)
    chain.quotes[i] = replace(chain.quotes[i], **changes)


def plant_butterfly(
    chain: OptionsChain,
    exp: date | None = None,
    right: Right = Right.CALL,
    k: float | None = None,
    excess: float = 0.30,
) -> tuple[date, float]:
    """Raise the body of a butterfly so selling two bodies against the wings collects a credit."""
    exp = exp or chain.expirations[0]
    side = chain.side(exp, right)
    i = len(side) // 2 if k is None else [q.strike for q in side].index(k)
    w1, body, w3 = side[i - 1], side[i], side[i + 1]
    target_bid = round(0.5 * (w1.ask + w3.ask) + excess, 2)
    _set(chain, exp, right, body.strike, bid=target_bid, ask=round(target_bid + body.spread, 2))
    return exp, body.strike


def plant_short_box(chain: OptionsChain, exp: date | None = None, excess: float = 0.25) -> tuple[date, float, float]:
    """Make a European box sellable above its width by richening the low-strike call."""
    exp = exp or chain.expirations[-1]
    pairs = chain.pairs(exp)
    (c1, p1), (c2, p2) = pairs[len(pairs) // 2 - 2], pairs[len(pairs) // 2 + 2]
    width = c2.strike - c1.strike
    needed = width - (-c2.ask + p2.bid - p1.ask) + excess  # c1.bid that makes the short box credit width + excess
    _set(chain, exp, Right.CALL, c1.strike, bid=round(needed, 2), ask=round(needed + c1.spread, 2))
    return exp, c1.strike, c2.strike


def plant_vertical(chain: OptionsChain, exp: date | None = None, excess: float = 0.20) -> tuple[date, float, float]:
    """Make a put vertical sellable for more than its strike width."""
    exp = exp or chain.expirations[0]
    side = chain.side(exp, Right.PUT)
    lo, hi = side[len(side) // 2], side[len(side) // 2 + 1]
    bid = round(lo.ask + (hi.strike - lo.strike) + excess, 2)
    _set(chain, exp, Right.PUT, hi.strike, bid=bid, ask=round(bid + hi.spread, 2))
    return exp, lo.strike, hi.strike


def sample_chain(seed: int = 7, planted: bool = True) -> OptionsChain:
    """The demo snapshot: a synthetic XSP-like European chain with three planted violations."""
    chain = make_chain(seed=seed)
    if planted:
        plant_butterfly(chain)
        plant_short_box(chain)
        plant_vertical(chain, exp=chain.expirations[1])
    return chain
