"""Regenerate ``schwab_spy_chain.json``: a Schwab ``/chains`` payload built from a model, not from market data.

It has the shape and field names of a real Schwab response so the parser, the Greeks and the whole
record -> scan -> study path can be tested without credentials. It deliberately includes the awkward
parts of real payloads: -999 and "NaN" placeholders, a non-standard (adjusted) contract, quote sizes,
and a few stale quote times. Prices come from a CRR tree; the "vendor" Greeks come from Black-Scholes,
the way a broker's quick Greeks would, so they differ a little from the tree's on in-the-money puts.

Run: python tests/fixtures/make_schwab_fixture.py
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

from optarb.models import OptionsChain, Right, Underlying
from optarb.pricing import bsm_greeks, crr_american_price

ASOF = datetime(2026, 11, 20, 15, 30, tzinfo=UTC)
SPOT, RATE, DIV = 595.40, 0.0425, 0.012
EXPIRIES = (ASOF.date() + timedelta(days=14), ASOF.date() + timedelta(days=49))
STRIKES = [575.0 + 5 * i for i in range(9)]


def vol(K: float, T: float) -> float:
    k = math.log(K / SPOT)
    return 0.16 - 0.45 * k + 1.2 * k * k + 0.01 / math.sqrt(T * 52)


def contract(right: Right, K: float, exp, T: float, i: int) -> dict:
    sigma = vol(K, T)
    fair = crr_american_price(SPOT, K, T, RATE, sigma, right, DIV, steps=200)
    half = max(0.01, round(0.004 * fair + 0.02, 2))
    g = bsm_greeks(SPOT, K, T, RATE, sigma, right, DIV)
    stale = i % 7 == 3
    quote_time = ASOF - timedelta(minutes=45 if stale else 0, seconds=i)
    code = "C" if right is Right.CALL else "P"
    return {
        "putCall": "CALL" if right is Right.CALL else "PUT",
        "symbol": f"SPY   {exp:%y%m%d}{code}{int(K * 1000):08d}",
        "bid": round(fair - half, 2),
        "ask": round(fair + half, 2),
        "last": round(fair, 2),
        "mark": round(fair, 2),
        "bidSize": 10 + 7 * (i % 5),
        "askSize": 12 + 5 * (i % 4),
        "totalVolume": 100 + 37 * i,
        "openInterest": 1000 + 211 * i,
        "quoteTimeInLong": int(quote_time.timestamp() * 1000),
        "tradeTimeInLong": int(quote_time.timestamp() * 1000),
        "volatility": round(100 * sigma, 3),
        "delta": round(float(g.delta), 3),
        "gamma": round(float(g.gamma), 4),
        "theta": round(float(g.theta), 3),
        "vega": round(float(g.vega), 3),
        "rho": round(float(g.rho), 3),
        "strikePrice": K,
        "multiplier": 100.0,
        "nonStandard": False,
        "mini": False,
        "exerciseType": "A",
        "daysToExpiration": (exp - ASOF.date()).days,
    }


def build() -> dict:
    probe = OptionsChain(Underlying("SPY", SPOT, SPOT), ASOF, None)  # type: ignore[arg-type]
    calls: dict = {}
    puts: dict = {}
    i = 0
    for exp in EXPIRIES:
        T = probe.year_fraction(exp)
        key = f"{exp.isoformat()}:{(exp - ASOF.date()).days}"
        calls[key], puts[key] = {}, {}
        for K in STRIKES:
            calls[key][f"{K:.1f}"] = [contract(Right.CALL, K, exp, T, i)]
            puts[key][f"{K:.1f}"] = [contract(Right.PUT, K, exp, T, i + 1)]
            i += 2
    first = next(iter(calls))
    # placeholders the parser has to survive
    calls[first]["575.0"][0]["volatility"] = -999.0
    calls[first]["575.0"][0]["delta"] = "NaN"
    # an adjusted contract with a non-standard deliverable, which must be dropped
    adj = dict(
        calls[first]["595.0"][0], symbol="SPY1  adjusted", nonStandard=True, multiplier=100.0, bid=1.0, ask=900.0
    )
    calls[first]["595.0"].append(adj)
    return {
        "symbol": "SPY",
        "status": "SUCCESS",
        "underlying": {"symbol": "SPY", "bid": round(SPOT - 0.01, 2), "ask": round(SPOT + 0.01, 2), "last": SPOT},
        "strategy": "SINGLE",
        "isDelayed": False,
        "isIndex": False,
        "interestRate": 100 * RATE,
        "underlyingPrice": SPOT,
        "volatility": 29.0,
        "numberOfContracts": sum(len(v) for m in (calls, puts) for s in m.values() for v in s.values()),
        "callExpDateMap": calls,
        "putExpDateMap": puts,
    }


if __name__ == "__main__":
    out = Path(__file__).with_name("schwab_spy_chain.json")
    out.write_text(json.dumps(build(), indent=1))
    print(out)
