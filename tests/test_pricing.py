import math

import pytest

from optarb.models import Right
from optarb.pricing import (
    black76_implied_vol,
    black76_price,
    bsm_greeks,
    bsm_price,
    crr_american_price,
    implied_vol,
)

CASES = [
    # S, K, T, r, sigma, q
    (100.0, 105.0, 0.40, 0.045, 0.25, 0.0),
    (100.0, 105.0, 0.40, 0.045, 0.25, 0.03),
    (450.0, 400.0, 0.08, 0.050, 0.18, 0.013),
    (50.0, 60.0, 1.50, 0.020, 0.45, 0.0),
]
RIGHTS = [Right.CALL, Right.PUT]


def _fd_greeks(S, K, T, r, sigma, right, q):
    h = 1e-4

    def price(S=S, T=T, r=r, sigma=sigma):
        return bsm_price(S, K, T, r, sigma, right, q)

    def delta(S=S, T=T, sigma=sigma):
        return bsm_greeks(S, K, T, r, sigma, right, q).delta

    def vega(sigma):
        return bsm_greeks(S, K, T, r, sigma, right, q).vega

    def gamma(S):
        return bsm_greeks(S, K, T, r, sigma, right, q).gamma

    return {
        "delta": (price(S=S + h) - price(S=S - h)) / (2 * h),
        "gamma": (price(S=S + h) - 2 * price() + price(S=S - h)) / h**2,
        "theta": -(price(T=T + h) - price(T=T - h)) / (2 * h) / 365,
        "vega": (price(sigma=sigma + h) - price(sigma=sigma - h)) / (2 * h) / 100,
        "rho": (price(r=r + h) - price(r=r - h)) / (2 * h) / 100,
        "vanna": (delta(sigma=sigma + h) - delta(sigma=sigma - h)) / (2 * h) / 100,
        "charm": -(delta(T=T + h) - delta(T=T - h)) / (2 * h) / 365,
        "vomma": (vega(sigma + h) - vega(sigma - h)) / (2 * h) / 100,
        "speed": (gamma(S + h) - gamma(S - h)) / (2 * h),
    }


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("right", RIGHTS)
def test_greeks_match_finite_differences(case, right):
    S, K, T, r, sigma, q = case
    analytic = bsm_greeks(S, K, T, r, sigma, right, q).to_dict()
    for name, expected in _fd_greeks(S, K, T, r, sigma, right, q).items():
        assert analytic[name] == pytest.approx(expected, rel=1e-3, abs=1e-7), name


@pytest.mark.parametrize("case", CASES)
def test_put_call_parity_holds_for_bsm(case):
    S, K, T, r, sigma, q = case
    call = bsm_price(S, K, T, r, sigma, Right.CALL, q)
    put = bsm_price(S, K, T, r, sigma, Right.PUT, q)
    assert call - put == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-10)


def test_known_value_hull_example():
    # Hull, Options, Futures and Other Derivatives, Example 15.6: S=42, K=40, r=10%, sigma=20%, T=0.5
    assert bsm_price(42, 40, 0.5, 0.10, 0.20, Right.CALL) == pytest.approx(4.76, abs=0.005)
    assert bsm_price(42, 40, 0.5, 0.10, 0.20, Right.PUT) == pytest.approx(0.81, abs=0.005)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("right", RIGHTS)
def test_implied_vol_round_trip(case, right):
    S, K, T, r, sigma, q = case
    price = bsm_price(S, K, T, r, sigma, right, q)
    assert implied_vol(price, S, K, T, r, right, q) == pytest.approx(sigma, abs=1e-7)


def test_implied_vol_rejects_price_below_intrinsic():
    assert implied_vol(1.0, 100, 90, 0.5, 0.05, Right.CALL) is None


def test_black76_matches_bsm_with_forward():
    S, K, T, r, sigma, q = 100, 95, 0.75, 0.04, 0.3, 0.02
    F, df = S * math.exp((r - q) * T), math.exp(-r * T)
    for right in RIGHTS:
        assert black76_price(F, K, T, df, sigma, right) == pytest.approx(bsm_price(S, K, T, r, sigma, right, q))
        price = black76_price(F, K, T, df, sigma, right)
        assert black76_implied_vol(price, F, K, T, df, right) == pytest.approx(sigma, abs=1e-7)


def test_american_put_is_worth_at_least_european_and_intrinsic():
    S, K, T, r, sigma = 90, 100, 1.0, 0.06, 0.25
    american = crr_american_price(S, K, T, r, sigma, Right.PUT)
    european = bsm_price(S, K, T, r, sigma, Right.PUT)
    assert american > european
    assert american >= K - S


def test_american_call_without_dividends_equals_european():
    S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.2
    american = crr_american_price(S, K, T, r, sigma, Right.CALL, steps=800)
    assert american == pytest.approx(bsm_price(S, K, T, r, sigma, Right.CALL), abs=5e-3)
