"""IV and Greeks from a snapshot's own quotes, checked against the source's Greeks and against BSM."""

import json
import math
from pathlib import Path

import pytest

from optarb.arbitrage import ScanConfig, fresh, scan
from optarb.greeks import analyze, expiry_inputs, surface_summary, vendor_comparison
from optarb.models import ExerciseStyle, Right
from optarb.pricing import bsm_greeks, bsm_price, crr_american_greeks, crr_american_price, crr_implied_vol
from optarb.providers.schwab import parse_chain
from optarb.providers.synthetic import sample_chain

FIXTURE = Path(__file__).parent / "fixtures" / "schwab_spy_chain.json"


@pytest.fixture(scope="module")
def chain():
    from datetime import UTC, datetime

    payload = json.loads(FIXTURE.read_text())
    return parse_chain(payload, symbol="SPY", asof=datetime(2026, 11, 20, 15, 30, tzinfo=UTC), dividend_yield=0.012)


@pytest.fixture(scope="module")
def table(chain):
    return analyze(chain)


def test_fixture_parses_like_a_real_payload(chain):
    assert chain.exercise_style is ExerciseStyle.AMERICAN
    assert chain.vendor_rate == pytest.approx(0.0425)
    assert all(q.ask < 900 for q in chain.quotes)  # the non-standard (adjusted) contract was dropped
    assert len(chain.quotes) == 36
    bad = next(q for q in chain.quotes if q.right is Right.CALL and q.strike == 575)
    assert bad.vendor_iv is None and bad.vendor_delta is None  # -999 and "NaN" become missing
    assert all(q.bid_size > 0 and q.ask_size > 0 and q.quote_time is not None for q in chain.quotes)


def test_rate_comes_from_the_source_for_american_chains(chain):
    inputs = expiry_inputs(chain)
    assert {i.rate_source for i in inputs.values()} == {"vendor"}
    assert expiry_inputs(chain, rate=0.05)[chain.expirations[0]].rate_source == "caller"


def test_european_chain_takes_its_rate_from_parity():
    inputs = expiry_inputs(sample_chain())
    assert {i.rate_source for i in inputs.values()} == {"parity"}


def test_every_quote_gets_an_iv(table, chain):
    assert len(table) == len(chain.quotes)
    assert table["iv"].notna().all()
    assert set(table["model"]) == {"crr"}


def test_our_greeks_agree_with_the_vendors(table):
    vc = vendor_comparison(table).set_index("measure")
    assert vc.loc["iv", "median_abs_gap"] < 0.002  # under 0.2 vol points
    for name in ("delta", "gamma", "theta", "vega"):
        assert vc.loc[name, "median_ratio"] == pytest.approx(1.0, abs=0.03), name
    # vendor Greeks in the fixture are European (BSM); American put rho is smaller, so allow more room
    assert vc.loc["rho", "median_ratio"] == pytest.approx(1.0, abs=0.1)


def test_unit_mismatch_shows_up_in_the_ratio(table):
    scaled = table.assign(vendor_vega=table["vendor_vega"] * 100)
    ratio = vendor_comparison(scaled).set_index("measure").loc["vega", "median_ratio"]
    assert ratio == pytest.approx(0.01, rel=0.05)


def test_surface_has_atm_and_skew(table):
    s = surface_summary(table).set_index("expiration")
    first = s.iloc[0]
    assert 10 < first["atm_iv"] < 30
    assert first["rr25"] < 0  # puts over calls, as built into the fixture
    assert s["quotes"].min() > 0


def test_crr_greeks_match_bsm_for_a_call_without_dividends():
    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.04, 0.25
    tree = crr_american_greeks(S, K, T, r, sigma, Right.CALL, 0.0, steps=800)
    bs = bsm_greeks(S, K, T, r, sigma, Right.CALL, 0.0)
    assert tree.delta == pytest.approx(bs.delta, abs=2e-3)
    assert tree.gamma == pytest.approx(bs.gamma, rel=0.02)
    assert tree.theta == pytest.approx(bs.theta, rel=0.03)
    assert tree.vega == pytest.approx(bs.vega, rel=0.02)
    assert tree.rho == pytest.approx(bs.rho, rel=0.03)
    assert math.isnan(tree.vanna)


def test_american_put_is_worth_more_and_delta_stays_in_bounds():
    S, K, T, r, sigma = 100.0, 110.0, 1.0, 0.06, 0.2
    assert crr_american_price(S, K, T, r, sigma, Right.PUT) > bsm_price(S, K, T, r, sigma, Right.PUT)
    g = crr_american_greeks(S, K, T, r, sigma, Right.PUT)
    assert -1.0 <= g.delta < 0 and g.gamma > 0


@pytest.mark.parametrize("right", [Right.CALL, Right.PUT])
def test_crr_implied_vol_round_trips(right):
    S, K, T, r, q, sigma = 595.0, 600.0, 30 / 365, 0.0425, 0.012, 0.18
    price = crr_american_price(S, K, T, r, sigma, right, q, steps=150)
    assert crr_implied_vol(price, S, K, T, r, right, q, steps=150) == pytest.approx(sigma, abs=1e-4)


def test_stale_quotes_are_dropped_before_scanning(chain):
    assert len(fresh(chain, ScanConfig()).quotes) == 36
    kept = fresh(chain, ScanConfig(max_quote_age_seconds=300)).quotes
    assert 0 < len(kept) < 36
    assert all((chain.asof - q.quote_time).total_seconds() <= 300 for q in kept)
    assert scan(chain, ScanConfig(max_quote_age_seconds=300)) == []


def test_min_size_blocks_thin_quotes():
    from optarb.arbitrage import _usable

    q = sample_chain().quotes[0]
    thin = q.__class__(**{**q.__dict__, "bid_size": 2, "ask_size": 2})
    assert _usable(thin, "BUY", ScanConfig(min_size=5)) is False
    assert _usable(thin, "BUY", ScanConfig(min_size=2)) is True
    unknown = q.__class__(**{**q.__dict__, "bid_size": 0, "ask_size": 0})  # size not reported: not filtered
    assert _usable(unknown, "BUY", ScanConfig(min_size=5)) is True
