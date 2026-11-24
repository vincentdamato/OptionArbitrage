import pytest

from optarb.forwards import fit_all_forwards
from optarb.providers.synthetic import make_chain


def test_forward_fit_recovers_carry_on_a_european_chain():
    chain = make_chain(rate=0.045, dividend_yield=0.012, n_strikes=41)
    fits = fit_all_forwards(chain)
    assert len(fits) == len(chain.expirations)
    longest = fits[chain.expirations[-1]]
    assert longest.implied_rate == pytest.approx(0.045, abs=0.01)
    assert longest.implied_carry(chain.underlying.mid) == pytest.approx(0.033, abs=0.003)
    assert not longest.approximate


def test_forward_fit_needs_enough_strikes():
    chain = make_chain(n_strikes=3)
    assert fit_all_forwards(chain) == {}


def test_forward_fit_ignores_a_mispriced_strike():
    from optarb.providers.synthetic import plant_short_box

    clean = make_chain(n_strikes=41)
    dirty = make_chain(n_strikes=41)
    exp, _, _ = plant_short_box(dirty)
    a, b = fit_all_forwards(clean)[exp], fit_all_forwards(dirty)[exp]
    assert b.forward == pytest.approx(a.forward, abs=0.02)
    assert b.n_strikes < a.n_strikes
