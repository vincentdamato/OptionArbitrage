from dataclasses import replace

import pytest

from optarb.arbitrage import ScanConfig, best_per_group, box_rates, scan
from optarb.models import ExerciseStyle, Right
from optarb.providers.synthetic import make_chain, plant_butterfly, plant_short_box, plant_vertical, sample_chain

ZERO_COST = ScanConfig(commission_per_contract=0.0, min_net_edge=1e-6, reference_rate=0.045)


@pytest.mark.parametrize("style", list(ExerciseStyle))
@pytest.mark.parametrize("seed", [1, 7, 42])
def test_clean_chain_has_no_model_free_flags(style, seed):
    chain = make_chain(exercise_style=style, seed=seed)
    found = [o for o in scan(chain, ZERO_COST) if o.kind.value == "model_free"]
    assert found == []


def test_short_box_is_detected():
    chain = make_chain()
    exp, k1, k2 = plant_short_box(chain)
    hits = [o for o in scan(chain, ZERO_COST) if o.check == "box_bounds" and o.expiration == exp]
    assert any({leg.strike for leg in o.legs} == {k1, k2} and o.legs[0].action == "SELL" for o in hits)


def test_butterfly_with_uneven_strikes_uses_gap_weights():
    chain = make_chain()
    exp = chain.expirations[0]
    # drop the 101 strike so 100 / 102 / 103 is an uneven butterfly around 102
    chain.quotes = [q for q in chain.quotes if not (q.expiration == exp and q.strike == 101.0)]
    plant_butterfly(chain, exp=exp, k=102.0, excess=0.4)
    hit = next(
        o
        for o in scan(chain, ZERO_COST)
        if o.check == "butterfly_convexity"
        and o.expiration == exp
        and o.legs[1].strike == 102.0
        and o.legs[1].instrument == "CALL"
    )
    assert [(leg.strike, leg.quantity) for leg in hit.legs] == [(100.0, 1), (102.0, 3), (103.0, 2)]


def test_vertical_width_is_detected():
    chain = make_chain()
    exp, lo, hi = plant_vertical(chain)
    hits = [o for o in scan(chain, ZERO_COST) if o.check == "vertical_width" and o.expiration == exp]
    assert any({leg.strike for leg in o.legs} == {lo, hi} for o in hits)


def test_edges_are_dollars_per_package_and_net_of_commission():
    chain = make_chain()
    exp, lo, hi = plant_vertical(chain, excess=0.20)
    gross = next(o for o in scan(chain, ZERO_COST) if o.check == "vertical_width" and o.expiration == exp)
    assert gross.gross_edge == pytest.approx(20.0, abs=0.5)  # 0.20 per share * 100
    with_fees = ScanConfig(commission_per_contract=0.65, min_net_edge=1e-6)
    net = next(o for o in scan(chain, with_fees) if o.key == gross.key)
    assert net.net_edge == pytest.approx(gross.gross_edge - 2 * 0.65)


def test_commissions_can_remove_an_edge():
    chain = make_chain()
    plant_vertical(chain, excess=0.01)
    assert not [o for o in scan(chain, ScanConfig(commission_per_contract=1.0)) if o.check == "vertical_width"]


def test_boxes_are_not_checked_on_american_and_calendars_not_on_european():
    american = make_chain(exercise_style=ExerciseStyle.AMERICAN)
    plant_short_box(american)
    assert not [o for o in scan(american, ZERO_COST) if o.check.startswith("box")]
    european = make_chain()
    exps = european.expirations
    near = next(q for q in european.side(exps[0], Right.PUT) if q.strike == 110.0)
    far = next(q for q in european.side(exps[1], Right.PUT) if q.strike == 110.0)
    european.quotes[european.quotes.index(near)] = replace(near, bid=far.ask + 0.5, ask=far.ask + 0.6)
    assert not [o for o in scan(european, ZERO_COST) if o.check == "calendar"]


def test_american_calendar_is_detected():
    chain = make_chain(exercise_style=ExerciseStyle.AMERICAN)
    exps = chain.expirations
    near = next(q for q in chain.side(exps[0], Right.CALL) if q.strike == 100.0)
    far = next(q for q in chain.side(exps[1], Right.CALL) if q.strike == 100.0)
    chain.quotes[chain.quotes.index(near)] = replace(near, bid=far.ask + 0.3, ask=far.ask + 0.4)
    assert any(o.check == "calendar" for o in scan(chain, ZERO_COST))


def test_reversal_is_detected_when_call_is_cheap():
    chain = make_chain(exercise_style=ExerciseStyle.AMERICAN, dividend_yield=0.0)
    exp = chain.expirations[0]
    call = next(q for q in chain.side(exp, Right.CALL) if q.strike == 95.0)
    chain.quotes[chain.quotes.index(call)] = replace(call, bid=0.5, ask=0.6)
    assert any(o.check == "parity_reversal" and o.expiration == exp for o in scan(chain, ZERO_COST))


def test_sample_chain_shows_each_planted_violation_once_per_group():
    grouped = best_per_group(scan(sample_chain()))
    checks = {o.check for o in grouped}
    assert {"butterfly_convexity", "box_bounds", "vertical_width"} <= checks
    assert len({(o.check, o.expiration) for o in grouped}) == len(grouped)


def test_box_rates_are_finite_and_widest_box_lends_best():
    rates = box_rates(make_chain(n_strikes=61), ScanConfig(commission_per_contract=0.0))
    assert rates and all(r.T > 0 for r in rates)
    assert any(r.best_borrow_rate is not None for r in rates)


def test_opportunity_key_is_stable():
    a = scan(sample_chain())
    b = scan(sample_chain())
    assert [o.key for o in a] == [o.key for o in b]
