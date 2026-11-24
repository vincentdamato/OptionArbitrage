from datetime import UTC, date, datetime

from optarb.models import ExerciseStyle, Right
from optarb.providers.schwab import exercise_style_for, parse_chain

PAYLOAD = {
    "symbol": "SPY",
    "underlying": {"bid": 600.1, "ask": 600.2, "last": 600.15},
    "callExpDateMap": {
        "2026-12-18:24": {
            "600.0": [
                {
                    "symbol": "SPY   261218C00600000",
                    "bid": 10.1,
                    "ask": 10.3,
                    "totalVolume": 12,
                    "openInterest": 340,
                    "volatility": 18.5,
                }
            ]
        }
    },
    "putExpDateMap": {
        "2026-12-18:24": {
            "600.0": [
                {
                    "symbol": "SPY   261218P00600000",
                    "bid": 9.0,
                    "ask": 9.2,
                    "totalVolume": 7,
                    "openInterest": 120,
                    "volatility": -999.0,
                }
            ]
        }
    },
}


def test_parse_chain_maps_fields_and_units():
    chain = parse_chain(PAYLOAD, asof=datetime(2026, 11, 24, 15, tzinfo=UTC))
    assert chain.exercise_style is ExerciseStyle.AMERICAN
    assert chain.source == "schwab"
    call, put = chain.pairs(date(2026, 12, 18))[0]
    assert call.right is Right.CALL and call.vendor_iv == 0.185
    assert put.vendor_iv is None  # -999 means not available
    assert call.open_interest == 340 and call.contract_symbol == "SPY   261218C00600000"


def test_index_roots_are_european():
    assert exercise_style_for("$SPX") is ExerciseStyle.EUROPEAN
    assert exercise_style_for("XSP") is ExerciseStyle.EUROPEAN
    assert exercise_style_for("AAPL") is ExerciseStyle.AMERICAN
