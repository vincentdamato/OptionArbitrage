"""Read-only Schwab market data provider: option chains in, ``OptionsChain`` out.

This module only reads quotes. It never touches accounts, orders or positions.

Credentials come from environment variables (``SCHWAB_API_KEY``, ``SCHWAB_APP_SECRET``,
``SCHWAB_CALLBACK_URL``, ``SCHWAB_TOKEN_PATH``) and the OAuth token file is managed by
``schwab-py``. Nothing here prints or stores them.

The parsing is a pure function (``parse_chain``) so it can be tested on a saved payload
without a network connection.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from optarb.models import ExerciseStyle, OptionQuote, OptionsChain, Right, Underlying, utc_now

# Cash-settled index options with European exercise. Everything else is treated as American.
EUROPEAN_ROOTS = {"SPX", "SPXW", "XSP", "NDX", "NDXP", "RUT", "RUTW", "VIX", "VIXW", "DJX", "MRUT"}


def exercise_style_for(symbol: str) -> ExerciseStyle:
    root = symbol.lstrip("$").upper()
    return ExerciseStyle.EUROPEAN if root in EUROPEAN_ROOTS else ExerciseStyle.AMERICAN


def _vendor_iv(raw: Any) -> float | None:
    """Schwab reports volatility in percent and uses -999 for 'not available'."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value / 100.0


def _number(d: dict, key: str, default: float = 0.0) -> float:
    try:
        value = float(d.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if value == value else default  # NaN guard


def parse_chain(
    payload: dict,
    symbol: str | None = None,
    asof: datetime | None = None,
    exercise_style: ExerciseStyle | None = None,
    dividend_yield: float = 0.0,
) -> OptionsChain:
    """Convert a Schwab ``/chains`` JSON payload into an ``OptionsChain``."""
    symbol = (symbol or payload.get("symbol", "")).upper()
    u = payload.get("underlying") or {}
    underlying = Underlying(
        symbol=symbol,
        bid=_number(u, "bid"),
        ask=_number(u, "ask"),
        last=_number(u, "last", _number(payload, "underlyingPrice")),
        dividend_yield=dividend_yield,
    )
    quotes: list[OptionQuote] = []
    for map_key, right in (("callExpDateMap", Right.CALL), ("putExpDateMap", Right.PUT)):
        for exp_key, strikes in (payload.get(map_key) or {}).items():
            expiration = date.fromisoformat(exp_key.split(":")[0])
            for strike_key, contracts in strikes.items():
                for c in contracts:
                    quotes.append(
                        OptionQuote(
                            strike=float(strike_key),
                            expiration=expiration,
                            right=right,
                            bid=max(_number(c, "bid"), 0.0),
                            ask=max(_number(c, "ask"), 0.0),
                            last=_number(c, "last"),
                            volume=int(_number(c, "totalVolume")),
                            open_interest=int(_number(c, "openInterest")),
                            vendor_iv=_vendor_iv(c.get("volatility")),
                            contract_symbol=str(c.get("symbol", "")).strip(),
                        )
                    )
    return OptionsChain(
        underlying=underlying,
        asof=asof or utc_now(),
        exercise_style=exercise_style or exercise_style_for(symbol),
        quotes=quotes,
        source="schwab",
    )


class SchwabProvider:
    """Fetches option chains with ``schwab-py`` (pinned below 2.0). Market data only."""

    def __init__(self, token_path: str | None = None, interactive: bool = False):
        try:
            from schwab import auth
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install the Schwab extra: pip install 'optarb[schwab]'") from exc
        try:
            api_key = os.environ["SCHWAB_API_KEY"]
            app_secret = os.environ["SCHWAB_APP_SECRET"]
        except KeyError as exc:
            raise RuntimeError("Set SCHWAB_API_KEY and SCHWAB_APP_SECRET in the environment") from exc
        callback = os.environ.get("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8182")
        token = Path(token_path or os.environ.get("SCHWAB_TOKEN_PATH", "~/.schwab/token.json")).expanduser()
        token.parent.mkdir(parents=True, exist_ok=True)
        self._client = auth.easy_client(api_key, app_secret, callback, str(token), interactive=interactive)

    def option_chain(
        self,
        symbol: str,
        max_dte: int = 60,
        strike_count: int | None = 40,
        dividend_yield: float = 0.0,
    ) -> OptionsChain:
        today = date.today()
        resp = self._client.get_option_chain(
            symbol,
            contract_type=self._client.Options.ContractType.ALL,
            strike_count=strike_count,
            include_underlying_quote=True,
            from_date=today,
            to_date=today + timedelta(days=max_dte),
        )
        resp.raise_for_status()
        asof = utc_now()
        return parse_chain(resp.json(), symbol=symbol.lstrip("$"), asof=asof, dividend_yield=dividend_yield)
