"""Read-only Schwab market data provider: option chains in, ``OptionsChain`` out.

This module only reads quotes. It never touches accounts, orders or positions.

Credentials come from environment variables (``SCHWAB_API_KEY``, ``SCHWAB_APP_SECRET``,
``SCHWAB_CALLBACK_URL``, ``SCHWAB_TOKEN_PATH``) and the OAuth token file is managed by
``schwab-py``. Nothing here prints or stores them.

``parse_chain`` is a pure function so it can be tested on a saved payload without a network
connection. Contracts that don't follow the standard 100-share deliverable (adjusted options
after a split or special dividend) are dropped, because every bound here assumes it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from optarb.models import CONTRACT_MULTIPLIER, ExerciseStyle, OptionQuote, OptionsChain, Right, Underlying, utc_now

# Cash-settled index options with European exercise. Everything else is treated as American.
EUROPEAN_ROOTS = {"SPX", "SPXW", "XSP", "NDX", "NDXP", "RUT", "RUTW", "VIX", "VIXW", "DJX", "MRUT"}
ENV_KEYS = ("SCHWAB_API_KEY", "SCHWAB_APP_SECRET")
DEFAULT_TOKEN_PATH = "~/.schwab/token.json"
DEFAULT_CALLBACK = "https://127.0.0.1:8182"


def exercise_style_for(symbol: str) -> ExerciseStyle:
    root = symbol.lstrip("$").upper()
    return ExerciseStyle.EUROPEAN if root in EUROPEAN_ROOTS else ExerciseStyle.AMERICAN


def _float(raw: Any) -> float | None:
    """A finite number, or None. Schwab uses -999 and the string "NaN" for 'not available'."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")) or value <= -999.0:
        return None
    return value


def _number(d: dict, key: str, default: float = 0.0) -> float:
    value = _float(d.get(key))
    return default if value is None else value


def _vendor_iv(raw: Any) -> float | None:
    """Schwab reports volatility in percent."""
    value = _float(raw)
    return None if value is None or value <= 0 else value / 100.0


def _epoch_ms(raw: Any) -> datetime | None:
    value = _float(raw)
    if value is None or value <= 0:
        return None
    return datetime.fromtimestamp(value / 1000.0, tz=UTC)


def _standard(contract: dict) -> bool:
    if contract.get("nonStandard") is True or contract.get("mini") is True:
        return False
    multiplier = _float(contract.get("multiplier"))
    return multiplier is None or multiplier == CONTRACT_MULTIPLIER


def parse_chain(
    payload: dict,
    symbol: str | None = None,
    asof: datetime | None = None,
    exercise_style: ExerciseStyle | None = None,
    dividend_yield: float = 0.0,
) -> OptionsChain:
    """Convert a Schwab ``/chains`` JSON payload into an ``OptionsChain``."""
    symbol = (symbol or payload.get("symbol", "")).lstrip("$").upper()
    u = payload.get("underlying") or {}
    underlying = Underlying(
        symbol=symbol,
        bid=_number(u, "bid"),
        ask=_number(u, "ask"),
        last=_number(u, "last", _number(payload, "underlyingPrice")),
        dividend_yield=dividend_yield,
    )
    styles: set[str] = set()
    quotes: list[OptionQuote] = []
    for map_key, right in (("callExpDateMap", Right.CALL), ("putExpDateMap", Right.PUT)):
        for exp_key, strikes in (payload.get(map_key) or {}).items():
            expiration = date.fromisoformat(exp_key.split(":")[0])
            for strike_key, contracts in strikes.items():
                for c in contracts:
                    if not _standard(c):
                        continue
                    if c.get("exerciseType"):
                        styles.add(str(c["exerciseType"]).upper()[:1])
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
                            bid_size=int(_number(c, "bidSize")),
                            ask_size=int(_number(c, "askSize")),
                            quote_time=_epoch_ms(c.get("quoteTimeInLong")),
                            vendor_delta=_float(c.get("delta")),
                            vendor_gamma=_float(c.get("gamma")),
                            vendor_theta=_float(c.get("theta")),
                            vendor_vega=_float(c.get("vega")),
                            vendor_rho=_float(c.get("rho")),
                        )
                    )
    if exercise_style is None:
        if styles == {"E"}:
            exercise_style = ExerciseStyle.EUROPEAN
        elif styles == {"A"}:
            exercise_style = ExerciseStyle.AMERICAN
        else:
            exercise_style = exercise_style_for(symbol)
    rate = _float(payload.get("interestRate"))
    return OptionsChain(
        underlying=underlying,
        asof=asof or utc_now(),
        exercise_style=exercise_style,
        quotes=quotes,
        source="schwab",
        delayed=bool(payload.get("isDelayed", False)),
        vendor_rate=None if rate is None else rate / 100.0,
    )


@dataclass(frozen=True)
class SetupCheck:
    """What ``optarb doctor`` reports. It says whether things are present, never what they are."""

    schwab_py_installed: bool
    schwab_py_version: str | None
    api_key_set: bool
    app_secret_set: bool
    token_file: str
    token_file_exists: bool
    token_age_days: float | None

    @property
    def ready(self) -> bool:
        return self.schwab_py_installed and self.api_key_set and self.app_secret_set

    def problems(self) -> list[str]:
        out = []
        if not self.schwab_py_installed:
            out.append("schwab-py is not installed: pip install -e '.[schwab]'")
        elif self.schwab_py_version and self.schwab_py_version.split(".")[0] != "1":
            out.append(f"schwab-py {self.schwab_py_version} is installed; this package is written for 1.x")
        if not self.api_key_set:
            out.append("SCHWAB_API_KEY is not set")
        if not self.app_secret_set:
            out.append("SCHWAB_APP_SECRET is not set")
        if not self.token_file_exists:
            out.append("no token file yet: run `optarb record SPY --login` once to sign in through the browser")
        elif self.token_age_days is not None and self.token_age_days > 6.5:
            out.append(
                "the refresh token is about a week old and Schwab expires it at 7 days: sign in again with --login"
            )
        return out


def check_setup(token_path: str | None = None) -> SetupCheck:
    try:
        import warnings
        from importlib.metadata import version

        with warnings.catch_warnings():  # schwab-py 1.x announces 2.0 on import; we already pin <2.0
            warnings.simplefilter("ignore")
            import schwab  # noqa: F401

        installed, ver = True, version("schwab-py")
    except Exception:
        installed, ver = False, None
    token = Path(token_path or os.environ.get("SCHWAB_TOKEN_PATH", DEFAULT_TOKEN_PATH)).expanduser()
    age = None
    if token.exists():
        age = (datetime.now().timestamp() - token.stat().st_mtime) / 86400.0
    return SetupCheck(
        schwab_py_installed=installed,
        schwab_py_version=ver,
        api_key_set=bool(os.environ.get("SCHWAB_API_KEY")),
        app_secret_set=bool(os.environ.get("SCHWAB_APP_SECRET")),
        token_file=str(token),
        token_file_exists=token.exists(),
        token_age_days=age,
    )


class SchwabProvider:
    """Fetches option chains with ``schwab-py`` (pinned below 2.0). Market data only.

    Pass ``client`` to use an already-built schwab-py client (or a stand-in in tests); otherwise
    ``from_env`` builds one from the environment and the token file.
    """

    def __init__(self, client: Any):
        self._client = client

    @classmethod
    def from_env(cls, token_path: str | None = None, interactive: bool = False) -> SchwabProvider:
        try:
            from schwab import auth
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install the Schwab extra: pip install -e '.[schwab]'") from exc
        missing = [k for k in ENV_KEYS if not os.environ.get(k)]
        if missing:
            raise RuntimeError(f"Set {' and '.join(missing)} in the environment")
        callback = os.environ.get("SCHWAB_CALLBACK_URL", DEFAULT_CALLBACK)
        token = Path(token_path or os.environ.get("SCHWAB_TOKEN_PATH", DEFAULT_TOKEN_PATH)).expanduser()
        token.parent.mkdir(parents=True, exist_ok=True)
        client = auth.easy_client(
            os.environ["SCHWAB_API_KEY"], os.environ["SCHWAB_APP_SECRET"], callback, str(token), interactive=interactive
        )
        return cls(client)

    def raw_chain(self, symbol: str, max_dte: int = 60, strike_count: int | None = 40) -> dict:
        today = date.today()
        kwargs: dict[str, Any] = {
            "strike_count": strike_count,
            "include_underlying_quote": True,
            "from_date": today,
            "to_date": today + timedelta(days=max_dte),
        }
        options = getattr(getattr(self._client, "Options", None), "ContractType", None)
        if options is not None:
            kwargs["contract_type"] = options.ALL
        resp = self._client.get_option_chain(symbol, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def option_chain(
        self,
        symbol: str,
        max_dte: int = 60,
        strike_count: int | None = 40,
        dividend_yield: float = 0.0,
        asof: datetime | None = None,
    ) -> OptionsChain:
        payload = self.raw_chain(symbol, max_dte, strike_count)
        return parse_chain(payload, symbol=symbol, asof=asof or utc_now(), dividend_yield=dividend_yield)

    def fetch(
        self,
        symbol: str,
        max_dte: int = 60,
        strike_count: int | None = 40,
        dividend_yield: float = 0.0,
        asof: datetime | None = None,
    ) -> tuple[dict, OptionsChain]:
        """The raw payload and the parsed chain from one request, so the raw JSON can be kept too."""
        payload = self.raw_chain(symbol, max_dte, strike_count)
        return payload, parse_chain(payload, symbol=symbol, asof=asof or utc_now(), dividend_yield=dividend_yield)
