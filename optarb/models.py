"""Core data model: option quotes and chains.

Prices are per share. Multiply by ``CONTRACT_MULTIPLIER`` for dollars per contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from enum import Enum
from zoneinfo import ZoneInfo

CONTRACT_MULTIPLIER = 100
_NEW_YORK = ZoneInfo("America/New_York")
_SECONDS_PER_YEAR = 365.0 * 24 * 3600


class Right(str, Enum):
    CALL = "C"
    PUT = "P"


class ExerciseStyle(str, Enum):
    """Exercise style decides which no-arbitrage bounds apply.

    European: index options such as SPX and XSP (cash settled, exercise at expiry only).
    American: equity and ETF options such as SPY and AAPL (physically settled, early exercise allowed).
    """

    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(frozen=True)
class OptionQuote:
    strike: float
    expiration: date
    right: Right
    bid: float
    ask: float
    last: float = 0.0
    volume: int = 0
    open_interest: int = 0
    vendor_iv: float | None = None  # decimal, as reported by the data source
    contract_symbol: str = ""
    bid_size: int = 0  # contracts shown at the bid; 0 when the source doesn't say
    ask_size: int = 0
    quote_time: datetime | None = None  # when this quote was last updated, if the source says
    # Greeks as reported by the data source, in this package's units (see optarb.pricing)
    vendor_delta: float | None = None
    vendor_gamma: float | None = None
    vendor_theta: float | None = None
    vendor_vega: float | None = None
    vendor_rho: float | None = None

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Bid/ask spread as a percent of mid; infinite when there is no two-sided market."""
        if self.bid <= 0 or self.ask <= 0 or self.mid <= 0:
            return float("inf")
        return 100.0 * self.spread / self.mid


@dataclass(frozen=True)
class Underlying:
    symbol: str
    bid: float
    ask: float
    last: float = 0.0
    dividend_yield: float = 0.0  # continuous, annualized; an input the user supplies

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return 0.5 * (self.bid + self.ask)
        return self.last


@dataclass
class OptionsChain:
    """One snapshot of an option chain for a single underlying."""

    underlying: Underlying
    asof: datetime
    exercise_style: ExerciseStyle
    quotes: list[OptionQuote] = field(default_factory=list)
    source: str = "unknown"  # e.g. "synthetic", "schwab"
    delayed: bool = False  # True when the source says these quotes are delayed
    vendor_rate: float | None = None  # risk-free rate the source used for its Greeks, decimal

    def replace_quotes(self, quotes: list[OptionQuote]) -> OptionsChain:
        """Same snapshot with a different set of quotes."""
        return OptionsChain(
            self.underlying, self.asof, self.exercise_style, quotes, self.source, self.delayed, self.vendor_rate
        )

    def __post_init__(self) -> None:
        if self.asof.tzinfo is None:
            raise ValueError("asof must be timezone aware")

    @property
    def symbol(self) -> str:
        return self.underlying.symbol

    @property
    def expirations(self) -> list[date]:
        return sorted({q.expiration for q in self.quotes})

    def side(self, expiration: date, right: Right) -> list[OptionQuote]:
        """Quotes for one expiry and right, sorted by strike."""
        return sorted(
            (q for q in self.quotes if q.expiration == expiration and q.right == right),
            key=lambda q: q.strike,
        )

    def pairs(self, expiration: date) -> list[tuple[OptionQuote, OptionQuote]]:
        """(call, put) pairs sharing a strike for one expiry, sorted by strike."""
        puts = {p.strike: p for p in self.side(expiration, Right.PUT)}
        return [(c, puts[c.strike]) for c in self.side(expiration, Right.CALL) if c.strike in puts]

    def year_fraction(self, expiration: date) -> float:
        """ACT/365 time from the snapshot to 4:00 pm New York time on the expiry date."""
        expiry_close = datetime.combine(expiration, time(16, 0), tzinfo=_NEW_YORK)
        seconds = (expiry_close - self.asof).total_seconds()
        return max(seconds, 0.0) / _SECONDS_PER_YEAR


def utc_now() -> datetime:
    return datetime.now(UTC)
