"""Where option chains come from. Anything with ``fetch`` can feed ``optarb record`` and ``optarb session``.

Two sources ship: ``SchwabProvider`` (live market data through schwab-py) and ``RecordedSource``
(Schwab JSON payloads saved earlier, replayed through the same parser). Because both hand back the
raw payload and the parsed chain, everything downstream (snapshots, scan, Greeks, study) runs the
same code whichever one is used.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from optarb.models import OptionsChain


class Exhausted(Exception):
    """A replayed source has no more payloads for this symbol."""


class ChainSource(Protocol):
    def fetch(
        self,
        symbol: str,
        max_dte: int = 60,
        strike_count: int | None = 40,
        dividend_yield: float = 0.0,
        asof: datetime | None = None,
    ) -> tuple[dict, OptionsChain]: ...
