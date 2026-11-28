"""Replay Schwab option-chain JSON saved earlier (``record --save-raw``, or a fixture) as if it were live.

Files are matched to a symbol by name (``SPY_20261120T153000Z.json``) or by the payload's own
``symbol``, and served oldest first. The quote time comes from the stamp in the file name when there
is one, so a replayed session keeps its real timing. Nothing here touches the network.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from optarb.models import OptionsChain
from optarb.providers.base import Exhausted
from optarb.providers.schwab import parse_chain

STAMP = re.compile(r"(\d{8}T\d{6}Z)")


def stamp_of(path: Path) -> datetime | None:
    m = STAMP.search(path.name)
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC) if m else None


def _symbol_of(path: Path) -> str | None:
    try:
        return json.loads(path.read_text()).get("symbol")
    except (OSError, ValueError):
        return None


class RecordedSource:
    def __init__(self, path: str | Path):
        root = Path(path)
        files = [root] if root.is_file() else sorted(root.rglob("*.json"))
        if not files:
            raise FileNotFoundError(f"no .json payloads under {root}")
        self._files = files
        self._served: dict[str, int] = {}

    def files_for(self, symbol: str) -> list[Path]:
        safe = symbol.replace("$", "").upper()
        named = [f for f in self._files if f.name.upper().startswith(f"{safe}_")]
        found = named or [f for f in self._files if (_symbol_of(f) or "").replace("$", "").upper() == safe]
        return sorted(found, key=lambda f: (stamp_of(f) or datetime.min.replace(tzinfo=UTC), f.name))

    def fetch(
        self,
        symbol: str,
        max_dte: int = 60,
        strike_count: int | None = 40,
        dividend_yield: float = 0.0,
        asof: datetime | None = None,
    ) -> tuple[dict, OptionsChain]:
        files = self.files_for(symbol)
        i = self._served.get(symbol, 0)
        if i >= len(files):
            raise Exhausted(symbol)
        self._served[symbol] = i + 1
        path = files[i]
        payload = json.loads(path.read_text())
        when = stamp_of(path) or asof or datetime.fromtimestamp(path.stat().st_mtime, UTC)
        return payload, parse_chain(payload, symbol=symbol, asof=when, dividend_yield=dividend_yield)
