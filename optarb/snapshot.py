"""Save and load chain snapshots as Parquet, one file per underlying per moment.

Each row is one option quote; the underlying's quote and the snapshot's metadata are
repeated on every row so a single file (or a folder of them) can be queried with DuckDB
without joins.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from optarb.models import ExerciseStyle, OptionQuote, OptionsChain, Right, Underlying

COLUMNS = [
    "symbol",
    "asof",
    "exercise_style",
    "source",
    "underlying_bid",
    "underlying_ask",
    "underlying_last",
    "dividend_yield",
    "expiration",
    "strike",
    "right",
    "bid",
    "ask",
    "last",
    "volume",
    "open_interest",
    "vendor_iv",
    "contract_symbol",
]


def to_frame(chain: OptionsChain) -> pd.DataFrame:
    u = chain.underlying
    rows = [
        {
            "symbol": chain.symbol,
            "asof": chain.asof,
            "exercise_style": chain.exercise_style.value,
            "source": chain.source,
            "underlying_bid": u.bid,
            "underlying_ask": u.ask,
            "underlying_last": u.last,
            "dividend_yield": u.dividend_yield,
            "expiration": q.expiration,
            "strike": q.strike,
            "right": q.right.value,
            "bid": q.bid,
            "ask": q.ask,
            "last": q.last,
            "volume": q.volume,
            "open_interest": q.open_interest,
            "vendor_iv": q.vendor_iv,
            "contract_symbol": q.contract_symbol,
        }
        for q in chain.quotes
    ]
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["asof"] = pd.to_datetime(df["asof"], utc=True)
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
    return df


def from_frame(df: pd.DataFrame) -> OptionsChain:
    if df.empty:
        raise ValueError("empty snapshot")
    if df[["symbol", "asof"]].drop_duplicates().shape[0] != 1:
        raise ValueError("a frame must hold exactly one symbol and one asof; use iter_snapshots for more")
    first = df.iloc[0]
    asof = pd.Timestamp(first["asof"])
    asof = (asof.tz_localize("UTC") if asof.tzinfo is None else asof).to_pydatetime()
    underlying = Underlying(
        symbol=str(first["symbol"]),
        bid=float(first["underlying_bid"]),
        ask=float(first["underlying_ask"]),
        last=float(first["underlying_last"]),
        dividend_yield=float(first["dividend_yield"]),
    )
    quotes = [
        OptionQuote(
            strike=float(r.strike),
            expiration=pd.Timestamp(r.expiration).date(),
            right=Right(r.right),
            bid=float(r.bid),
            ask=float(r.ask),
            last=float(r.last),
            volume=int(r.volume),
            open_interest=int(r.open_interest),
            vendor_iv=None if pd.isna(r.vendor_iv) else float(r.vendor_iv),
            contract_symbol=str(r.contract_symbol),
        )
        for r in df.itertuples(index=False)
    ]
    return OptionsChain(underlying, asof, ExerciseStyle(first["exercise_style"]), quotes, str(first["source"]))


def snapshot_path(root: str | Path, chain: OptionsChain) -> Path:
    stamp = chain.asof.strftime("%Y%m%dT%H%M%SZ")
    return Path(root) / chain.symbol / f"{chain.symbol}_{stamp}.parquet"


def save(chain: OptionsChain, root: str | Path) -> Path:
    path = snapshot_path(root, chain)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_frame(chain).to_parquet(path, index=False)
    return path


def load(path: str | Path) -> OptionsChain:
    return from_frame(pd.read_parquet(path))


def iter_snapshots(df: pd.DataFrame):
    """Yield one ``OptionsChain`` per (symbol, asof) in a frame that holds many, oldest first."""
    for _, group in df.sort_values("asof").groupby(["symbol", "asof"], sort=False):
        yield from_frame(group)
