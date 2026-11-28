"""Save and load chain snapshots as Parquet, one file per underlying per moment.

Each row is one option quote; the underlying's quote and the snapshot's metadata are
repeated on every row so a single file (or a folder of them) can be queried with DuckDB
without joins. Columns added in later versions are optional on read, so older snapshots
still load.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from optarb.models import ExerciseStyle, OptionQuote, OptionsChain, Right, Underlying

CHAIN_COLUMNS = [
    "symbol", "asof", "exercise_style", "source", "delayed", "vendor_rate",
    "underlying_bid", "underlying_ask", "underlying_last", "dividend_yield",
]  # fmt: skip
QUOTE_COLUMNS = [
    "expiration", "strike", "right", "bid", "ask", "last", "volume", "open_interest", "bid_size", "ask_size",
    "quote_time", "vendor_iv", "vendor_delta", "vendor_gamma", "vendor_theta", "vendor_vega", "vendor_rho",
    "contract_symbol",
]  # fmt: skip
COLUMNS = CHAIN_COLUMNS + QUOTE_COLUMNS
_OPTIONAL_FLOATS = ("vendor_iv", "vendor_delta", "vendor_gamma", "vendor_theta", "vendor_vega", "vendor_rho")


def to_frame(chain: OptionsChain) -> pd.DataFrame:
    u = chain.underlying
    head = {
        "symbol": chain.symbol, "asof": chain.asof, "exercise_style": chain.exercise_style.value,
        "source": chain.source, "delayed": chain.delayed, "vendor_rate": chain.vendor_rate,
        "underlying_bid": u.bid, "underlying_ask": u.ask, "underlying_last": u.last,
        "dividend_yield": u.dividend_yield,
    }  # fmt: skip
    rows = [{**head, **{c: getattr(q, c) for c in QUOTE_COLUMNS}} for q in chain.quotes]
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["right"] = df["right"].map(lambda r: r.value if isinstance(r, Right) else r)
    df["asof"] = pd.to_datetime(df["asof"], utc=True)
    df["quote_time"] = pd.to_datetime(df["quote_time"], utc=True)
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
    df["vendor_rate"] = df["vendor_rate"].astype(float)
    for col in _OPTIONAL_FLOATS:
        df[col] = df[col].astype(float)
    return df


def _utc(ts) -> object:
    if ts is None or pd.isna(ts):
        return None
    ts = pd.Timestamp(ts)
    return (ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")).to_pydatetime()


def _opt_float(v) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


def from_frame(df: pd.DataFrame) -> OptionsChain:
    if df.empty:
        raise ValueError("empty snapshot")
    if df[["symbol", "asof"]].drop_duplicates().shape[0] != 1:
        raise ValueError("a frame must hold exactly one symbol and one asof; use iter_snapshots for more")
    df = df.copy()
    for col in COLUMNS:  # older snapshots lack the newer columns
        if col not in df:
            df[col] = 0 if col in ("bid_size", "ask_size") else (False if col == "delayed" else None)
    first = df.iloc[0]
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
            vendor_iv=_opt_float(r.vendor_iv),
            contract_symbol=str(r.contract_symbol),
            bid_size=int(r.bid_size or 0),
            ask_size=int(r.ask_size or 0),
            quote_time=_utc(r.quote_time),
            vendor_delta=_opt_float(r.vendor_delta),
            vendor_gamma=_opt_float(r.vendor_gamma),
            vendor_theta=_opt_float(r.vendor_theta),
            vendor_vega=_opt_float(r.vendor_vega),
            vendor_rho=_opt_float(r.vendor_rho),
        )
        for r in df.itertuples(index=False)
    ]
    return OptionsChain(
        underlying,
        _utc(first["asof"]),
        ExerciseStyle(first["exercise_style"]),
        quotes,
        str(first["source"]),
        delayed=bool(first["delayed"]) if not pd.isna(first["delayed"]) else False,
        vendor_rate=_opt_float(first["vendor_rate"]),
    )


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
