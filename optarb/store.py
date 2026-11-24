"""A folder of Parquet snapshots, queried with DuckDB.

Layout: ``<root>/<SYMBOL>/<SYMBOL>_<YYYYMMDDTHHMMSSZ>.parquet``. DuckDB reads the whole
folder as one table called ``quotes``, so ad hoc SQL works without loading everything into
memory first.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from optarb.snapshot import iter_snapshots


class SnapshotStore:
    """Note that ``asof`` is a DuckDB keyword, so quote it in SQL: ``SELECT "asof" FROM quotes``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @property
    def pattern(self) -> str:
        return str(self.root / "**" / "*.parquet")

    def files(self) -> list[Path]:
        return sorted(self.root.rglob("*.parquet"))

    def connect(self) -> duckdb.DuckDBPyConnection:
        if not self.files():
            raise FileNotFoundError(f"no snapshots under {self.root}")
        con = duckdb.connect()
        con.execute(f"CREATE VIEW quotes AS SELECT * FROM read_parquet('{self.pattern}', union_by_name = true)")
        return con

    def query(self, sql: str) -> pd.DataFrame:
        with self.connect() as con:
            return con.execute(sql).df()

    def inventory(self) -> pd.DataFrame:
        """One row per symbol: snapshot count, first and last timestamp, quote rows."""
        return self.query(
            """
            SELECT symbol, count(DISTINCT "asof") AS snapshots, min("asof") AS first_asof, max("asof") AS last_asof,
                   count(*) AS quote_rows, any_value(source) AS source
            FROM quotes GROUP BY symbol ORDER BY symbol
            """
        )

    def chains(self, symbol: str | None = None):
        """Yield every snapshot as an ``OptionsChain``, oldest first."""
        where = "WHERE symbol = ?" if symbol else ""
        with self.connect() as con:
            df = con.execute(f'SELECT * FROM quotes {where} ORDER BY "asof"', [symbol] if symbol else []).df()
        yield from iter_snapshots(df)
