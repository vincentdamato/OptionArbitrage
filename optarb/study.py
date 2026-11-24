"""Research study: how often do no-arbitrage bounds break at executable prices, by how much, and for how long?

Run every check across a series of snapshots, then measure:

* frequency: share of snapshots with at least one violation, by check;
* size: net edge after commissions, in dollars per package;
* persistence: how many consecutive snapshots the same package stays mispriced;
* box-implied rates: the best lending and borrowing rates boxes offer, against a reference rate.

The answer on real data is the point of the study, and it needs recorded chains. ``simulate_session``
makes a labeled synthetic series with known, planted violations so the pipeline itself can be tested
and demonstrated; its numbers say nothing about real markets.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from optarb.arbitrage import ScanConfig, box_rates, scan
from optarb.models import ExerciseStyle, OptionsChain
from optarb.providers.synthetic import make_chain, plant_butterfly, plant_short_box, plant_vertical


@dataclass
class StudyResult:
    opportunities: pd.DataFrame  # one row per violation per snapshot
    snapshots: pd.DataFrame  # one row per snapshot
    box_rates: pd.DataFrame  # one row per snapshot per expiry (European only)
    config: ScanConfig

    @property
    def synthetic(self) -> bool:
        return bool((self.snapshots["source"] == "synthetic").any())


def run(chains: Iterable[OptionsChain], cfg: ScanConfig | None = None) -> StudyResult:
    cfg = cfg or ScanConfig()
    opp_rows, snap_rows, rate_rows = [], [], []
    for chain in chains:
        found = scan(chain, cfg)
        snap_rows.append(
            {
                "symbol": chain.symbol,
                "asof": chain.asof,
                "source": chain.source,
                "exercise_style": chain.exercise_style.value,
                "quotes": len(chain.quotes),
                "spot": chain.underlying.mid,
                "violations": len(found),
            }
        )
        for o in found:
            opp_rows.append(
                {
                    "symbol": o.symbol,
                    "asof": o.asof,
                    "check": o.check,
                    "kind": o.kind.value,
                    "expiration": o.expiration,
                    "key": o.key,
                    "legs": len(o.legs),
                    "gross_edge": o.gross_edge,
                    "costs": o.costs,
                    "net_edge": o.net_edge,
                }
            )
        broken = {o.expiration for o in found if o.kind.value == "model_free"}
        for br in box_rates(chain, cfg):
            if br.expiration in broken:  # a hard violation at this expiry would distort the rate
                continue
            rate_rows.append(
                {
                    "symbol": chain.symbol,
                    "asof": chain.asof,
                    "expiration": br.expiration,
                    "T": br.T,
                    "lend_rate": br.best_lend_rate,
                    "borrow_rate": br.best_borrow_rate,
                }
            )
    opp_cols = ["symbol", "asof", "check", "kind", "expiration", "key", "legs", "gross_edge", "costs", "net_edge"]
    return StudyResult(
        opportunities=pd.DataFrame(opp_rows, columns=opp_cols),
        snapshots=pd.DataFrame(snap_rows),
        box_rates=pd.DataFrame(rate_rows, columns=["symbol", "asof", "expiration", "T", "lend_rate", "borrow_rate"]),
        config=cfg,
    )


def persistence(result: StudyResult) -> pd.DataFrame:
    """One row per episode: a check violated at one expiry in an unbroken run of consecutive snapshots.

    Episodes are keyed on (symbol, check, expiry), not on the exact package, because one bad quote
    breaks many overlapping packages at once and strikes shift as spot moves. The episode's size is
    the best net edge any package reached during it.
    """
    cols = ["symbol", "check", "expiration", "start", "end", "snapshots", "minutes", "max_net_edge", "packages"]
    opps = result.opportunities
    if opps.empty:
        return pd.DataFrame(columns=cols)
    order = {sym: {ts: i for i, ts in enumerate(sorted(g["asof"]))} for sym, g in result.snapshots.groupby("symbol")}
    per_snap = (
        opps.groupby(["symbol", "check", "expiration", "asof"])
        .agg(net_edge=("net_edge", "max"), packages=("key", "nunique"))
        .reset_index()
    )
    rows = []
    for (sym, check, exp), g in per_snap.groupby(["symbol", "check", "expiration"]):
        g = g.sort_values("asof")
        idx = [order[sym][ts] for ts in g["asof"]]
        start = 0
        for i in range(1, len(idx) + 1):
            if i == len(idx) or idx[i] != idx[i - 1] + 1:
                run_ = g.iloc[start:i]
                rows.append(
                    {
                        "symbol": sym,
                        "check": check,
                        "expiration": exp,
                        "start": run_["asof"].iloc[0],
                        "end": run_["asof"].iloc[-1],
                        "snapshots": i - start,
                        "minutes": (run_["asof"].iloc[-1] - run_["asof"].iloc[0]).total_seconds() / 60.0,
                        "max_net_edge": run_["net_edge"].max(),
                        "packages": int(run_["packages"].max()),
                    }
                )
                start = i
    return pd.DataFrame(rows, columns=cols)


def summarize(result: StudyResult) -> pd.DataFrame:
    """Per check: snapshots with a violation, episode count, and the size and life of episodes."""
    n_snaps = len(result.snapshots)
    opps = result.opportunities
    if opps.empty or n_snaps == 0:
        return pd.DataFrame(
            columns=[
                "check",
                "kind",
                "share_of_snapshots",
                "episodes",
                "median_net_edge",
                "p95_net_edge",
                "median_snapshots_alive",
                "max_snapshots_alive",
            ]
        )
    eps = persistence(result)
    rows = []
    for check, g in opps.groupby("check"):
        e = eps[eps["check"] == check]
        rows.append(
            {
                "check": check,
                "kind": g["kind"].iloc[0],
                "share_of_snapshots": g[["symbol", "asof"]].drop_duplicates().shape[0] / n_snaps,
                "episodes": len(e),
                "median_net_edge": float(e["max_net_edge"].median()),
                "p95_net_edge": float(e["max_net_edge"].quantile(0.95)),
                "median_snapshots_alive": float(e["snapshots"].median()),
                "max_snapshots_alive": int(e["snapshots"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("episodes", ascending=False, ignore_index=True)


def _md_table(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    if df.empty:
        return "_none_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(result: StudyResult, out_dir: str | Path) -> Path:
    """Write ``report.md`` plus CSVs of the raw tables to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(result)
    eps = persistence(result)
    result.opportunities.to_csv(out / "opportunities.csv", index=False)
    eps.to_csv(out / "episodes.csv", index=False)
    result.box_rates.to_csv(out / "box_rates.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)

    snaps = result.snapshots
    cfg = result.config
    lines = ["# No-arbitrage violations at executable prices", ""]
    if result.synthetic:
        lines += [
            "> **Synthetic data.** These snapshots were generated with planted violations to exercise the",
            "> pipeline. The numbers below describe the simulation, not any real market.",
            "",
        ]
    lines += [
        "## Data",
        "",
        f"- Snapshots: {len(snaps)} across {snaps['symbol'].nunique()} underlying(s)",
        f"- From {snaps['asof'].min():%Y-%m-%d %H:%M} to {snaps['asof'].max():%Y-%m-%d %H:%M} UTC",
        f"- Sources: {', '.join(sorted(snaps['source'].unique()))}",
        "",
        "## Assumptions",
        "",
        f"- Buy at the ask, sell at the bid; commission ${cfg.commission_per_contract:.2f} per contract, "
        f"${cfg.commission_per_share:.4f} per share; stock borrow {cfg.borrow_rate:.2%} a year.",
        f"- Minimum net edge to count: ${cfg.min_net_edge:.2f} per package.",
        "- Reference rate for rate-relative checks: "
        + (f"{cfg.reference_rate:.2%}" if cfg.reference_rate is not None else "none (rate-relative checks off)"),
        "- A snapshot is one moment. Nothing here says the quotes were fillable in size, or at the same instant.",
        "",
        "## Results by check",
        "",
        _md_table(summary),
        "",
        "`share_of_snapshots` is the fraction of snapshots with at least one violation of that check. An episode is",
        "one check broken at one expiry in consecutive snapshots; its size is the best net edge any package reached",
        "during it, in dollars per package.",
        "",
    ]
    if not result.box_rates.empty:
        rates = result.box_rates.groupby("expiration")[["lend_rate", "borrow_rate"]].median().reset_index()
        lines += [
            "## Box-implied rates (median across snapshots)",
            "",
            "The best rate any box at that expiry offered after commissions, from snapshots where that expiry had",
            "no hard violation. `lend_rate` is earned buying a box and `borrow_rate` is paid selling one. A blank",
            "means no box at that expiry priced inside its bounds after costs.",
            "",
            _md_table(rates, "{:.2%}").replace("nan%", ""),
            "",
        ]
    longest = (
        eps.sort_values("snapshots", ascending=False).head(10)[
            ["symbol", "check", "expiration", "start", "snapshots", "minutes", "max_net_edge"]
        ]
        if not eps.empty
        else eps
    )
    lines += ["## Longest-lived episodes", "", _md_table(longest), ""]
    path = out / "report.md"
    path.write_text("\n".join(lines))
    return path


def simulate_session(
    n_snapshots: int = 40,
    interval_minutes: int = 10,
    seed: int = 11,
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN,
    plant_probability: float = 0.15,
    mean_life: float = 2.5,
) -> list[OptionsChain]:
    """A labeled synthetic intraday series: spot follows a random walk and violations appear and decay.

    Each planted violation is held on the same strikes for a geometric number of snapshots, so the
    study's persistence measure has a known answer to recover.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2026, 11, 20, 14, 40, tzinfo=UTC)
    spot = 100.0
    active: list[tuple[str, int]] = []  # (kind, snapshots left)
    planters = {"butterfly": plant_butterfly, "short_box": plant_short_box, "vertical": plant_vertical}
    chains = []
    for i in range(n_snapshots):
        spot *= float(np.exp(rng.normal(0, 0.0015)))
        chain = make_chain(
            spot=round(spot, 2),
            asof=start + timedelta(minutes=interval_minutes * i),
            seed=seed + i,
            exercise_style=exercise_style,
            crr_steps=120,
        )
        active = [(k, left - 1) for k, left in active if left > 1]
        for kind in planters:
            if kind not in {k for k, _ in active} and rng.random() < plant_probability:
                active.append((kind, int(rng.geometric(1.0 / mean_life))))
        for kind, _ in active:
            if kind == "short_box" and exercise_style is not ExerciseStyle.EUROPEAN:
                continue
            planters[kind](chain)
        chains.append(chain)
    return chains
