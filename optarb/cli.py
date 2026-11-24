"""Command line: ``optarb scan | forwards | sample | simulate | record | study``.

Only ``record`` touches the network, and only to read market data.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from optarb.arbitrage import ScanConfig, best_per_group, scan
from optarb.forwards import fit_all_forwards
from optarb.models import ExerciseStyle
from optarb.snapshot import load, save


def _config(args: argparse.Namespace) -> ScanConfig:
    return ScanConfig(
        commission_per_contract=args.commission,
        borrow_rate=args.borrow,
        min_net_edge=args.min_edge,
        reference_rate=args.rate,
        max_spread_pct=args.max_spread_pct,
    )


def _add_cost_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--commission", type=float, default=0.65, help="dollars per contract (default 0.65)")
    p.add_argument("--borrow", type=float, default=0.0, help="annual stock borrow fee for short stock legs")
    p.add_argument("--min-edge", type=float, default=0.01, help="minimum net edge in dollars per package")
    p.add_argument("--rate", type=float, default=None, help="reference funding rate, e.g. 0.045; enables rate checks")
    p.add_argument("--max-spread-pct", type=float, default=100.0, help="skip quotes wider than this percent of mid")


def cmd_scan(args: argparse.Namespace) -> int:
    chain = load(args.snapshot)
    found = scan(chain, _config(args))
    if not args.all:
        found = best_per_group(found)
    print(
        f"{chain.symbol} {chain.asof:%Y-%m-%d %H:%M} UTC  source={chain.source}  "
        f"style={chain.exercise_style.value}  quotes={len(chain.quotes)}"
    )
    if chain.source == "synthetic":
        print("Synthetic snapshot: violations were planted to demonstrate the checks.")
    if not found:
        print("No violations after costs.")
        return 0
    rows = [
        {
            "check": o.check,
            "expiry": o.expiration,
            "net $": round(o.net_edge, 2),
            "gross $": round(o.gross_edge, 2),
            "legs": "  ".join(
                f"{leg.action} {leg.quantity} {leg.instrument} {leg.strike:g}@{leg.price:.2f}"
                if leg.instrument != "STOCK"
                else f"{leg.action} {leg.quantity} STOCK@{leg.price:.2f}"
                for leg in o.legs
            ),
        }
        for o in found[: args.limit]
    ]
    with pd.option_context("display.max_colwidth", 120, "display.width", 200):
        print(pd.DataFrame(rows).to_string(index=False))
    return 0


def cmd_forwards(args: argparse.Namespace) -> int:
    chain = load(args.snapshot)
    fits = fit_all_forwards(chain)
    rows = [
        {
            "expiry": e,
            "T": round(f.T, 4),
            "DF": round(f.discount_factor, 6),
            "forward": round(f.forward, 4),
            "implied r": f"{f.implied_rate:.2%}",
            "implied r-q": f"{f.implied_carry(chain.underlying.mid):.2%}",
            "strikes": f.n_strikes,
            "approx": f.approximate,
        }
        for e, f in fits.items()
    ]
    print(pd.DataFrame(rows).to_string(index=False) if rows else "Not enough two-sided pairs to fit.")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    from optarb.providers.synthetic import make_chain, sample_chain

    chain = make_chain(exercise_style=ExerciseStyle.AMERICAN, symbol="SYN") if args.american else sample_chain()
    print(save(chain, args.out))
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    from optarb.study import simulate_session

    for chain in simulate_session(n_snapshots=args.n, seed=args.seed):
        save(chain, args.out)
    print(f"wrote {args.n} synthetic snapshots under {args.out}")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    from optarb.providers.schwab import SchwabProvider

    provider = SchwabProvider(interactive=args.login)
    for i in range(args.count):
        for symbol in args.symbols:
            try:
                chain = provider.option_chain(
                    symbol, max_dte=args.max_dte, strike_count=args.strikes, dividend_yield=args.dividend_yield
                )
                print(save(chain, args.out))
            except Exception as exc:  # keep recording the other symbols
                print(f"{symbol}: {type(exc).__name__}: {exc}", file=sys.stderr)
        if i < args.count - 1:
            time.sleep(args.every * 60)
    return 0


def cmd_study(args: argparse.Namespace) -> int:
    from optarb.store import SnapshotStore
    from optarb.study import run, write_report

    store = SnapshotStore(args.store)
    result = run(store.chains(args.symbol), _config(args))
    print(write_report(result, args.out))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="optarb", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scan", help="run every check on one snapshot")
    p.add_argument("snapshot")
    p.add_argument("--all", action="store_true", help="list every package, not just the best per check and expiry")
    p.add_argument("--limit", type=int, default=25)
    _add_cost_args(p)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("forwards", help="implied forward and discount factor per expiry")
    p.add_argument("snapshot")
    p.set_defaults(func=cmd_forwards)

    p = sub.add_parser("sample", help="write the synthetic demo snapshot")
    p.add_argument("--out", default="data/sample")
    p.add_argument("--american", action="store_true", help="a clean American chain instead")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("simulate", help="write a synthetic intraday series for the study")
    p.add_argument("--out", default="data/simulated")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=11)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("record", help="record Schwab option chains to Parquet (market data only)")
    p.add_argument("symbols", nargs="+", help="e.g. SPY QQQ $XSP")
    p.add_argument("--out", default="data/recorded")
    p.add_argument("--every", type=float, default=5.0, help="minutes between snapshots")
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--max-dte", type=int, default=60)
    p.add_argument("--strikes", type=int, default=40, help="strikes around the money per expiry")
    p.add_argument("--dividend-yield", type=float, default=0.0)
    p.add_argument("--login", action="store_true", help="allow the interactive Schwab OAuth login")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("study", help="run the study over a folder of snapshots and write a report")
    p.add_argument("store")
    p.add_argument("--out", default="reports/study")
    p.add_argument("--symbol", default=None)
    _add_cost_args(p)
    p.set_defaults(func=cmd_study)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
