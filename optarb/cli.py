"""Command line for the screener.

``doctor`` checks the Schwab setup; ``record`` and ``session`` read Schwab option chains; ``import-schwab``
turns a saved Schwab JSON payload into a snapshot; ``scan``, ``forwards`` and ``greeks`` look at one
snapshot; ``study`` runs over a folder of them. Only ``record`` and ``session`` touch the network, and
only to read market data. Nothing here can place an order.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from optarb.arbitrage import ScanConfig, best_per_group, scan
from optarb.forwards import fit_all_forwards
from optarb.models import ExerciseStyle, utc_now
from optarb.providers.base import Exhausted
from optarb.snapshot import load, save

NEW_YORK = ZoneInfo("America/New_York")


def _config(args: argparse.Namespace) -> ScanConfig:
    return ScanConfig(
        commission_per_contract=args.commission,
        borrow_rate=args.borrow,
        min_net_edge=args.min_edge,
        reference_rate=args.rate,
        max_spread_pct=args.max_spread_pct,
        min_size=args.min_size,
        max_quote_age_seconds=args.max_quote_age,
    )


def _add_cost_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--commission", type=float, default=0.65, help="dollars per contract (default 0.65)")
    p.add_argument("--borrow", type=float, default=0.0, help="annual stock borrow fee for short stock legs")
    p.add_argument("--min-edge", type=float, default=0.01, help="minimum net edge in dollars per package")
    p.add_argument("--rate", type=float, default=None, help="reference funding rate, e.g. 0.045; enables rate checks")
    p.add_argument("--max-spread-pct", type=float, default=100.0, help="skip quotes wider than this percent of mid")
    p.add_argument("--min-size", type=int, default=0, help="contracts needed on the side traded (when sizes are known)")
    p.add_argument(
        "--max-quote-age", type=float, default=None, help="drop quotes older than this many seconds at the snapshot"
    )


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


def market_close(now: datetime) -> datetime:
    """4:00 pm New York time on the day of ``now``, in UTC. Holidays and half days are not handled."""
    local = now.astimezone(NEW_YORK)
    return local.replace(hour=16, minute=0, second=0, microsecond=0).astimezone(UTC)


def _save_raw(payload: dict, root: str | Path, symbol: str, asof: datetime) -> Path:
    safe = symbol.replace("$", "").replace("/", "_")
    path = Path(root) / "raw" / safe / f"{safe}_{asof:%Y%m%dT%H%M%SZ}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def record_loop(
    provider,
    symbols: list[str],
    out: str | Path,
    every_minutes: float = 5.0,
    count: int | None = 1,
    until: datetime | None = None,
    max_dte: int = 60,
    strike_count: int | None = 40,
    dividend_yield: float = 0.0,
    save_raw: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = utc_now,
    log=print,
) -> list[Path]:
    """Record every symbol, then wait, until ``count`` rounds are done or ``until`` has passed."""
    written: list[Path] = []
    exhausted: set[str] = set()
    rounds = 0
    while True:
        if until is not None and now() >= until:
            break
        for symbol in symbols:
            asof = now()
            try:
                payload, chain = provider.fetch(
                    symbol, max_dte=max_dte, strike_count=strike_count, dividend_yield=dividend_yield, asof=asof
                )
            except Exhausted:
                exhausted.add(symbol)
                continue
            except Exception as exc:  # keep recording the other symbols
                print(f"{symbol}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            if save_raw:
                _save_raw(payload, out, symbol, chain.asof)
            path = save(chain, out)
            written.append(path)
            note = "  (delayed quotes)" if chain.delayed else ""
            log(f"{path}  quotes={len(chain.quotes)}{note}")
        rounds += 1
        if exhausted >= set(symbols):  # a replayed source has nothing left
            break
        if count is not None and rounds >= count:
            break
        if until is not None and now() + timedelta(minutes=every_minutes) > until:
            break
        sleep(every_minutes * 60)
    return written


def _provider(args: argparse.Namespace):
    if args.source == "recorded":
        from optarb.providers.recorded import RecordedSource

        if not args.replay:
            raise SystemExit("--source recorded needs --replay PATH (a .json payload or a folder of them)")
        return RecordedSource(args.replay)
    from optarb.providers.schwab import SchwabProvider

    return SchwabProvider.from_env(token_path=args.token_path, interactive=args.login)


def _record(args: argparse.Namespace) -> list[Path]:
    if args.source == "recorded":  # replay every saved payload back to back, keeping their own timestamps
        return record_loop(
            _provider(args),
            args.symbols,
            args.out,
            count=None,
            dividend_yield=args.dividend_yield,
            save_raw=False,
            sleep=lambda _s: None,
        )
    until = market_close(utc_now()) if args.until_close else None
    if until is not None and utc_now() >= until:
        print("The regular session has closed for today; nothing recorded.")
        return []
    return record_loop(
        _provider(args),
        args.symbols,
        args.out,
        every_minutes=args.every,
        count=None if until is not None else args.count,
        until=until,
        max_dte=args.max_dte,
        strike_count=args.strikes,
        dividend_yield=args.dividend_yield,
        save_raw=args.save_raw,
    )


def cmd_record(args: argparse.Namespace) -> int:
    _record(args)
    return 0


def cmd_session(args: argparse.Namespace) -> int:
    """Record, then run the study (violations, IV surface, vendor Greeks) over what was just recorded."""
    from optarb.study import run, write_report

    written = _record(args)
    if not written:
        return 1
    chains = [load(p) for p in written]
    print(write_report(run(chains, _config(args), rate=args.rate), args.report))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report what the Schwab setup is missing. Prints only yes/no facts, never a key, secret or token."""
    from optarb.providers.schwab import check_setup

    c = check_setup(args.token_path)
    yn = lambda b: "yes" if b else "no"  # noqa: E731
    print(
        f"schwab-py installed      {yn(c.schwab_py_installed)}"
        + (f" ({c.schwab_py_version})" if c.schwab_py_version else "")
    )
    print(f"SCHWAB_API_KEY set       {yn(c.api_key_set)}")
    print(f"SCHWAB_APP_SECRET set    {yn(c.app_secret_set)}")
    age = f", {c.token_age_days:.1f} days old" if c.token_age_days is not None else ""
    print(f"token file present       {yn(c.token_file_exists)}{age}")
    problems = c.problems()
    if problems:
        print("\nTo fix:")
        for line in problems:
            print(f"  - {line}")
    else:
        print("\nReady to record.")
    return 0 if c.ready else 1


def cmd_import_schwab(args: argparse.Namespace) -> int:
    from optarb.providers.recorded import stamp_of
    from optarb.providers.schwab import parse_chain

    path = Path(args.payload)
    payload = json.loads(path.read_text())
    if args.asof:
        asof = datetime.fromisoformat(args.asof)
        asof = asof.replace(tzinfo=UTC) if asof.tzinfo is None else asof.astimezone(UTC)
    elif stamped := stamp_of(path):
        asof = stamped
    else:
        asof = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    symbol = args.symbol or payload.get("symbol")
    if not symbol:
        print("No symbol in the payload; pass --symbol.", file=sys.stderr)
        return 2
    chain = parse_chain(payload, symbol=symbol, asof=asof, dividend_yield=args.dividend_yield)
    print(save(chain, args.out))
    return 0


def cmd_greeks(args: argparse.Namespace) -> int:
    from optarb.greeks import analyze, surface_summary, vendor_comparison

    chain = load(args.snapshot)
    df = analyze(chain, rate=args.rate, otm_only=args.otm)
    print(
        f"{chain.symbol} {chain.asof:%Y-%m-%d %H:%M} UTC  source={chain.source}  style={chain.exercise_style.value}"
        f"  quotes priced={len(df)}" + ("  (delayed quotes)" if chain.delayed else "")
    )
    if df.empty:
        print("No two-sided quotes to price.")
        return 0
    src = ", ".join(sorted(df["rate_source"].unique()))
    print(f"rate: {df['rate'].median():.4%} from {src}; model: {', '.join(sorted(df['model'].unique()))}")
    with pd.option_context("display.width", 200, "display.float_format", "{:,.2f}".format):
        print("\nSurface (vol points)")
        print(surface_summary(df if args.otm else analyze(chain, rate=args.rate, otm_only=True)).to_string(index=False))
    vc = vendor_comparison(df)
    if not vc.empty:
        with pd.option_context("display.width", 200, "display.float_format", "{:,.4f}".format):
            print("\nAgainst the source's own IV and Greeks")
            print(vc.to_string(index=False))
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\n{args.csv}")
    return 0


def cmd_study(args: argparse.Namespace) -> int:
    from optarb.store import SnapshotStore
    from optarb.study import run, write_report

    store = SnapshotStore(args.store)
    result = run(store.chains(args.symbol), _config(args), greeks=not args.no_greeks, rate=args.rate)
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

    def recording_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("symbols", nargs="+", help="e.g. SPY QQQ $XSP")
        p.add_argument(
            "--source",
            choices=["schwab", "recorded"],
            default="schwab",
            help="schwab: live market data; recorded: replay saved Schwab JSON from --replay",
        )
        p.add_argument("--replay", default=None, help="with --source recorded: a .json payload or a folder of them")
        p.add_argument("--out", default="data/recorded")
        p.add_argument("--every", type=float, default=5.0, help="minutes between snapshots")
        p.add_argument("--count", type=int, default=1, help="rounds to record (ignored with --until-close)")
        p.add_argument("--until-close", action="store_true", help="keep recording until 4:00 pm New York time")
        p.add_argument("--max-dte", type=int, default=60)
        p.add_argument("--strikes", type=int, default=40, help="strikes around the money per expiry")
        p.add_argument("--dividend-yield", type=float, default=0.0)
        p.add_argument("--save-raw", action="store_true", help="also keep the raw Schwab JSON under OUT/raw")
        p.add_argument("--token-path", default=None, help="where schwab-py keeps its token (default ~/.optarb)")
        p.add_argument("--login", action="store_true", help="allow the interactive Schwab OAuth login")

    p = sub.add_parser("record", help="record Schwab option chains to Parquet (market data only)")
    recording_args(p)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("session", help="record Schwab chains, then write the study report for what was recorded")
    recording_args(p)
    p.add_argument("--report", default="reports/session")
    _add_cost_args(p)
    p.set_defaults(func=cmd_session)

    p = sub.add_parser("doctor", help="check the Schwab setup without printing any secret")
    p.add_argument("--token-path", default=None)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("import-schwab", help="turn a saved Schwab option-chain JSON into a snapshot")
    p.add_argument("payload")
    p.add_argument("--out", default="data/recorded")
    p.add_argument("--symbol", default=None)
    p.add_argument("--asof", default=None, help="ISO time of the quotes; default from the file name or its mtime")
    p.add_argument("--dividend-yield", type=float, default=0.0)
    p.set_defaults(func=cmd_import_schwab)

    p = sub.add_parser("greeks", help="implied vol and Greeks for one snapshot, against the source's own")
    p.add_argument("snapshot")
    p.add_argument("--rate", type=float, default=None, help="rate for American chains (default: the source's)")
    p.add_argument("--otm", action="store_true", help="out-of-the-money quotes only")
    p.add_argument("--csv", default=None, help="write the per-contract table here")
    p.set_defaults(func=cmd_greeks)

    p = sub.add_parser("study", help="run the study over a folder of snapshots and write a report")
    p.add_argument("store")
    p.add_argument("--out", default="reports/study")
    p.add_argument("--symbol", default=None)
    p.add_argument("--no-greeks", action="store_true", help="skip the IV surface and vendor Greek sections")
    _add_cost_args(p)
    p.set_defaults(func=cmd_study)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
