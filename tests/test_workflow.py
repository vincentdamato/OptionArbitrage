"""End to end on Schwab-shaped data: record -> snapshot -> store -> scan -> Greeks -> study, with no network."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from optarb.cli import main, market_close, record_loop
from optarb.providers.schwab import SchwabProvider, check_setup
from optarb.snapshot import from_frame, load, to_frame
from optarb.store import SnapshotStore
from optarb.study import run, write_report

FIXTURE = Path(__file__).parent / "fixtures" / "schwab_spy_chain.json"
ASOF = datetime(2026, 11, 20, 15, 30, tzinfo=UTC)


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeClient:
    """Stands in for a schwab-py client: returns the fixture for every option-chain request."""

    def __init__(self):
        self.calls = []

    def get_option_chain(self, symbol, **kwargs):
        self.calls.append((symbol, kwargs))
        return _Response(json.loads(FIXTURE.read_text()))


class Clock:
    def __init__(self, start):
        self.t = start

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += timedelta(seconds=seconds)


def test_record_loop_writes_snapshots_and_raw_json(tmp_path):
    client, clock = FakeClient(), Clock(ASOF)
    written = record_loop(
        SchwabProvider(client),
        ["SPY"],
        tmp_path,
        every_minutes=5,
        count=3,
        save_raw=True,
        sleep=clock.sleep,
        now=clock,
        log=lambda *_: None,
    )
    assert len(written) == 3 and len(client.calls) == 3
    assert client.calls[0][1]["include_underlying_quote"] is True
    assert len(list((tmp_path / "raw" / "SPY").glob("*.json"))) == 3
    assert SnapshotStore(tmp_path).inventory().loc[0, "snapshots"] == 3


def test_record_loop_stops_at_the_close(tmp_path):
    clock = Clock(market_close(ASOF) - timedelta(minutes=12))
    written = record_loop(
        SchwabProvider(FakeClient()),
        ["SPY"],
        tmp_path,
        every_minutes=5,
        count=None,
        until=market_close(ASOF),
        sleep=clock.sleep,
        now=clock,
        log=lambda *_: None,
    )
    assert len(written) == 3  # at -12, -7 and -2 minutes; the next round would start after 4 pm


def test_market_close_is_four_pm_new_york():
    assert market_close(ASOF) == datetime(2026, 11, 20, 21, 0, tzinfo=UTC)  # EST
    assert market_close(datetime(2026, 7, 1, 15, tzinfo=UTC)).hour == 20  # EDT


def test_snapshot_round_trip_keeps_sizes_times_and_vendor_greeks(tmp_path):
    path = record_loop(SchwabProvider(FakeClient()), ["SPY"], tmp_path, now=lambda: ASOF, log=lambda *_: None)[0]
    chain = load(path)
    assert chain.delayed is False and chain.vendor_rate == pytest.approx(0.0425)
    q = next(q for q in chain.quotes if q.vendor_delta is not None)
    assert q.bid_size > 0 and q.quote_time.tzinfo is not None and q.vendor_gamma is not None


def test_old_snapshots_without_the_new_columns_still_load(tmp_path):
    chain = load(record_loop(SchwabProvider(FakeClient()), ["SPY"], tmp_path, now=lambda: ASOF, log=print)[0])
    new_cols = ["bid_size", "ask_size", "quote_time", "delayed", "vendor_rate"] + [
        f"vendor_{g}" for g in ("delta", "gamma", "theta", "vega", "rho")
    ]
    old = to_frame(chain).drop(columns=new_cols)
    back = from_frame(old)
    assert len(back.quotes) == len(chain.quotes)
    assert back.vendor_rate is None and back.quotes[0].bid_size == 0 and back.quotes[0].quote_time is None


def test_study_over_recorded_snapshots_reports_surface_and_vendor_greeks(tmp_path):
    clock = Clock(ASOF)
    record_loop(
        SchwabProvider(FakeClient()),
        ["SPY"],
        tmp_path / "data",
        every_minutes=5,
        count=2,
        sleep=clock.sleep,
        now=clock,
        log=lambda *_: None,
    )
    result = run(SnapshotStore(tmp_path / "data").chains())
    assert len(result.snapshots) == 2
    assert set(result.surface["asof"]) == {ASOF, ASOF + timedelta(minutes=5)}
    assert not result.vendor.empty
    report = write_report(result, tmp_path / "rep").read_text()
    assert "Implied volatility surface over the session" in report
    assert "against the data source's" in report
    assert (tmp_path / "rep" / "surface.csv").exists() and (tmp_path / "rep" / "vendor_greeks.csv").exists()


def test_synthetic_study_recovers_the_generating_vol(tmp_path):
    from optarb.greeks import vendor_comparison
    from optarb.study import simulate_session

    result = run(simulate_session(n_snapshots=3, seed=2))
    assert not result.surface.empty
    vc = vendor_comparison(result.vendor).set_index("measure")
    assert list(vc.index) == ["iv"]  # synthetic quotes carry the vol they were built from, and no Greeks
    assert vc.loc["iv", "median_abs_gap"] < 0.01


def test_cli_import_greeks_scan_study(tmp_path, capsys):
    raw = tmp_path / "SPY_20261120T153000Z.json"
    raw.write_text(FIXTURE.read_text())
    assert main(["import-schwab", str(raw), "--out", str(tmp_path / "data"), "--dividend-yield", "0.012"]) == 0
    snap = next((tmp_path / "data").rglob("*.parquet"))
    assert load(snap).asof == ASOF  # time taken from the file name
    assert main(["greeks", str(snap), "--csv", str(tmp_path / "g.csv")]) == 0
    out = capsys.readouterr().out
    assert "rate: 4.2500% from vendor" in out and "Against the source's own IV and Greeks" in out
    assert (tmp_path / "g.csv").exists()
    assert main(["scan", str(snap), "--max-quote-age", "300", "--min-size", "1"]) == 0
    assert main(["study", str(tmp_path / "data"), "--out", str(tmp_path / "rep")]) == 0


def test_doctor_never_prints_secret_values(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("SCHWAB_API_KEY", "key-value-that-must-not-print")
    monkeypatch.setenv("SCHWAB_APP_SECRET", "secret-value-that-must-not-print")
    token = tmp_path / "token.json"
    token.write_text("{}")
    rc = main(["doctor", "--token-path", str(token)])
    out = capsys.readouterr().out
    assert "must-not-print" not in out
    assert check_setup(str(token)).api_key_set
    assert rc == (0 if check_setup(str(token)).schwab_py_installed else 1)


def test_doctor_lists_what_is_missing(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("SCHWAB_API_KEY", raising=False)
    monkeypatch.delenv("SCHWAB_APP_SECRET", raising=False)
    assert main(["doctor", "--token-path", str(tmp_path / "none.json")]) == 1
    out = capsys.readouterr().out
    assert "SCHWAB_API_KEY is not set" in out and "no token file yet" in out


def _replay_folder(tmp_path, n=3):
    folder = tmp_path / "raw"
    folder.mkdir()
    for i in range(n):
        stamp = ASOF + timedelta(minutes=5 * i)
        (folder / f"SPY_{stamp:%Y%m%dT%H%M%SZ}.json").write_text(FIXTURE.read_text())
    return folder


def test_recorded_source_replays_in_time_order_and_stops(tmp_path):
    from optarb.providers.recorded import RecordedSource

    src = RecordedSource(_replay_folder(tmp_path))
    written = record_loop(src, ["SPY"], tmp_path / "data", count=None, sleep=lambda _s: None, log=lambda *_: None)
    assert [load(p).asof for p in written] == [ASOF + timedelta(minutes=5 * i) for i in range(3)]


def test_session_on_recorded_chains_runs_the_whole_workflow(tmp_path, capsys):
    folder = _replay_folder(tmp_path)
    rc = main(
        [
            "session",
            "SPY",
            "--source",
            "recorded",
            "--replay",
            str(folder),
            "--out",
            str(tmp_path / "data"),
            "--report",
            str(tmp_path / "rep"),
            "--dividend-yield",
            "0.012",
            "--max-quote-age",
            "300",
        ]
    )
    assert rc == 0
    report = (tmp_path / "rep" / "report.md").read_text()
    assert "Snapshots: 3" in report and "Implied volatility surface over the session" in report
    assert "against the data source's" in report
