from datetime import UTC, datetime, timedelta

import pandas as pd

from optarb.models import ExerciseStyle
from optarb.providers.synthetic import make_chain, plant_vertical
from optarb.snapshot import load, save
from optarb.store import SnapshotStore
from optarb.study import persistence, run, simulate_session, summarize, write_report

T0 = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)


def _series(planted_at: set[int], n: int = 6):
    chains = []
    for i in range(n):
        c = make_chain(asof=T0 + timedelta(minutes=5 * i), seed=i)
        if i in planted_at:
            plant_vertical(c)
        chains.append(c)
    return chains


def test_snapshot_round_trip(tmp_path):
    chain = make_chain(exercise_style=ExerciseStyle.AMERICAN)
    loaded = load(save(chain, tmp_path))
    assert loaded.asof == chain.asof and loaded.exercise_style is chain.exercise_style
    assert sorted(loaded.quotes, key=repr) == sorted(chain.quotes, key=repr)


def test_store_reads_a_folder_with_duckdb(tmp_path):
    for c in _series(set(), n=3):
        save(c, tmp_path)
    store = SnapshotStore(tmp_path)
    inv = store.inventory()
    assert inv.loc[0, "snapshots"] == 3
    assert len(list(store.chains())) == 3
    n = store.query("SELECT count(*) AS n FROM quotes WHERE \"right\" = 'C'").loc[0, "n"]
    assert n == 3 * len(make_chain().quotes) // 2


def test_persistence_counts_consecutive_runs():
    result = run(_series({1, 2, 3, 5}))
    eps = persistence(result)
    vert = eps[eps["check"] == "vertical_width"].sort_values("start")
    assert list(vert["snapshots"]) == [3, 1]
    assert vert["minutes"].iloc[0] == 10.0


def test_clean_series_has_no_violations_and_an_empty_summary():
    result = run(_series(set()))
    assert result.opportunities.empty
    assert summarize(result).empty


def test_report_is_written_and_labeled_synthetic(tmp_path):
    result = run(simulate_session(n_snapshots=8))
    path = write_report(result, tmp_path)
    text = path.read_text()
    assert "Synthetic data" in text and "Results by check" in text
    for name in ("opportunities.csv", "episodes.csv", "box_rates.csv", "summary.csv"):
        assert isinstance(pd.read_csv(tmp_path / name), pd.DataFrame)
