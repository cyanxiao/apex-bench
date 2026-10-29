"""Optional full replay integration tests. No network calls or real model claims."""

import os

import pytest

from apexbench.cache import PredictionCache
from apexbench.evaluation import evaluate
from apexbench.jev import InferenceBudget
from apexbench.playback import leaderboard_frame, ranked_teams
from apexbench.prediction import PredictionService
from apexbench.replay import Replay
from apexbench.state import reconstruct

SOURCE = os.getenv("APEXBENCH_TEST_REPLAY")
pytestmark = pytest.mark.skipif(
    not SOURCE, reason="Set APEXBENCH_TEST_REPLAY for full replay checks"
)


@pytest.fixture(scope="module")
def source():
    return Replay.load(SOURCE)


def test_supplied_alliance_timeline(source):
    assert len(source.data.summary.teams) == 20
    assert len(source.events) == 25841
    assert source.team("Alliance").placement == 1
    assert source.data.summary.duration == 1157
    at450 = reconstruct(source, "Alliance", 450)
    unlucky = next(p for p in at450.target_team.players if p.name == "Alliance_Unlucky")
    assert unlucky.status == "dead"
    assert at450.squads_remaining == 18
    at473 = reconstruct(source, "Alliance", 473)
    assert (
        next(p for p in at473.target_team.players if p.name == "Alliance_Unlucky").status == "alive"
    )
    assert reconstruct(source, "Alliance", 1157).target_team.kills == 13


def test_supplied_cached_forecast_evaluation(source, mock_api, tmp_path):
    provider, calls = mock_api
    with PredictionCache(tmp_path / "source.sqlite3") as cache:
        service = PredictionService(source, cache, provider, InferenceBudget(1))
        record, hit = service.predict("Alliance", 450)
        assert not hit
        assert service.predict("Alliance", 450)[1]
        report = evaluate(source, [record])
    assert len(calls) == 1
    assert len(record.forecast.probabilities) == 20
    assert report["mean_brier_score"] == pytest.approx(0.95)
    assert report["rows"][0]["actual_placement"] == 1


def test_supplied_top_ten_fits_small_terminal(source):
    import io

    from rich.console import Console

    state = reconstruct(source, "Alliance", 450)
    rows = ranked_teams(state, {}, 10)
    assert len(rows) == 10
    assert all(not team.eliminated for team, _ in rows)
    output = io.StringIO()
    Console(file=output, width=80, height=24).print(
        leaderboard_frame(
            state, {}, 10, 1, True, compact=True, budget=InferenceBudget(120, requests_used=34)
        )
    )
    assert len(output.getvalue().splitlines()) <= 24
    assert "q quit" in output.getvalue()
    assert "API requests: 34/120 used · 86 remaining" in output.getvalue()
