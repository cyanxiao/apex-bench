"""Replay must stop at the first incomplete scheduled forecast checkpoint."""

import io
from contextlib import contextmanager

import httpx
import pytest
from rich.console import Console

from apexbench.cache import PredictionCache
from apexbench.jev import InferenceBudget, JevProvider
from apexbench.playback import play
from apexbench.playback_session import PlaybackSession
from apexbench.prediction import PredictionService
from apexbench.replay import Replay


@pytest.mark.parametrize("top", [None, 2])
def test_sparse_cache_stops_at_first_missing_second(replay, mock_api, tmp_path, top):
    provider, calls = mock_api
    with PredictionCache(tmp_path / "sparse.sqlite3") as cache:
        seed = PredictionService(replay, cache, provider, InferenceBudget(6))
        for timestamp in [25, 27]:
            for team in [2] if top is None else [2, 3, 4]:
                seed.predict(team, timestamp)
        calls.clear()
        service = PredictionService(replay, cache, provider, InferenceBudget(0))
        output = io.StringIO()
        complete = play(
            service, 2, 25, 27, 1, 1, "team", False, True, Console(file=output, width=100), top=top
        )
        assert not complete
        assert "Replay stopped at 26s" in output.getvalue()
        assert "00027.0s" not in output.getvalue()
        assert calls == []
        session = PlaybackSession(service, 2, 25, 27, 1, "team", False, top)
        blocked = session.prepare(26)
        assert blocked.blocked
        assert blocked.forecasts == {}  # Neither the older nor the future cache entry qualifies.


def test_eight_of_ten_cached_and_one_request_left_spends_nothing(raw, mock_api, tmp_path):
    provider, calls = mock_api
    for team_id in range(5, 12):
        raw["summary"]["teams"].append(
            {
                "teamId": team_id,
                "teamName": f"Extra {team_id}",
                "players": [{"name": f"Player {team_id}", "teamId": team_id}],
            }
        )
    replay = Replay.from_dict(raw)
    with PredictionCache(tmp_path / "ten.sqlite3") as cache:
        seed = PredictionService(replay, cache, provider, InferenceBudget(8))
        for team_id in range(2, 10):
            seed.predict(team_id, 25)
        calls.clear()
        service = PredictionService(replay, cache, provider, InferenceBudget(1))
        session = PlaybackSession(service, 2, 25, 25, 1, "team", True, 10)
        snapshot = session.prepare(25)
        assert snapshot.blocked
        assert len(snapshot.forecasts) == 8 and len(snapshot.required) == 10
        assert "Need 2 requests; 1 remaining" in snapshot.reason
        assert "Extra 10, Extra 11" in snapshot.reason
        assert calls == [] and service.budget.requests_used == 0


@pytest.mark.parametrize("live", [False, True])
def test_complete_cache_plays_with_no_budget(replay, mock_api, tmp_path, live):
    provider, calls = mock_api
    with PredictionCache(tmp_path / "complete.sqlite3") as cache:
        seed = PredictionService(replay, cache, provider, InferenceBudget(9))
        for timestamp in [25, 26, 27]:
            for team in [2, 3, 4]:
                seed.predict(team, timestamp)
        calls.clear()
        service = PredictionService(replay, cache, provider, InferenceBudget(0))
        output = io.StringIO()
        assert play(
            service, 2, 25, 27, 1, 1, "team", live, True, Console(file=output, width=100), top=2
        )
        assert "00027.0s" in output.getvalue()
        assert "Replay stopped" not in output.getvalue()
        assert calls == [] and service.budget.requests_used == 0


def test_live_fills_only_missing_exact_checkpoint_forecasts(replay, mock_api, tmp_path):
    provider, calls = mock_api
    with PredictionCache(tmp_path / "fill.sqlite3") as cache:
        seed = PredictionService(replay, cache, provider, InferenceBudget(2))
        for team in [2, 3]:
            seed.predict(team, 25)
        calls.clear()
        service = PredictionService(replay, cache, provider, InferenceBudget(1))
        session = PlaybackSession(service, 2, 25, 26, 1, "team", True, 2)
        snapshot = session.prepare(25)
        assert not snapshot.blocked and len(snapshot.forecasts) == 3
        assert len(calls) == 1 and calls[0]["state"]["target_team"]["team_id"] == 4
        # One complete checkpoint covers its step interval.
        assert not session.prepare(25.5).blocked
        assert session.prepare(26).blocked  # The next checkpoint requires fresh forecasts.
        assert len(calls) == 1


@pytest.mark.parametrize("budget", [3, 6])
def test_provider_failure_stops_without_automatic_retry(replay, tmp_path, budget):
    calls = []

    def failure(request):
        calls.append(request)
        return httpx.Response(500, json={"error": "test failure"})

    with httpx.Client(transport=httpx.MockTransport(failure)) as client:
        with PredictionCache(tmp_path / "failure.sqlite3") as cache:
            service = PredictionService(
                replay, cache, JevProvider("test", client=client), InferenceBudget(budget)
            )
            session = PlaybackSession(service, 2, 25, 25, 1, "team", True, 2)
            assert "HTTP 500" in session.prepare(25).reason
            assert "HTTP 500" in session.prepare(25).reason
            assert len(calls) == 1  # Drawing a paused frame cannot silently spend more money.
            assert session.prepare(25, retry=True).blocked
            assert len(calls) == (2 if budget == 6 else 1)


def test_retry_reloads_cache_filled_by_another_run(replay, mock_api, tmp_path):
    provider, calls = mock_api
    with PredictionCache(tmp_path / "reload.sqlite3") as cache:
        service = PredictionService(replay, cache, provider, InferenceBudget(0))
        session = PlaybackSession(service, 2, 25, 25, 1, "team", False)
        assert session.prepare(25).blocked
        other = PredictionService(replay, cache, provider, InferenceBudget(1))
        other.predict(2, 25)
        assert not session.prepare(25, retry=True).blocked
        assert len(calls) == 1 and service.budget.requests_used == 0


@pytest.mark.parametrize(("team", "timestamp", "top"), [(3, 90, None), (2, 110, 10), (2, 120, 10)])
def test_resolved_outcomes_need_no_predictions(replay, mock_api, tmp_path, team, timestamp, top):
    provider, calls = mock_api
    with PredictionCache(tmp_path / "resolved.sqlite3") as cache:
        service = PredictionService(replay, cache, provider, InferenceBudget(0))
        snapshot = PlaybackSession(
            service, team, timestamp, timestamp, 1, "team", False, top
        ).prepare(timestamp)
        assert not snapshot.blocked
        assert snapshot.required == []
        assert calls == []


def test_fast_interactive_playback_cannot_skip_a_gap(replay, mock_api, tmp_path, monkeypatch):
    provider, calls = mock_api
    keys = iter(["", "", "", "q"])

    @contextmanager
    def keyboard():
        yield lambda: next(keys)

    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr("apexbench.playback.keyboard", keyboard)
    monkeypatch.setattr("apexbench.playback.sys.stdin.isatty", lambda: True)
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system=None, width=100, height=40)
    with PredictionCache(tmp_path / "fast.sqlite3") as cache:
        seed = PredictionService(replay, cache, provider, InferenceBudget(2))
        seed.predict(2, 25)
        seed.predict(2, 27)
        calls.clear()
        service = PredictionService(replay, cache, provider, InferenceBudget(0))
        assert not play(service, 2, 25, 28, 1, 1024, "team", False, False, console)
        assert "Replay stopped at 26s" in output.getvalue()
        assert "00027.0s" not in output.getvalue()
        assert "PAUSED" in output.getvalue()
        assert calls == []
