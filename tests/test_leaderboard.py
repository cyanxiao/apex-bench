import io
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from apexbench.cache import PredictionCache, PredictionRecord
from apexbench.cli import app
from apexbench.jev import DEFAULT_MODEL, InferenceBudget, ProviderResult, build_request
from apexbench.models import Forecast
from apexbench.playback import leaderboard_frame, ranked_teams
from apexbench.prediction import PredictionService
from apexbench.state import reconstruct

FIXTURE = Path(__file__).parent / "fixtures" / "replay.json"


def prediction(replay, team, timestamp, probabilities):
    request = build_request(reconstruct(replay, team, timestamp))
    response = {
        "model": DEFAULT_MODEL,
        "answers": {
            "placement": {
                "type": "choice",
                "choice": str(probabilities.index(max(probabilities)) + 1),
                "probabilities": {str(i): p for i, p in enumerate(probabilities, 1)},
            }
        },
    }
    return PredictionRecord.create(
        replay.fingerprint,
        replay.data.summary.game_id,
        request,
        ProviderResult(Forecast(probabilities=probabilities), response, DEFAULT_MODEL),
    )


def test_rank_by_expected_placement_and_never_use_future(replay):
    records = {
        2: [prediction(replay, 2, 25, [0.1, 0.2, 0.7])],
        3: [prediction(replay, 3, 60, [1.0, 0.0, 0.0])],
        4: [prediction(replay, 4, 25, [0.6, 0.4, 0.0])],
    }
    at35 = ranked_teams(reconstruct(replay, 2, 35), records, 3)
    assert [team.team_id for team, _ in at35] == [4, 2, 3]
    assert at35[-1][1] is None
    at60 = ranked_teams(reconstruct(replay, 2, 60), records, 2)
    assert [team.team_id for team, _ in at60] == [3, 4]
    # Rewinding restores the earlier order and hides the future record again.
    assert ranked_teams(reconstruct(replay, 2, 35), records, 3) == at35


def test_missing_forecasts_use_public_kills_and_tie_by_id(replay):
    rows = ranked_teams(reconstruct(replay, 2, 35), {}, 10)
    assert [team.team_id for team, _ in rows] == [3, 2, 4]
    assert all(record is None for _, record in rows)
    rows = ranked_teams(reconstruct(replay, 2, 0), {}, 10)
    assert [team.team_id for team, _ in rows] == [2, 3, 4]


def test_eliminated_squads_are_excluded_even_with_a_good_forecast(replay):
    records = {3: [prediction(replay, 3, 25, [1.0, 0.0, 0.0])]}
    assert ranked_teams(reconstruct(replay, 2, 89), records, 1)[0][0].team_id == 3
    assert [t.team_id for t, _ in ranked_teams(reconstruct(replay, 2, 90), records, 10)] == [2, 4]


def test_partial_coverage_and_age_are_visible(replay):
    state = reconstruct(replay, 2, 35)
    records = {2: [prediction(replay, 2, 25, [0.1, 0.2, 0.7])]}
    output = io.StringIO()
    Console(file=output, width=80).print(leaderboard_frame(state, records, 10, 1, True))
    text = output.getvalue()
    assert "Forecasts: 1/3" in text
    assert "Partial forecast coverage" in text
    assert "25s / 10.00s" in text
    assert "missing" in text
    assert all(name in text for name in ["Alliance", "Beta", "Gamma"])


def test_top_mode_cache_only_and_input_validation(mock_api, monkeypatch, tmp_path):
    provider, calls = mock_api
    monkeypatch.setattr("apexbench.cli.JevProvider", lambda *a, **kw: provider)
    runner = CliRunner()
    args = [
        "replay",
        str(FIXTURE),
        "--top",
        "10",
        "--start",
        "35",
        "--end",
        "35",
        "--plain",
        "--cache",
        str(tmp_path / "top.sqlite3"),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 2, result.output
    assert "Top 10 teams" in result.output
    assert "Forecasts @ 35s: 0/3" in result.output
    assert "Ranking withheld" in result.output
    assert "Replay stopped at 35s" in result.output
    assert calls == []
    # The multi-team view ignores the single-team selector.
    result = runner.invoke(app, args + ["--team", "not-in-this-match"])
    assert result.exit_code == 2, result.output
    assert "not-in-this-match" not in result.output
    args[3] = "0"
    assert runner.invoke(app, args).exit_code == 2


def test_top_live_shared_budget_and_resume(mock_api, monkeypatch, tmp_path):
    provider, calls = mock_api
    monkeypatch.setattr("apexbench.cli.JevProvider", lambda *a, **kw: provider)
    runner = CliRunner()
    args = [
        "replay",
        str(FIXTURE),
        "--top",
        "2",
        "--start",
        "25",
        "--end",
        "25",
        "--plain",
        "--live-predictions",
        "--max-requests",
        "2",
        "--cache",
        str(tmp_path / "live-top.sqlite3"),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 2, result.output
    assert len(calls) == 0  # Do not spend on a checkpoint that cannot be completed.
    assert "Forecasts @ 25s: 0/3" in result.output
    assert "Insufficient API request budget" in result.output
    assert "API requests: 0/2 used · 2 remaining" in result.output
    args[args.index("--max-requests") + 1] = "3"
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert len(calls) == 3
    assert "Forecasts @ 25s: 3/3" in result.output
    assert "Partial forecast" not in result.output
    assert "API requests: 3/3 used · 0 remaining" in result.output
    # Fully cached checkpoints still play when no requests are allowed.
    args[args.index("--max-requests") + 1] = "0"
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert len(calls) == 3
    assert "API requests: 0/0 used · 0 remaining" in result.output


def test_top_live_does_not_forecast_eliminated_teams(mock_api, monkeypatch, tmp_path):
    provider, calls = mock_api
    monkeypatch.setattr("apexbench.cli.JevProvider", lambda *a, **kw: provider)
    result = CliRunner().invoke(
        app,
        [
            "replay",
            str(FIXTURE),
            "--top",
            "10",
            "--start",
            "95",
            "--end",
            "95",
            "--plain",
            "--live-predictions",
            "--max-requests",
            "10",
            "--cache",
            str(tmp_path / "eliminated.sqlite3"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert {call["state"]["target_team"]["team_id"] for call in calls} == {2, 4}
    assert "Forecasts @ 95s: 2/2" in result.output


def test_top_uses_cached_forecasts_only_for_selected_observation_mode(replay, mock_api, tmp_path):
    provider, _ = mock_api
    with PredictionCache(tmp_path / "modes.sqlite3") as cache:
        service = PredictionService(replay, cache, provider, InferenceBudget(1))
        service.predict(2, 25, "oracle")
        assert service.cached(2, "team") == []
        assert len(service.cached(2, "oracle")) == 1


def test_invalid_limit(replay):
    with pytest.raises(ValueError, match="positive"):
        ranked_teams(reconstruct(replay, 2, 0), {}, 0)
