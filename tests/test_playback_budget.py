"""Request usage remains visible independently of forecast coverage."""

import io
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from apexbench.cache import PredictionCache
from apexbench.cli import app
from apexbench.jev import InferenceBudget, JevProvider
from apexbench.playback import play
from apexbench.prediction import PredictionService
from apexbench.replay import Replay

FIXTURE = Path(__file__).parent / "fixtures" / "replay.json"


@pytest.mark.parametrize("top", [False, True])
def test_failed_attempts_count_in_both_views(monkeypatch, tmp_path, top):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(500, json={"error": "mock failure"})
        )
    ) as client:
        provider = JevProvider("test-key", client=client)
        monkeypatch.setattr("apexbench.cli.JevProvider", lambda *a, **kw: provider)
        result = CliRunner().invoke(
            app,
            [
                "replay",
                str(FIXTURE),
                "--start",
                "25",
                "--end",
                "25",
                "--plain",
                "--live-predictions",
                "--max-requests",
                "3" if top else "1",
                "--cache",
                str(tmp_path / "failure.sqlite3"),
                *(["--top", "10"] if top else []),
            ],
        )
    assert result.exit_code == 2, result.output
    expected = "1/3 used · 2 remaining" if top else "1/1 used · 0 remaining"
    assert f"API requests: {expected}" in result.output
    assert "HTTP 500" in result.output


@pytest.mark.parametrize("top", [False, True])
def test_cached_playback_preserves_entire_request_budget(mock_api, monkeypatch, tmp_path, top):
    provider, calls = mock_api
    monkeypatch.setattr("apexbench.cli.JevProvider", lambda *a, **kw: provider)
    path = tmp_path / "cached.sqlite3"
    with PredictionCache(path) as cache:
        service = PredictionService(Replay.load(FIXTURE), cache, provider, InferenceBudget(3))
        for team in [2, 3, 4] if top else [2]:
            service.predict(team, 25)
    calls.clear()
    result = CliRunner().invoke(
        app,
        [
            "replay",
            str(FIXTURE),
            "--start",
            "25",
            "--end",
            "25",
            "--plain",
            "--max-requests",
            "120",
            "--cache",
            str(path),
            *(["--top", "10"] if top else []),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "API requests: 0/120 used · 120 remaining" in result.output
    assert calls == []


def test_interactive_counter_refreshes_after_each_team_and_survives_rewind(
    replay,
    mock_api,
    monkeypatch,
    tmp_path,
):
    provider, calls = mock_api
    keys = iter(["", "p", "b", "n", "q"])

    @contextmanager
    def keyboard():
        yield lambda: next(keys)

    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr("apexbench.playback.keyboard", keyboard)
    monkeypatch.setattr("apexbench.playback.sys.stdin.isatty", lambda: True)
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, width=80, height=24, color_system=None)
    with PredictionCache(tmp_path / "interactive.sqlite3") as cache:
        service = PredictionService(replay, cache, provider, InferenceBudget(3))
        play(service, 2, 25, 45, 20, 1, "team", True, False, console, top=2)
    text = output.getvalue()
    assert "API requests: 1/3 used · 2 remaining" in text
    assert "API requests: 2/3 used · 1 remaining" in text
    assert text.count("API requests: 3/3 used · 0 remaining") >= 3
    assert len(calls) == 3
