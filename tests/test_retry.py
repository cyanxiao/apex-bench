"""Invalid responses remain rejected; explicit checkpoint retries can recover."""

import io
import json
from collections import Counter
from contextlib import contextmanager

import httpx
import pytest
from rich.console import Console

from apexbench.cache import PredictionCache
from apexbench.jev import (
    DEFAULT_MODEL,
    InferenceBudget,
    JevProvider,
    PredictionError,
    build_request,
)
from apexbench.playback import play
from apexbench.prediction import PredictionService
from apexbench.state import reconstruct


def response_body(probabilities, model=DEFAULT_MODEL):
    return {
        "model": model,
        "answers": {
            "placement": {
                "type": "choice",
                "choice": "1",
                "probabilities": probabilities,
            }
        },
    }


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (response_body({"1": 0.33, "2": 0.33, "3": 0.33}), "sum to 0.99"),
        (response_body({"1": 0.5, "2": 0.5}), "missing 3"),
        (
            response_body({"1": 0.5, "2": "0.5", "3": 0.0}),
            "placement 2 probability must be a number",
        ),
        (
            response_body({"1": 1.0, "2": 0.0, "3": 0.0}, "typesafe/jev-1.14"),
            "unexpected resolved model",
        ),
    ],
)
def test_precise_validation_error_and_safe_diagnostic(replay, tmp_path, body, reason):
    directory = tmp_path / "diagnostics"
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    ) as client:
        provider = JevProvider("test-secret-key", client=client, diagnostics_dir=directory)
        with PredictionCache(tmp_path / "cache.sqlite3") as cache:
            service = PredictionService(replay, cache, provider, InferenceBudget(1))
            with pytest.raises(PredictionError, match=reason) as caught:
                service.predict(2, 25)
            assert service.cached(2) == []
            assert service.budget.requests_used == 1
    assert "Response details saved" in str(caught.value)
    files = list(directory.glob("*.json"))
    assert len(files) == 1
    contents = files[0].read_text()
    assert "test-secret-key" not in contents
    assert "Authorization" not in contents
    saved = json.loads(contents)
    assert saved["request"]["state"]["timestamp"] == 25
    assert json.loads(saved["response_text"]) == body


def test_non_json_response_is_saved_without_being_cached(replay, tmp_path):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, text="<html>upstream problem</html>")
        )
    ) as client:
        provider = JevProvider("test-secret", client=client, diagnostics_dir=tmp_path)
        with pytest.raises(PredictionError, match="expected JSON"):
            provider.predict(build_request(reconstruct(replay, 2, 25)), InferenceBudget(1))
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["response_text"] == "<html>upstream problem</html>"


def test_c_key_retries_failed_team_and_preserves_successful_cache(replay, tmp_path, monkeypatch):
    calls = []
    counts = Counter()

    def handler(request):
        team = json.loads(request.content)["state"]["target_team"]["team_id"]
        calls.append(team)
        counts[team] += 1
        probabilities = (
            {"1": 0.33, "2": 0.33, "3": 0.33}
            if team == 3 and counts[team] == 1
            else {"1": 1 / 3, "2": 1 / 3, "3": 1 / 3}
        )
        return httpx.Response(200, json=response_body(probabilities))

    keys = iter(["", "", "c", "q"])

    @contextmanager
    def keyboard():
        yield lambda: next(keys)

    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr("apexbench.playback.keyboard", keyboard)
    monkeypatch.setattr("apexbench.playback.sys.stdin.isatty", lambda: True)
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system=None, width=120, height=40)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = JevProvider(
            "test-secret", client=client, diagnostics_dir=tmp_path / "diagnostics"
        )
        with PredictionCache(tmp_path / "retry.sqlite3") as cache:
            service = PredictionService(replay, cache, provider, InferenceBudget(4))
            assert play(service, 2, 25, 26, 1, 1, "team", True, False, console, top=2)
            assert all(len(service.cached(team)) == 1 for team in [2, 3, 4])
            assert service.budget.requests_used == 4
    assert calls == [2, 3, 3, 4]
    text = output.getvalue()
    assert "Beta at 25s" in text
    assert "sum to 0.99" in text
    assert "c retry" in text
    assert "Forecasts @ 25s: 3/3" in text
    assert len(list((tmp_path / "diagnostics").glob("*.json"))) == 1
