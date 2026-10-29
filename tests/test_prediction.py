import copy

import httpx
import pytest
from pydantic import ValidationError

from apexbench.cache import PredictionCache, PredictionRecord, cache_key
from apexbench.jev import (
    DEFAULT_MODEL,
    ENDPOINT,
    BudgetExceeded,
    InferenceBudget,
    JevProvider,
    PredictionError,
    build_request,
)
from apexbench.models import Forecast
from apexbench.prediction import PredictionService, checkpoints, latest_before
from apexbench.state import reconstruct


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.5, 0.6],
        [-0.1, 1.1],
        [float("nan"), 1.0],
        [float("inf"), 0.0],
        [True, 0.0],
        ["0.5", "0.5"],
        [1.0],
        [0.0, 0.0],
    ],
)
def test_invalid_distributions(probabilities):
    with pytest.raises(ValidationError):
        Forecast(probabilities=probabilities)


def test_native_decisions_contract_and_cache(replay, mock_api, tmp_path):
    provider, calls = mock_api
    budget = InferenceBudget(1)
    path = tmp_path / "cache.sqlite3"
    with PredictionCache(path) as cache:
        service = PredictionService(replay, cache, provider, budget)
        first, hit = service.predict(2, 25)
        assert not hit
        again, hit = service.predict("Alliance", 25)
        assert hit and first == again
        assert len(calls) == 1 and budget.requests_used == 1
        assert calls[0]["questions"]["placement"]["type"] == "choice"
        assert set(calls[0]["questions"]["placement"]["criteria"]) == {"1", "2", "3"}
        assert "messages" not in calls[0]
        assert "game_id" not in calls[0]["state"]
        assert "test-key" not in first.model_dump_json()
        with pytest.raises(BudgetExceeded):
            service.predict(2, 26)
        with pytest.raises(BudgetExceeded):
            service.predict(2, 25, "oracle")
    # Cache survives process/service lifetime and needs no API key or remaining budget.
    with PredictionCache(path) as cache:
        service = PredictionService(replay, cache, JevProvider(None), InferenceBudget(0))
        assert service.predict(2, 25, cache_only=True) == (first, True)
        with pytest.raises(PredictionError, match="No cached"):
            service.predict(2, 24, cache_only=True)


def test_key_includes_source_model_mode_prompt_and_state(replay):
    request = build_request(reconstruct(replay, 2, 25))
    original = cache_key(replay.fingerprint, request)
    assert original != cache_key("other-source", request)
    assert original != cache_key(replay.fingerprint, request, "new-prompt")
    assert original != cache_key(replay.fingerprint, build_request(reconstruct(replay, 2, 26)))
    assert original != cache_key(
        replay.fingerprint, build_request(reconstruct(replay, 2, 25, "oracle"))
    )
    assert original != cache_key(
        replay.fingerprint, build_request(reconstruct(replay, 2, 25), "typesafe/jev-1.14")
    )


def test_zero_request_and_input_budgets_make_no_calls(replay, mock_api):
    provider, calls = mock_api
    request = build_request(reconstruct(replay, 2, 25))
    for budget in (InferenceBudget(0), InferenceBudget(10, 1)):
        with pytest.raises(BudgetExceeded):
            provider.predict(request, budget)
    assert calls == []


@pytest.mark.parametrize("status", [401, 402, 429, 500])
def test_http_failures_count_once_and_do_not_leak_key(replay, status):
    attempts = []

    def handler(request):
        assert str(request.url) == ENDPOINT
        attempts.append(request)
        return httpx.Response(status, json={"error": "untrusted error"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = JevProvider("secret-api-key", client=client)
        budget = InferenceBudget(3)
        with pytest.raises(PredictionError, match=f"HTTP {status}") as exc:
            provider.predict(build_request(reconstruct(replay, 2, 25)), budget)
        assert "secret-api-key" not in str(exc.value)
        assert len(attempts) == budget.requests_used == 1


@pytest.mark.parametrize(
    "response",
    [
        {},
        {
            "answers": {
                "placement": {"type": "choice", "choice": "1", "probabilities": {"1": 1.0}}
            },
            "model": DEFAULT_MODEL,
        },
        {
            "answers": {
                "placement": {
                    "type": "choice",
                    "choice": "1",
                    "probabilities": {"1": 0.5, "2": 0.5, "3": 0.5},
                }
            },
            "model": DEFAULT_MODEL,
        },
        {
            "answers": {
                "placement": {
                    "type": "choice",
                    "choice": "1",
                    "probabilities": {"1": 1.0, "2": 0.0, "3": 0.0},
                }
            },
            "model": "other/model",
        },
    ],
)
def test_invalid_response_not_cached(replay, response, tmp_path):
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response))
    ) as c:
        with PredictionCache(tmp_path / "invalid.sqlite3") as cache:
            service = PredictionService(
                replay, cache, JevProvider("test", client=c), InferenceBudget(1)
            )
            with pytest.raises(PredictionError, match="Invalid Jev"):
                service.predict(2, 25)
            assert service.cached(2) == []


def test_timeout_is_counted_without_retry(replay):
    def handler(request):
        raise httpx.ReadTimeout("secret details", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        budget = InferenceBudget(1)
        with pytest.raises(PredictionError, match="timed out"):
            JevProvider("test", client=client).predict(
                build_request(reconstruct(replay, 2, 25)), budget
            )
        assert budget.requests_used == 1


def test_forecast_provenance_tampering(replay, mock_api, tmp_path):
    provider, _ = mock_api
    with PredictionCache(tmp_path / "cache.sqlite3") as cache:
        record, _ = PredictionService(replay, cache, provider, InferenceBudget(1)).predict(2, 25)
        changed = copy.deepcopy(record.model_dump())
        changed["timestamp"] = 24
        with pytest.raises(ValidationError):
            PredictionRecord.model_validate(changed)
        changed = copy.deepcopy(record.model_dump())
        changed["forecast"]["probabilities"] = [1.0, 0.0, 0.0]
        with pytest.raises(ValidationError, match="raw Jev response"):
            PredictionRecord.model_validate(changed)


@pytest.mark.parametrize(("team", "time"), [(3, 90), (2, 110), (2, 120)])
def test_resolved_outcomes_refused(replay, mock_api, tmp_path, team, time):
    provider, calls = mock_api
    with PredictionCache(tmp_path / "cache.sqlite3") as cache:
        service = PredictionService(replay, cache, provider, InferenceBudget(10))
        with pytest.raises(PredictionError, match="resolved"):
            service.predict(team, time)
    assert calls == []


def test_checkpoint_selection_never_uses_future(replay, mock_api, tmp_path):
    provider, _ = mock_api
    with PredictionCache(tmp_path / "cache.sqlite3") as cache:
        service = PredictionService(replay, cache, provider, InferenceBudget(2))
        early, _ = service.predict(2, 25)
        late, _ = service.predict(2, 60)
        assert latest_before([early, late], 24) is None
        assert latest_before([early, late], 59) == early
        assert latest_before([early, late], 60) == late
        assert latest_before([early, late], 25) == early


def test_checkpoints_and_jev_only():
    assert checkpoints(0, 125, 60) == [0, 60, 120]
    assert checkpoints(1, 1.3, 0.1) == [1, 1.1, 1.2, 1.3]
    for args in [(0, 10, 0), (10, 0, 1), (0, 10, float("nan"))]:
        with pytest.raises(ValueError):
            checkpoints(*args)
    for model in ["openai/any", "typesafe/jev-router", "~typesafe/jev-latest"]:
        with pytest.raises(ValueError, match="only pinned"):
            JevProvider(None, model)
