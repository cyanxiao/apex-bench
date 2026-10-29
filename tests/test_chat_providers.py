"""Chat models share the benchmark's validation, cache, budget, and CLI workflows."""

import copy
import io
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from rich.console import Console
from typer.testing import CliRunner

from apexbench.cache import PredictionCache, PredictionRecord, cache_key
from apexbench.cli import app
from apexbench.evaluation import evaluate, export_predictions, import_predictions
from apexbench.inference import (
    DEFAULT_MODEL,
    BudgetExceeded,
    InferenceBudget,
    PredictionError,
    api_request,
    canonical,
)
from apexbench.jev import JevProvider
from apexbench.jev import build_request as jev_request
from apexbench.playback import frame
from apexbench.prediction import PredictionService
from apexbench.providers import (
    CHAT_ENDPOINT,
    MODEL_EXAMPLES,
    OpenRouterChatProvider,
    build_request,
    parse_response,
    validate_model,
)
from apexbench.state import reconstruct

MODEL = "openai/gpt-4.1-mini"
CHAT_MODELS = [model for model in MODEL_EXAMPLES if model != DEFAULT_MODEL]
FIXTURE = Path(__file__).parent / "fixtures" / "replay.json"


def chat_response(model=MODEL, content=None, **choice_fields):
    return {
        "model": model,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content
                    if content is not None
                    else json.dumps({"probabilities": {"1": 0.5, "2": 0.3, "3": 0.2}}),
                },
                **choice_fields,
            }
        ],
        "usage": {"prompt_tokens": 400, "completion_tokens": 30},
    }


@pytest.mark.parametrize("model", CHAT_MODELS)
def test_models_use_chat_contract_and_roundtrip_records(replay, tmp_path, model):
    calls = []
    state = reconstruct(replay, 2, 25)

    def handler(request):
        assert str(request.url) == CHAT_ENDPOINT
        body = json.loads(request.content)
        calls.append(body)
        assert body["model"] == model
        assert "state" not in body and "questions" not in body
        assert json.loads(body["messages"][1]["content"]) == state.model_dump(mode="json")
        assert "not remembered results" in body["messages"][0]["content"]
        assert body["provider"] == {"require_parameters": True, "allow_fallbacks": False}
        schema = body["response_format"]["json_schema"]
        assert schema["strict"] is True
        assert schema["schema"]["properties"]["probabilities"]["required"] == ["1", "2", "3"]
        assert body["stream"] is False
        return httpx.Response(200, json=chat_response(model))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OpenRouterChatProvider("secret-chat-key", model, client=client)
        with PredictionCache(tmp_path / "cache.sqlite3") as cache:
            service = PredictionService(replay, cache, provider, InferenceBudget(1))
            record, hit = service.predict(2, 25)
            assert not hit
            assert service.predict("Alliance", 25) == (record, True)
            assert service.cached(2) == [record]
            assert len(calls) == service.budget.requests_used == 1
            assert record.forecast.probabilities == [0.5, 0.3, 0.2]
            assert record.response["usage"]["prompt_tokens"] == 400
            assert "secret-chat-key" not in record.model_dump_json()
            for suffix in ("json", "jsonl"):
                path = tmp_path / f"forecasts.{suffix}"
                export_predictions([record], path)
                imported = import_predictions(path)
                assert imported == [record]
                report = evaluate(replay, imported)
                assert report["count"] == 1
                assert report["rows"][0]["model"] == model
    # The same persisted cache is usable without a key or HTTP client.
    with PredictionCache(tmp_path / "cache.sqlite3") as cache:
        service = PredictionService(
            replay, cache, OpenRouterChatProvider(None, model), InferenceBudget(0)
        )
        assert service.predict(2, 25, cache_only=True) == (record, True)
    output = io.StringIO()
    Console(file=output, width=160).print(frame(state, record, 1, False))
    assert f"{model} @ 25s" in output.getvalue()


@pytest.mark.parametrize(
    "content",
    [
        "not JSON",
        '```json\n{"probabilities":{"1":1,"2":0,"3":0}}\n```',
        "null",
        "[]",
        '{"probabilities":{"1":0.5,"2":0.5}}',
        '{"probabilities":{"1":1,"2":0,"3":0,"4":0}}',
        '{"probabilities":[0.5,0.3,0.2]}',
        '{"probabilities":{"1":0.33,"2":0.33,"3":0.33}}',
        '{"probabilities":{"1":true,"2":0,"3":0}}',
        '{"probabilities":{"1":"0.5","2":0.3,"3":0.2}}',
        '{"probabilities":{"1":NaN,"2":0,"3":0}}',
        '{"probabilities":{"1":Infinity,"2":0,"3":0}}',
        '{"probabilities":{"1":-0.1,"2":0.8,"3":0.3}}',
        '{"probabilities":{"1":1.1,"2":0,"3":0}}',
        '{"probabilities":{"1":0.5,"2":0.3,"3":0.2},"extra":"unexpected"}',
    ],
)
def test_invalid_chat_distribution_is_not_cached(replay, tmp_path, content):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=chat_response(content=content))
        )
    ) as client:
        with PredictionCache(tmp_path / "cache.sqlite3") as cache:
            provider = OpenRouterChatProvider(
                "secret-key", MODEL, client=client, diagnostics_dir=tmp_path / "diagnostics"
            )
            service = PredictionService(replay, cache, provider, InferenceBudget(1))
            with pytest.raises(PredictionError, match="Invalid chat completion"):
                service.predict(2, 25)
            assert service.cached(2) == []
            assert service.budget.requests_used == 1
    saved = next((tmp_path / "diagnostics").glob("chat-*.json")).read_text()
    assert "secret-key" not in saved and "Authorization" not in saved
    assert json.loads(saved)["request"]["model"] == MODEL


@pytest.mark.parametrize(
    "body",
    [
        {},
        [],
        {"error": {"message": "upstream failure"}},
        {"model": MODEL, "choices": []},
        {"model": MODEL, "choices": [None]},
        chat_response(finish_reason="length"),
        chat_response(finish_reason="content_filter"),
        chat_response(message={"role": "assistant", "content": None, "refusal": "refused"}),
        chat_response(message={"role": "assistant", "content": None, "tool_calls": [{}]}),
        chat_response(message={"role": "assistant", "content": []}),
        chat_response(model="anthropic/claude-sonnet-4.5"),
        chat_response(model="openai/gpt-4.1-mini-other"),
    ],
)
def test_chat_completion_errors_rejected(replay, body):
    with pytest.raises(PredictionError, match="Invalid chat completion"):
        parse_response(body, build_request(reconstruct(replay, 2, 25), MODEL))


@pytest.mark.parametrize(
    ("requested", "resolved"),
    [
        (MODEL, MODEL + "-2025-04-14"),
        (MODEL, MODEL + "-20250414"),
        (MODEL + ":free", MODEL),
    ],
)
def test_resolved_snapshot_and_routing_variant(replay, requested, resolved):
    result = parse_response(
        chat_response(model=resolved), build_request(reconstruct(replay, 2, 25), requested)
    )
    assert result.resolved_model == resolved


def test_model_caches_are_separate_and_old_jev_requests_unchanged(replay, mock_api, tmp_path):
    jev, calls = mock_api
    state = reconstruct(replay, 2, 25)
    original = jev_request(state)
    assert build_request(state) == original
    with PredictionCache(tmp_path / "cache.sqlite3") as cache:
        jev_record, _ = PredictionService(replay, cache, jev, InferenceBudget(1)).predict(2, 25)
        chat = PredictionService(
            replay, cache, OpenRouterChatProvider(None, MODEL), InferenceBudget(0)
        )
        with pytest.raises(PredictionError, match="No cached"):
            chat.predict(2, 25, cache_only=True)
        assert chat.cached(2) == []
        assert cache_key(replay.fingerprint, build_request(state, MODEL)) != jev_record.cache_key
        assert len(calls) == 1
        assert PredictionService(replay, cache, JevProvider(None), InferenceBudget(0)).predict(
            2, 25, cache_only=True
        ) == (jev_record, True)


@pytest.mark.parametrize("status", [400, 401, 402, 429, 500])
def test_chat_http_failures_count_once_and_do_not_leak_secrets(replay, status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": "secret-key"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OpenRouterChatProvider("secret-key", MODEL, client=client)
        budget = InferenceBudget(3)
        with pytest.raises(PredictionError, match=f"HTTP {status}") as caught:
            provider.predict(build_request(reconstruct(replay, 2, 25), MODEL), budget)
        assert "secret-key" not in str(caught.value)
        assert len(calls) == budget.requests_used == 1


def test_chat_budgets_measure_api_payload_and_fail_before_spending(replay):
    request = build_request(reconstruct(replay, 2, 25), MODEL)
    size = len(canonical(api_request(request)).encode())
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=chat_response())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OpenRouterChatProvider("test", MODEL, client=client)
        for budget in (InferenceBudget(0), InferenceBudget(1, size - 1)):
            with pytest.raises(BudgetExceeded):
                provider.predict(request, budget)
            assert budget.requests_used == 0
        assert calls == []
        provider.predict(request, InferenceBudget(1, size))
        assert len(calls) == 1


def test_chat_timeout_and_non_json_response_have_diagnostics(replay, tmp_path):
    request = build_request(reconstruct(replay, 2, 25), MODEL)

    def timeout(request):
        raise httpx.ReadTimeout("private connection details", request=request)

    for handler, expected in [
        (timeout, "timed out"),
        (lambda r: httpx.Response(200, text="upstream unavailable"), "expected JSON"),
    ]:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            provider = OpenRouterChatProvider(
                "secret", MODEL, client=client, diagnostics_dir=tmp_path
            )
            budget = InferenceBudget(2)
            with pytest.raises(PredictionError, match=expected):
                provider.predict(request, budget)
            assert budget.requests_used == 1
    assert len(list(tmp_path.glob("chat-*.json"))) == 1


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4.1-mini",
        "",
        " openai/gpt-4.1-mini",
        "openai/model\n",
        "openrouter/auto",
        "typesafe/jev-latest",
        "https://example.com/model",
        None,
    ],
)
def test_invalid_model_ids(model):
    with pytest.raises(ValueError):
        validate_model(model)


def test_additional_openrouter_models_do_not_need_a_code_change():
    model = "another-vendor/new-model-1:free"
    assert OpenRouterChatProvider(None, model).model == model


def test_chat_forecast_tampering_rejected(replay):
    request = build_request(reconstruct(replay, 2, 25), MODEL)
    result = parse_response(chat_response(), request)
    record = PredictionRecord.create(replay.fingerprint, "fixture", request, result)
    changed = copy.deepcopy(record.model_dump())
    changed["forecast"]["probabilities"] = [1.0, 0.0, 0.0]
    with pytest.raises(ValidationError, match="raw chat completion response"):
        PredictionRecord.model_validate(changed)


def test_chat_cli_precompute_resume_replay_and_evaluate(replay, tmp_path, monkeypatch):
    runner = CliRunner()
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OpenRouterChatProvider("test", MODEL, client=client)
        monkeypatch.setattr("apexbench.cli.OpenRouterChatProvider", lambda *a, **kw: provider)
        cache = tmp_path / "cache.sqlite3"
        output = tmp_path / "predictions.jsonl"
        common = [str(FIXTURE), "--team", "2", "--model", MODEL, "--cache", str(cache)]
        precompute = [
            "precompute",
            *common,
            "--end",
            "25",
            "--interval",
            "25",
            "--max-requests",
            "1",
            "--output",
            str(output),
        ]
        result = runner.invoke(app, [*precompute, "--dry-run"])
        assert result.exit_code == 0, result.output
        size = len(
            canonical(api_request(build_request(reconstruct(replay, 2, 25), MODEL))).encode()
        )
        assert f"25s: missing, {size} request bytes" in result.output
        assert calls == []
        assert runner.invoke(app, precompute).exit_code == 2
        result = runner.invoke(app, precompute)
        assert result.exit_code == 0, result.output
        assert len(calls) == 2
        assert len(import_predictions(output)) == 2
        result = runner.invoke(app, ["predict", *common, "--at", "25", "--cache-only", "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["cache_hit"]
        result = runner.invoke(app, ["replay", *common, "--end", "25", "--step", "25", "--plain"])
        assert result.exit_code == 0, result.output
        assert MODEL in result.stdout and "Jev @" not in result.stdout
        for options in ([], ["--predictions", str(output)]):
            result = runner.invoke(app, ["evaluate", *common, *options, "--json"])
            assert result.exit_code == 0, result.output
            assert json.loads(result.stdout)["count"] == 2
        assert len(calls) == 2
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0, result.output
    for model in MODEL_EXAMPLES:
        assert model in result.stdout
