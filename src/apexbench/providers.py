"""Model routing and structured placement forecasts through OpenRouter chat completions."""

from __future__ import annotations

import json
import re

from apexbench import jev
from apexbench.inference import (
    DEFAULT_MODEL,
    OpenRouterProvider,
    PredictionError,
    ProviderResult,
    canonical,
)
from apexbench.models import Forecast, GameState

CHAT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL_EXAMPLES = {
    DEFAULT_MODEL: "Jev (default; Decisions API)",
    "openai/gpt-4.1-mini": "GPT-4.1 Mini",
    "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "deepseek/deepseek-chat-v3.1": "DeepSeek V3.1",
    "qwen/qwen3-235b-a22b-2507": "Qwen3 235B Instruct",
}


def is_jev(model: str) -> bool:
    return model.startswith("typesafe/jev-")


def validate_model(model: str) -> str:
    if not isinstance(model, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*(?::[a-z0-9][a-z0-9._-]*)?", model
    ):
        raise ValueError("Use an explicit OpenRouter model ID such as openai/gpt-4.1-mini")
    if is_jev(model):
        return jev.validate_model(model)
    if model.startswith("openrouter/"):
        raise ValueError("Choose a specific model; automatic model routers are not supported")
    return model


def build_request(state: GameState, model: str = DEFAULT_MODEL) -> dict:
    validate_model(model)
    if is_jev(model):
        return jev.build_request(state, model)
    # Explicit placement keys avoid relying on array ordering or count in model output.
    placements = {
        str(i): {"type": "number", "description": f"Probability of final placement {i}."}
        for i in range(1, state.team_count + 1)
    }
    return {
        "model": model,
        "state": state.model_dump(mode="json"),
        "messages": [
            {
                "role": "system",
                "content": jev.PLACEMENT_INSTRUCTIONS
                + " Return only a JSON object with a probabilities object keyed by placement."
                + " Include every placement from 1 to team_count. Probabilities must be numbers"
                + " in [0,1] and sum to 1 within 0.000001. Do not round away probability mass.",
            },
            {"role": "user", "content": canonical(state.model_dump(mode="json"))},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "placement_forecast",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "probabilities": {
                            "type": "object",
                            "properties": placements,
                            "required": list(placements),
                            "additionalProperties": False,
                        }
                    },
                    "required": ["probabilities"],
                    "additionalProperties": False,
                },
            },
        },
        "provider": {"require_parameters": True, "allow_fallbacks": False},
        "stream": False,
    }


def _same_model(requested: str, resolved: str) -> bool:
    # Routing variants can be omitted in OpenRouter's response. Date snapshots are
    # accepted, but another family/version must never satisfy the requested model.
    base = requested.split(":", 1)[0]
    return resolved in (requested, base) or bool(
        re.fullmatch(re.escape(base) + r"-(?:\d{8}|\d{4}-\d{2}-\d{2})", resolved)
    )


def parse_chat_response(body: dict, request: dict) -> ProviderResult:
    try:
        if not isinstance(body, dict):
            raise ValueError("response must be a JSON object")
        if "error" in body:
            raise ValueError("provider returned an error")
        choices = body["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("expected exactly one completion")
        choice = choices[0]
        if choice["finish_reason"] != "stop":
            raise ValueError("completion did not finish normally (truncated, refused, or failed)")
        message = choice["message"]
        if (
            message.get("refusal")
            or message.get("tool_calls")
            or message.get("role") != "assistant"
        ):
            raise ValueError("expected an assistant forecast without refusal or tool calls")
        content = message["content"]
        if not isinstance(content, str):
            raise ValueError("message content must be a JSON string")
        answer = json.loads(content)
        if not isinstance(answer, dict) or set(answer) != {"probabilities"}:
            raise ValueError("forecast must contain only probabilities")
        probabilities = answer["probabilities"]
        count = request["state"]["team_count"]
        expected = {str(i) for i in range(1, count + 1)}
        if not isinstance(probabilities, dict) or set(probabilities) != expected:
            raise ValueError("probabilities must contain exactly one key per placement")
        forecast = Forecast(probabilities=[probabilities[str(i)] for i in range(1, count + 1)])
        resolved = validate_model(body["model"])
        requested = validate_model(request["model"])
        if not _same_model(requested, resolved):
            raise ValueError("unexpected resolved model")
        return ProviderResult(forecast, body, resolved)
    except (KeyError, IndexError) as exc:
        raise PredictionError(f"Invalid chat completion response: missing field {exc}") from exc
    except (TypeError, AttributeError) as exc:
        raise PredictionError("Invalid chat completion response: incorrect field types") from exc
    except ValueError as exc:
        raise PredictionError(f"Invalid chat completion response: {exc}") from exc


def parse_response(body: dict, request: dict) -> ProviderResult:
    """Use the requested model to validate live responses and imported records."""
    model = validate_model(request.get("model"))
    if is_jev(model):
        return jev.parse_response(body, request)
    return parse_chat_response(body, request)


class OpenRouterChatProvider(OpenRouterProvider):
    endpoint = CHAT_ENDPOINT
    response_name = "chat completion"
    diagnostic_prefix = "chat"
    parse_response = staticmethod(parse_chat_response)

    @staticmethod
    def validate_model(model: str) -> str:
        validate_model(model)
        if is_jev(model):
            raise ValueError("Jev models require JevProvider and the Decisions API")
        return model
