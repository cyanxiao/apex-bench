"""Shared inference contracts, request budgets, and OpenRouter HTTP transport."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from apexbench.models import Forecast

PROMPT_VERSION = "apexbench-placement-v1"
DEFAULT_MODEL = "typesafe/jev-1.13"


def canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def api_request(request: dict) -> dict:
    """Chat records retain structured state for provenance, outside the API payload."""
    if "messages" in request:
        return {key: value for key, value in request.items() if key != "state"}
    return request


class PredictionError(ValueError):
    pass


class BudgetExceeded(PredictionError):
    pass


@dataclass
class InferenceBudget:
    max_requests: int = 10
    max_input_bytes: int = 32_000
    requests_used: int = 0

    def __post_init__(self):
        if self.max_requests < 0 or self.max_input_bytes <= 0:
            raise ValueError("request budget must be nonnegative and input budget positive")

    def reserve(self, request: dict):
        size = len(canonical(request).encode())
        if size > self.max_input_bytes:
            raise BudgetExceeded(
                f"Request has {size} bytes; --max-input-bytes is {self.max_input_bytes}"
            )
        if self.requests_used >= self.max_requests:
            raise BudgetExceeded(f"API request budget exhausted ({self.max_requests})")
        self.requests_used += 1  # Count failures too. There are no hidden HTTP retries.


@dataclass
class ProviderResult:
    forecast: Forecast
    response: dict
    resolved_model: str


class PlacementProvider(Protocol):
    model: str

    def predict(self, request: dict, budget: InferenceBudget) -> ProviderResult: ...


class OpenRouterProvider:
    """Shared single-attempt transport; adapters supply validation and response parsing."""

    endpoint: str
    response_name: str
    diagnostic_prefix: str

    @staticmethod
    def validate_model(model: str) -> str:
        raise NotImplementedError

    @staticmethod
    def parse_response(body: dict, request: dict) -> ProviderResult:
        raise NotImplementedError

    def __init__(
        self,
        api_key: str | None,
        model: str = DEFAULT_MODEL,
        timeout: float = 30,
        client: httpx.Client | None = None,
        diagnostics_dir: Path | None = None,
    ):
        self.api_key = api_key
        self.model = self.validate_model(model)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.timeout = timeout
        self.client = client
        self.diagnostics_dir = diagnostics_dir

    def _invalid_response(
        self, request: dict, response: httpx.Response, error: PredictionError
    ) -> PredictionError:
        if self.diagnostics_dir is None:
            return error
        created = datetime.now(UTC)
        path = self.diagnostics_dir / (
            f"{self.diagnostic_prefix}-{created.strftime('%Y%m%dT%H%M%S%fZ')}-{digest(request)[:12]}.json"
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "application": "ApexBench",
                        "format_version": "apexbench-failed-response-v1",
                        "created_at": created.isoformat(),
                        "error": str(error),
                        "request": request,
                        "response_text": response.text,
                    },
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            return PredictionError(f"{error}\nCould not save response diagnostics.")
        return PredictionError(f"{error}\nResponse details saved to {path}.")

    def predict(self, request: dict, budget: InferenceBudget) -> ProviderResult:
        if not self.api_key:
            raise PredictionError(
                "Set OPENROUTER_API_KEY to generate predictions; "
                "cached predictions work without a key"
            )
        if request.get("model") != self.model:
            raise PredictionError("request model does not match provider")
        payload = api_request(request)
        budget.reserve(payload)
        headers = {"Authorization": f"Bearer {self.api_key}", "X-Title": "ApexBench"}
        try:
            if self.client is None:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self.endpoint, json=payload, headers=headers)
            else:
                response = self.client.post(
                    self.endpoint, json=payload, headers=headers, timeout=self.timeout
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            raise PredictionError(
                f"OpenRouter returned HTTP {code}; check API access, credits, "
                "model availability or rate limits. No automatic retry was made."
            ) from None
        except httpx.RequestError:
            raise PredictionError(
                "OpenRouter connection failed or timed out. "
                "No automatic retry was made; the attempt consumed one request."
            ) from None
        try:
            body = response.json()
        except ValueError:
            error = PredictionError(f"Invalid {self.response_name} response: expected JSON")
            raise self._invalid_response(request, response, error) from None
        try:
            return self.parse_response(body, request)
        except PredictionError as exc:
            raise self._invalid_response(request, response, exc) from None
