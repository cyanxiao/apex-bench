"""Content-addressed SQLite storage and portable prediction records."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from apexbench.inference import PROMPT_VERSION, ProviderResult, digest
from apexbench.models import Forecast, GameState, ObservationMode, SafeModel
from apexbench.providers import is_jev, parse_response


def cache_key(replay_sha256: str, request: dict, prompt_version: str = PROMPT_VERSION) -> str:
    return digest(
        {"replay_sha256": replay_sha256, "prompt_version": prompt_version, "request": request}
    )


class PredictionRecord(SafeModel):
    format_version: Literal["apexbench-prediction-v1"] = "apexbench-prediction-v1"
    cache_key: str
    replay_sha256: str
    game_id: str
    team_id: int
    timestamp: float = Field(ge=0)
    observation_mode: ObservationMode
    requested_model: str
    resolved_model: str
    prompt_version: str = PROMPT_VERSION
    created_at: str
    forecast: Forecast
    request: dict[str, Any]
    response: dict[str, Any]

    @model_validator(mode="after")
    def consistent(self):
        state = GameState.model_validate(self.request.get("state"))
        if (
            state.timestamp != self.timestamp
            or state.target_team.team_id != self.team_id
            or state.observation_mode != self.observation_mode
            or self.request.get("model") != self.requested_model
            or len(self.forecast.probabilities) != state.team_count
        ):
            raise ValueError("prediction provenance does not match its request")
        if self.cache_key != cache_key(self.replay_sha256, self.request, self.prompt_version):
            raise ValueError("prediction cache key does not match its contents")
        parsed = parse_response(self.response, self.request)
        if self.forecast != parsed.forecast or self.resolved_model != parsed.resolved_model:
            source = "Jev" if is_jev(self.requested_model) else "chat completion"
            raise ValueError(f"forecast does not match its raw {source} response")
        return self

    @classmethod
    def create(cls, replay_sha256: str, game_id: str, request: dict, result: ProviderResult):
        state = request["state"]
        return cls(
            cache_key=cache_key(replay_sha256, request),
            replay_sha256=replay_sha256,
            game_id=game_id,
            team_id=state["target_team"]["team_id"],
            timestamp=state["timestamp"],
            observation_mode=state["observation_mode"],
            requested_model=request["model"],
            resolved_model=result.resolved_model,
            created_at=datetime.now(UTC).isoformat(),
            forecast=result.forecast,
            request=request,
            response=result.response,
        )


class PredictionCache:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS apexbench_predictions (
            cache_key TEXT PRIMARY KEY, replay_sha256 TEXT NOT NULL, team_id INTEGER NOT NULL,
            timestamp REAL NOT NULL, observation_mode TEXT NOT NULL, model TEXT NOT NULL,
            record_json TEXT NOT NULL)""")
        self.connection.execute("""CREATE INDEX IF NOT EXISTS apexbench_lookup
            ON apexbench_predictions(replay_sha256, team_id, observation_mode, model, timestamp)""")
        self.connection.commit()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.connection.close()

    def get(self, key: str) -> PredictionRecord | None:
        row = self.connection.execute(
            "SELECT record_json FROM apexbench_predictions WHERE cache_key = ?", (key,)
        ).fetchone()
        return PredictionRecord.model_validate_json(row[0]) if row else None

    def put(self, record: PredictionRecord):
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO apexbench_predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.cache_key,
                    record.replay_sha256,
                    record.team_id,
                    record.timestamp,
                    record.observation_mode,
                    record.requested_model,
                    record.model_dump_json(),
                ),
            )

    def records(
        self, replay_sha256: str, team_id: int, mode: str, model: str
    ) -> list[PredictionRecord]:
        rows = self.connection.execute(
            """SELECT record_json FROM apexbench_predictions
            WHERE replay_sha256 = ? AND team_id = ? AND observation_mode = ? AND model = ?
            ORDER BY timestamp, cache_key""",
            (replay_sha256, team_id, mode, model),
        ).fetchall()
        return [PredictionRecord.model_validate_json(row[0]) for row in rows]
