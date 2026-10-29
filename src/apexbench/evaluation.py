"""Multiclass scoring and self-contained, reproducible JSON/JSONL/CSV exports."""

from __future__ import annotations

import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from apexbench.cache import PredictionRecord, cache_key
from apexbench.jev import PROMPT_VERSION, build_request
from apexbench.models import Forecast
from apexbench.replay import Replay
from apexbench.state import reconstruct

EPSILON = 1e-15


def scores(forecast: Forecast, actual: int) -> dict[str, float]:
    if not 1 <= actual <= len(forecast.probabilities):
        raise ValueError("actual placement is outside the forecast's class range")
    return {
        "brier_score": math.fsum(
            (p - int(i == actual)) ** 2 for i, p in enumerate(forecast.probabilities, 1)
        ),
        "log_loss": -math.log(max(EPSILON, forecast.probabilities[actual - 1])),
    }


def export_predictions(records: list[PredictionRecord], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        text = "".join(r.model_dump_json() + "\n" for r in records)
    elif path.suffix.lower() == ".json":
        text = (
            json.dumps(
                {
                    "application": "ApexBench",
                    "format_version": "apexbench-export-v1",
                    "predictions": [r.model_dump(mode="json") for r in records],
                },
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
    else:
        raise ValueError("prediction export must end in .json or .jsonl")
    path.write_text(text, encoding="utf-8")


def import_predictions(path: Path) -> list[PredictionRecord]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [
            PredictionRecord.model_validate_json(line) for line in text.splitlines() if line.strip()
        ]
    body = json.loads(text)
    if isinstance(body, dict) and "predictions" in body:
        body = body["predictions"]
    elif isinstance(body, dict) and "cache_key" in body:
        body = [body]
    if not isinstance(body, list):
        raise ValueError("expected an ApexBench prediction export or JSONL records")
    return [PredictionRecord.model_validate(row) for row in body]


def evaluate(replay: Replay, records: list[PredictionRecord]) -> dict:
    if not records:
        raise ValueError(
            "no matching predictions to evaluate; run precompute or supply --predictions"
        )
    rows, seen = [], set()
    cohorts = {
        (r.team_id, r.observation_mode, r.requested_model, r.prompt_version) for r in records
    }
    if len(cohorts) != 1:
        raise ValueError("evaluate one team, observation mode, model and prompt version at a time")
    for record in sorted(records, key=lambda r: r.timestamp):
        if record.cache_key in seen:
            continue  # Do not weight duplicated imports twice.
        seen.add(record.cache_key)
        if record.replay_sha256 != replay.fingerprint:
            raise ValueError("prediction replay SHA256 does not match the supplied file")
        if record.game_id != replay.data.summary.game_id:
            raise ValueError("prediction game ID does not match the supplied file")
        if record.prompt_version != PROMPT_VERSION:
            raise ValueError("unsupported prompt version; use the matching ApexBench version")
        state = reconstruct(replay, record.team_id, record.timestamp, record.observation_mode)
        expected = build_request(state, record.requested_model)
        if cache_key(replay.fingerprint, expected) != record.cache_key:
            raise ValueError("prediction input does not match reconstructed historical state")
        if (
            state.target_team.eliminated
            or state.squads_remaining <= 1
            or record.timestamp >= replay.data.summary.duration
        ):
            raise ValueError("resolved-outcome predictions are excluded from forecasting metrics")
        actual = replay.team(record.team_id).placement
        if actual is None:
            raise ValueError("replay has no final placement for this team")
        row = {
            "cache_key": record.cache_key,
            "replay_sha256": replay.fingerprint,
            "game_id": replay.data.summary.game_id,
            "team_id": record.team_id,
            "team_name": state.target_team.team_name,
            "timestamp": record.timestamp,
            "observation_mode": record.observation_mode,
            "model": record.requested_model,
            "resolved_model": record.resolved_model,
            "prompt_version": record.prompt_version,
            "actual_placement": actual,
            "expected_placement": record.forecast.expected_placement,
            "actual_probability": record.forecast.probabilities[actual - 1],
            **scores(record.forecast, actual),
        }
        row.update({f"p_{i}": p for i, p in enumerate(record.forecast.probabilities, 1)})
        rows.append(row)
    count = len(rows)
    return {
        "application": "ApexBench",
        "format_version": "apexbench-evaluation-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "replay_sha256": replay.fingerprint,
        "count": count,
        "metric_definitions": {
            "brier_score": "sum over placements of (p_k - 1[k=actual])^2; range [0,2]",
            "log_loss": "-ln(max(p_actual, epsilon)); natural logarithm",
            "epsilon": EPSILON,
            "aggregation": "unweighted mean over unique checkpoints of one team/mode/model",
            "warning": (
                "Checkpoints from one match are correlated; this is not a multi-match estimate."
            ),
        },
        "mean_brier_score": math.fsum(r["brier_score"] for r in rows) / count,
        "mean_log_loss": math.fsum(r["log_loss"] for r in rows) / count,
        "zero_probabilities_clipped": sum(r["actual_probability"] < EPSILON for r in rows),
        "rows": rows,
        "predictions": list({r.cache_key: r.model_dump(mode="json") for r in records}.values()),
    }


def export_evaluation(report: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report["rows"][0]))
            writer.writeheader()
            writer.writerows(report["rows"])
    elif path.suffix.lower() == ".json":
        path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    else:
        raise ValueError("evaluation export must end in .json or .csv")
