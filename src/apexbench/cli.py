"""ApexBench command line entry point."""

from __future__ import annotations

import json
import math
import os
import sqlite3
from enum import StrEnum
from functools import wraps
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from apexbench import __version__
from apexbench.cache import PredictionCache
from apexbench.evaluation import (
    evaluate as evaluate_records,
)
from apexbench.evaluation import export_evaluation, export_predictions, import_predictions
from apexbench.jev import DEFAULT_MODEL, InferenceBudget, JevProvider
from apexbench.playback import play
from apexbench.prediction import PredictionService, checkpoints
from apexbench.replay import Replay
from apexbench.state import reconstruct

app = typer.Typer(
    name="apexbench",
    help="ApexBench — Apex Legends placement benchmarking with Jev.",
    no_args_is_help=True,
    invoke_without_command=True,
    pretty_exceptions_enable=False,
)
console = Console()
errors = Console(stderr=True)

ReplayPath = Annotated[
    Path,
    typer.Argument(
        help="Local Apex Legends replay JSON file.", exists=True, dir_okay=False, readable=True
    ),
]
TeamOption = Annotated[str, typer.Option("--team", "-t", help="Team name or numeric ID.")]
CacheOption = Annotated[
    Path, typer.Option("--cache", envvar="APEXBENCH_CACHE", help="SQLite prediction cache.")
]
ModelOption = Annotated[str, typer.Option("--model", help="Pinned Jev model ID; Jev only in v0.1.")]
OutputOption = Annotated[Path | None, typer.Option("--output", "-o", help="Export file path.")]
DEFAULT_CACHE = Path(".apexbench/predictions.sqlite3")


class Mode(StrEnum):
    team = "team"
    oracle = "oracle"


ModeOption = Annotated[Mode, typer.Option("--mode", help="Oracle exposes all observer telemetry.")]


def guarded(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, OSError, sqlite3.Error) as exc:
            errors.print(Text(f"ApexBench: {exc}", style="red"))
            raise typer.Exit(1) from None

    return wrapped


def parse_time(value: str) -> float:
    """Seconds, MM:SS or HH:MM:SS; never silently accept NaN or malformed clocks."""
    try:
        parts = value.split(":")
        if not 1 <= len(parts) <= 3:
            raise ValueError
        numbers = [float(part) for part in parts]
        if any(not math.isfinite(n) or n < 0 for n in numbers):
            raise ValueError
        if len(numbers) > 1:
            if any(n >= 60 for n in numbers[1:]) or any(n != int(n) for n in numbers[:-1]):
                raise ValueError
        return sum(n * 60**i for i, n in enumerate(reversed(numbers)))
    except ValueError:
        raise ValueError(f"Invalid timestamp {value!r}; use seconds, MM:SS or HH:MM:SS") from None


def bounds(replay: Replay, start: str, end: str | None, interval: float):
    first = parse_time(start)
    last = parse_time(end) if end is not None else replay.data.summary.duration
    if last > replay.data.summary.duration:
        raise ValueError("end exceeds replay duration")
    return checkpoints(first, last, interval), first, last


def service_for(
    replay: Replay,
    cache: PredictionCache,
    model: str,
    max_requests: int,
    max_input_bytes: int,
    timeout: float = 30,
) -> PredictionService:
    return PredictionService(
        replay,
        cache,
        JevProvider(
            os.getenv("OPENROUTER_API_KEY"),
            model,
            timeout=timeout,
            diagnostics_dir=cache.path.parent / "diagnostics",
        ),
        InferenceBudget(max_requests, max_input_bytes),
    )


def emit_json(value):
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


@app.callback()
def root(version: Annotated[bool, typer.Option("--version", is_eager=True)] = False):
    if version:
        typer.echo(f"ApexBench {__version__}")
        raise typer.Exit()


@app.command("inspect")
@guarded
def inspect_replay(
    replay_file: ReplayPath, json_output: Annotated[bool, typer.Option("--json")] = False
):
    """Inspect metadata, teams, final results and fields; never makes an API call."""
    report = Replay.load(replay_file).inspection()
    if json_output:
        emit_json(report)
        return
    console.print(Text(f"ApexBench | {report['map']} | {report['duration_seconds']:g}s"))
    console.print(Text(f"Game: {report['game_id']}\nSHA256: {report['sha256']}"))
    table = Table("ID", "Team", "Final placement")
    for team in report["teams"]:
        table.add_row(str(team["id"]), Text(team["name"]), str(team["placement"] or "unknown"))
    console.print(table)
    console.print_json(data=report["sections"])
    for warning in report["warnings"]:
        console.print(Text(warning, style="yellow"))
    console.print(report["notice"], style="dim")


@app.command()
@guarded
def state(
    replay_file: ReplayPath,
    team: TeamOption = "Alliance",
    at: Annotated[str, typer.Option("--at", help="Seconds, MM:SS or HH:MM:SS.")] = "0",
    mode: ModeOption = Mode.team,
    output: OutputOption = None,
):
    """Reconstruct historical observations as JSON, excluding future information."""
    value = reconstruct(Replay.load(replay_file), team, parse_time(at), mode.value)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(value.model_dump_json(indent=2) + "\n", encoding="utf-8")
    emit_json(value.model_dump(mode="json"))


@app.command()
@guarded
def predict(
    replay_file: ReplayPath,
    team: TeamOption = "Alliance",
    at: Annotated[str, typer.Option("--at")] = "0",
    mode: ModeOption = Mode.team,
    cache: CacheOption = DEFAULT_CACHE,
    model: ModelOption = DEFAULT_MODEL,
    cache_only: Annotated[bool, typer.Option("--cache-only")] = False,
    max_requests: Annotated[int, typer.Option("--max-requests", min=0)] = 1,
    max_input_bytes: Annotated[int, typer.Option("--max-input-bytes", min=1)] = 32_000,
    timeout: Annotated[float, typer.Option("--timeout", min=0.1)] = 30,
    output: OutputOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
):
    """Obtain a validated Jev distribution; reuse the exact cached request first."""
    if output and output.suffix.lower() not in (".json", ".jsonl"):
        raise ValueError("prediction export must end in .json or .jsonl")
    replay = Replay.load(replay_file)
    with PredictionCache(cache) as storage:
        service = service_for(replay, storage, model, max_requests, max_input_bytes, timeout)
        record, hit = service.predict(team, parse_time(at), mode.value, cache_only)
        if output:
            export_predictions([record], output)
        if json_output:
            emit_json({"cache_hit": hit, "predictions": [record.model_dump(mode="json")]})
        else:
            console.print(
                Text(
                    f"ApexBench | {replay.team(team).team_name} @ {record.timestamp:g}s | "
                    f"{record.observation_mode} | {'cache hit' if hit else 'new Jev prediction'}"
                )
            )
            table = Table("Placement", "Probability")
            for i, probability in enumerate(record.forecast.probabilities, 1):
                table.add_row(str(i), f"{probability:.4%}")
            console.print(table)
            console.print(f"Expected placement: {record.forecast.expected_placement:.3f}")
            console.print(Text(f"Resolved model: {record.resolved_model}"))


@app.command()
@guarded
def precompute(
    replay_file: ReplayPath,
    team: TeamOption = "Alliance",
    start: Annotated[str, typer.Option("--start")] = "0",
    end: Annotated[str | None, typer.Option("--end")] = None,
    interval: Annotated[float, typer.Option("--interval", min=0.001)] = 60,
    mode: ModeOption = Mode.team,
    cache: CacheOption = DEFAULT_CACHE,
    model: ModelOption = DEFAULT_MODEL,
    max_requests: Annotated[int, typer.Option("--max-requests", min=0)] = 10,
    max_input_bytes: Annotated[int, typer.Option("--max-input-bytes", min=1)] = 32_000,
    timeout: Annotated[float, typer.Option("--timeout", min=0.1)] = 30,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    output: OutputOption = None,
):
    """Generate interval forecasts with resumable caching and a strict request budget."""
    from apexbench.cache import cache_key
    from apexbench.jev import BudgetExceeded, build_request, canonical

    if output and output.suffix.lower() not in (".json", ".jsonl"):
        raise ValueError("prediction export must end in .json or .jsonl")
    replay = Replay.load(replay_file)
    times, _, _ = bounds(replay, start, end, interval)
    records, planned = [], []
    with PredictionCache(cache) as storage:
        service = service_for(replay, storage, model, max_requests, max_input_bytes, timeout)
        # Preflight every checkpoint before spending money or writing an export.
        for timestamp in times:
            snapshot = reconstruct(replay, team, timestamp, mode.value)
            if (
                timestamp >= replay.data.summary.duration
                or snapshot.target_team.eliminated
                or snapshot.squads_remaining <= 1
            ):
                continue
            request = build_request(snapshot, model)
            key = cache_key(replay.fingerprint, request)
            planned.append((timestamp, storage.get(key), len(canonical(request).encode())))
        misses = sum(record is None for _, record, _ in planned)
        console.print(
            f"ApexBench: {len(planned)} eligible checkpoints, "
            f"{len(planned) - misses} cached, {misses} missing; "
            f"API request limit {max_requests}."
        )
        if dry_run:
            for timestamp, record, size in planned:
                console.print(
                    f"{timestamp:g}s: {'cached' if record else 'missing'}, {size} request bytes"
                )
            return
        stopped = False
        for timestamp, record, _ in planned:
            if record is not None:
                records.append(record)
                continue
            try:
                record, _ = service.predict(team, timestamp, mode.value)
                records.append(record)
                console.print(f"Cached Jev forecast @ {timestamp:g}s")
            except BudgetExceeded as exc:
                console.print(Text(str(exc) + "; progress saved. Rerun to resume.", style="yellow"))
                stopped = True
                # Cached later checkpoints remain available in partial exports.
                records.extend(r for t, r, _ in planned if t > timestamp and r is not None)
                break
        if output:
            export_predictions(records, output)
        console.print(
            f"Available: {len(records)}/{len(planned)} forecasts; "
            f"API attempts: {service.budget.requests_used}."
        )
        if stopped:
            raise typer.Exit(2)


@app.command()
@guarded
def replay(
    replay_file: ReplayPath,
    team: TeamOption = "Alliance",
    top: Annotated[
        int | None,
        typer.Option(
            "--top", min=1, help="Show top N active teams by expected placement; overrides --team."
        ),
    ] = None,
    start: Annotated[str, typer.Option("--start")] = "0",
    end: Annotated[str | None, typer.Option("--end")] = None,
    step: Annotated[
        float,
        typer.Option(
            "--step",
            min=0.001,
            help="Forecast checkpoint interval in game seconds; also the keyboard seek step.",
        ),
    ] = 30,
    speed: Annotated[float, typer.Option("--speed", min=0.01)] = 10,
    mode: ModeOption = Mode.team,
    cache: CacheOption = DEFAULT_CACHE,
    model: ModelOption = DEFAULT_MODEL,
    live_predictions: Annotated[bool, typer.Option("--live-predictions")] = False,
    max_requests: Annotated[int, typer.Option("--max-requests", min=0)] = 0,
    max_input_bytes: Annotated[int, typer.Option("--max-input-bytes", min=1)] = 32_000,
    plain: Annotated[
        bool, typer.Option("--plain", help="Print checkpoints without waiting.")
    ] = False,
):
    """Play complete forecast checkpoints; pause at missing forecasts or insufficient budget."""
    source = Replay.load(replay_file)
    _, first, last = bounds(source, start, end, step)
    with PredictionCache(cache) as storage:
        service = service_for(source, storage, model, max_requests, max_input_bytes)
        complete = play(
            service,
            team,
            first,
            last,
            step,
            speed,
            mode.value,
            live_predictions,
            plain,
            console,
            top,
        )
        if not complete:
            raise typer.Exit(2)


@app.command()
@guarded
def evaluate(
    replay_file: ReplayPath,
    team: TeamOption = "Alliance",
    mode: ModeOption = Mode.team,
    cache: CacheOption = DEFAULT_CACHE,
    model: ModelOption = DEFAULT_MODEL,
    predictions: Annotated[
        Path | None, typer.Option("--predictions", exists=True, dir_okay=False)
    ] = None,
    output: OutputOption = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
):
    """Score cached or exported forecasts against final placement; never calls Jev."""
    source = Replay.load(replay_file)
    team_id = source.team(team).team_id
    if predictions:
        records = [
            r
            for r in import_predictions(predictions)
            if r.team_id == team_id
            and r.observation_mode == mode.value
            and r.requested_model == model
        ]
    else:
        with PredictionCache(cache) as storage:
            records = storage.records(source.fingerprint, team_id, mode.value, model)
    report = evaluate_records(source, records)
    if output:
        export_evaluation(report, output)
    if json_output:
        emit_json(report)
    else:
        console.print(
            Text(
                f"ApexBench | {source.team(team).team_name} | "
                f"actual placement {source.team(team).placement} | {mode.value}"
            )
        )
        console.print(
            f"Checkpoints: {report['count']}\n"
            f"Mean Brier score: {report['mean_brier_score']:.6f}\n"
            f"Mean log loss: {report['mean_log_loss']:.6f}"
        )
        console.print(report["metric_definitions"]["warning"], style="dim")


def main():
    app()
