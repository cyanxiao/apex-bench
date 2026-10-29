"""Rich terminal playback with causal checkpoint selection and keyboard controls."""

from __future__ import annotations

import math
import os
import select
import sys
import time
from bisect import bisect_left, bisect_right
from contextlib import contextmanager

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from apexbench.cache import PredictionRecord
from apexbench.inference import InferenceBudget
from apexbench.models import GameState, ObservationMode
from apexbench.playback_session import PlaybackSession, PlaybackSnapshot
from apexbench.prediction import PredictionService, latest_before
from apexbench.providers import is_jev

CONTROLS = "Space/p pause · c retry · n/l next · b/h back · r restart · +/- speed · q quit"


def _request_usage(budget: InferenceBudget) -> str:
    remaining = max(0, budget.max_requests - budget.requests_used)
    return (
        f"API requests: {budget.requests_used}/{budget.max_requests} used · {remaining} remaining"
    )


def frame(
    state: GameState,
    record: PredictionRecord | None,
    speed: float,
    paused: bool,
    message: str = "",
    compact: bool = False,
    *,
    budget: InferenceBudget | None = None,
    checkpoint: float | None = None,
    outcome_resolved: bool = False,
) -> Group:
    team = state.target_team
    title = (
        f"ApexBench | {team.team_name} | {state.timestamp:07.1f}s | "
        f"{state.observation_mode.upper()}"
    )
    rows = Table(expand=True, box=None if compact else box.HEAVY_HEAD)
    rows.add_column("Player / Legend", no_wrap=True)
    for label in ("Status", "Last HP/shield", "Position @ time", "Kills"):
        rows.add_column(label)
    for player in team.players:
        pos = player.position
        health = "?" if player.health is None else f"{player.health:g}"
        shield = "?" if player.shield is None else f"{player.shield:g}"
        rows.add_row(
            Text(f"{player.name}\n{player.character or '?'}"),
            player.status,
            f"{health}/{shield}",
            f"{pos.x:.0f}, {pos.y:.0f} @ {pos.observed_at:g}s" if pos else "unknown",
            str(player.kills),
        )
    ring = (
        f"stage {state.ring.stage}, {state.ring.phase}, observed @ {state.ring.observed_at:g}s"
        if state.ring
        else "not yet observed"
    )
    summary = Text(
        f"Squads: {state.squads_remaining}/{state.team_count} | Kills: {team.kills}"
        f" | Ring: {ring}\n{'PAUSED' if paused else 'PLAYING'} {speed:g}x"
    )
    if budget is not None:
        summary.append(f"\n{_request_usage(budget)}")
    if record:
        forecast = record.forecast
        model_label = "Jev" if is_jev(record.requested_model) else record.requested_model
        prediction = Table(
            title=Text(
                f"{model_label} @ {record.timestamp:g}s "
                f"(age {state.timestamp - record.timestamp:.2f}s)"
            ),
            expand=True,
            box=None if compact else box.HEAVY_HEAD,
        )
        prediction.add_column("Placement")
        prediction.add_column("Probability")
        prediction.add_column("Distribution")
        best = sorted(enumerate(forecast.probabilities, 1), key=lambda p: (-p[1], p[0]))[
            : 3 if compact else 5
        ]
        for placement, probability in best:
            prediction.add_row(str(placement), f"{probability:.2%}", "━" * round(probability * 40))
        forecast_view = Group(
            Text(
                f"Expected placement: {forecast.expected_placement:.2f} | "
                f"Win: {forecast.probabilities[0]:.2%} | "
                f"Top 5: {sum(forecast.probabilities[:5]):.2%}"
            ),
            prediction,
            Text(f"Model: {record.resolved_model}", style="dim"),
        )
    elif outcome_resolved:
        forecast_view = Text("Outcome resolved; no forecast required.")
    elif checkpoint is not None:
        forecast_view = Text(
            f"Forecast unavailable at {checkpoint:g}s; playback is stopped.", style="yellow"
        )
    else:
        forecast_view = Text(
            "No cached forecast at or before this time. Use precompute or "
            "--live-predictions with a request budget.",
            style="yellow",
        )
    return Group(
        Panel(Group(summary, rows), title=Text(title)),
        Panel(forecast_view, title="Placement forecast"),
        Text(message, style="yellow"),
        Text(CONTROLS, style="dim"),
    )


def ranked_teams(state: GameState, records: dict[int, list[PredictionRecord]], limit: int):
    """Rank active squads using only forecasts already available at the playhead."""
    if limit < 1:
        raise ValueError("top team count must be positive")
    rows = [
        (team, latest_before(records.get(team.team_id, []), state.timestamp))
        for team in [state.target_team, *state.opponents]
        if not team.eliminated
    ]
    rows.sort(
        key=lambda row: (
            row[1] is None,
            row[1].forecast.expected_placement if row[1] else math.inf,
            -row[1].forecast.probabilities[0] if row[1] else 0,
            -row[0].kills,
            row[0].team_id,
        )
    )
    return rows[:limit]


def leaderboard_frame(
    state: GameState,
    records: dict[int, list[PredictionRecord]],
    limit: int,
    speed: float,
    paused: bool,
    message: str = "",
    compact: bool = False,
    *,
    budget: InferenceBudget | None = None,
    checkpoint: float | None = None,
    complete: bool = True,
    outcome_resolved: bool = False,
) -> Group:
    active = [team for team in [state.target_team, *state.opponents] if not team.eliminated]
    coverage = sum(
        latest_before(records.get(t.team_id, []), state.timestamp) is not None for t in active
    )
    table = Table(expand=True, box=None if compact else box.HEAVY_HEAD)
    for label in ("#", "Team", "Kills", "Expected", "Win", "Forecast / age"):
        table.add_column(label, no_wrap=True)
    table.columns[1].no_wrap = False
    table.columns[1].ratio = 1
    for index, (team, record) in enumerate(ranked_teams(state, records, limit), 1):
        table.add_row(
            str(index),
            Text(team.team_name),
            str(team.kills),
            f"{record.forecast.expected_placement:.2f}" if record else "—",
            f"{record.forecast.probabilities[0]:.1%}" if record else "—",
            f"{record.timestamp:g}s / {state.timestamp - record.timestamp:.2f}s"
            if record
            else "missing",
        )
    title = (
        f"ApexBench | Top {limit} teams | {state.timestamp:07.1f}s | "
        f"{state.observation_mode.upper()}"
    )
    coverage_label = f"Forecasts @ {checkpoint:g}s" if checkpoint is not None else "Forecasts"
    forecast_status = (
        "Outcome resolved" if outcome_resolved else f"{coverage_label}: {coverage}/{len(active)}"
    )
    summary = Text(
        f"Active squads: {len(active)}/{state.team_count} | {forecast_status}"
        f" | {'PAUSED' if paused else 'PLAYING'} {speed:g}x"
    )
    if budget is not None:
        summary.append(f"\n{_request_usage(budget)}")
    ordering = Text("Order: expected placement ↑.", style="dim")
    warning = (
        "Partial forecast coverage; this is not a complete forecast ranking."
        if coverage < len(active)
        else ""
    )
    if not coverage:
        warning = "No forecasts yet; teams ordered by observed kills (ties: team ID)."
    if checkpoint is not None:
        warning = ""
    if not complete:
        ordering = Text()
    if outcome_resolved:
        ordering = Text("Outcome resolved; no forecasts required.", style="dim")
    return Group(
        Panel(
            Group(
                summary,
                table
                if complete
                else Text(
                    "Ranking withheld until this checkpoint has complete forecasts.", style="yellow"
                ),
            ),
            title=Text(title),
        ),
        ordering,
        Text(warning, style="yellow"),
        Text(message, style="yellow"),
        Text(CONTROLS, style="dim"),
    )


@contextmanager
def keyboard():
    if os.name == "nt":
        import msvcrt

        yield lambda: msvcrt.getwch() if msvcrt.kbhit() else ""
    else:
        import termios
        import tty

        old = termios.tcgetattr(sys.stdin.fileno())
        try:
            tty.setcbreak(sys.stdin.fileno())
            # TextIO can prefetch multiple keys, leaving select() blind to buffered
            # input. Read one byte from the descriptor so rapid keys are not lost.
            yield (
                lambda: (
                    os.read(sys.stdin.fileno(), 1).decode("ascii", errors="ignore")
                    if select.select([sys.stdin], [], [], 0)[0]
                    else ""
                )
            )
        finally:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)


def play(
    service: PredictionService,
    team: str | int,
    start: float,
    end: float,
    step: float,
    speed: float,
    mode: ObservationMode,
    live_predictions: bool,
    plain: bool,
    console: Console,
    top: int | None = None,
) -> bool:
    """Return False when playback stops or exits at an incomplete checkpoint."""
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive")
    session = PlaybackSession(service, team, start, end, step, mode, live_predictions, top)
    boundaries = sorted(set([*session.times, end]))

    def render_snapshot(snapshot: PlaybackSnapshot, paused: bool, compact: bool = False):
        state = snapshot.state
        records = {tid: [record] for tid, record in snapshot.forecasts.items()}
        paused = paused or snapshot.blocked or snapshot.loading
        if top is not None:
            return leaderboard_frame(
                state,
                records,
                top,
                speed,
                paused,
                snapshot.reason,
                compact,
                budget=service.budget,
                checkpoint=snapshot.checkpoint,
                complete=not (snapshot.blocked or snapshot.loading),
                outcome_resolved=not snapshot.required,
            )
        return frame(
            state,
            snapshot.forecasts.get(state.target_team.team_id) if not snapshot.blocked else None,
            speed,
            paused,
            snapshot.reason,
            compact,
            budget=service.budget,
            checkpoint=snapshot.checkpoint,
            outcome_resolved=not snapshot.required,
        )

    if plain or console.is_dumb_terminal or not (sys.stdin.isatty() and console.is_terminal):
        for timestamp in session.times:
            snapshot = session.prepare(timestamp)
            console.print(render_snapshot(snapshot, True))
            if snapshot.blocked:
                return False
        return True

    timestamp, paused = start, False
    snapshot = None
    with keyboard() as read_key, Live(console=console, screen=True, auto_refresh=False) as live:

        def refresh_usage(progress: PlaybackSnapshot):
            live.update(
                render_snapshot(progress, True, compact=console.size.height < 32), refresh=True
            )

        while True:
            tick = time.monotonic()
            key = read_key()
            retry = False
            if key.lower() == "q":
                break
            if key in (" ", "p"):
                paused = not paused
                retry = not paused
            elif key.lower() == "c":
                retry, paused = True, False
            elif key in ("n", "l"):
                index = min(len(boundaries) - 1, bisect_right(boundaries, timestamp))
                timestamp, paused = boundaries[index], True
            elif key in ("b", "h"):
                timestamp, paused = boundaries[max(0, bisect_left(boundaries, timestamp) - 1)], True
            elif key == "r":
                timestamp = start
            elif key in ("+", "="):
                speed = min(1024, speed * 2)
            elif key == "-":
                speed = max(0.25, speed / 2)
            snapshot = session.prepare(timestamp, retry=retry, progress=refresh_usage)
            if snapshot.blocked:
                paused = True
            live.update(
                render_snapshot(snapshot, paused, compact=console.size.height < 32), refresh=True
            )
            # Inference time pauses the playhead; API latency cannot skip checkpoints.
            finished = time.monotonic()
            time.sleep(max(0, 0.1 - (finished - tick)))
            if not paused:
                # Never jump over a checkpoint that has not passed completeness checks.
                index = min(len(boundaries) - 1, bisect_right(boundaries, timestamp))
                timestamp = min(
                    boundaries[index], timestamp + (time.monotonic() - finished) * speed
                )
                if timestamp >= end:
                    paused = True
    if snapshot is not None and snapshot.blocked:
        console.print(Text(snapshot.reason, style="yellow"))
        return False
    return True
