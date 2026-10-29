"""Require a complete set of forecasts for each scheduled replay checkpoint."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass

from apexbench.cache import PredictionRecord
from apexbench.jev import PredictionError
from apexbench.models import GameState, ObservationMode, TeamState
from apexbench.prediction import PredictionService, checkpoints
from apexbench.state import reconstruct

RETRY_HINT = "Press c or space/p to retry this checkpoint, or rewind/quit."


@dataclass
class PlaybackSnapshot:
    state: GameState
    checkpoint: float
    required: list[TeamState]
    forecasts: dict[int, PredictionRecord]
    reason: str = ""
    loading: bool = False

    @property
    def blocked(self) -> bool:
        return bool(self.reason) and not self.loading


class PlaybackSession:
    def __init__(
        self,
        service: PredictionService,
        team: str | int,
        start: float,
        end: float,
        step: float,
        mode: ObservationMode,
        live_predictions: bool,
        top: int | None = None,
    ):
        self.service = service
        self.mode = mode
        self.live_predictions = live_predictions
        self.start, self.end = start, end
        self.times = checkpoints(start, end, step)
        if top is not None and top < 1:
            raise ValueError("top team count must be positive")
        self.team_ids = (
            sorted(t.team_id for t in service.replay.data.summary.teams)
            if top is not None
            else [service.replay.team(team).team_id]
        )
        self.records: dict[int, dict[float, PredictionRecord]] = {}
        self.failures: dict[tuple[int, float], str] = {}
        self.reload_cache()

    def reload_cache(self):
        self.records = {
            tid: {r.timestamp: r for r in self.service.cached(tid, self.mode)}
            for tid in self.team_ids
        }

    def prepare(
        self,
        timestamp: float,
        *,
        retry: bool = False,
        progress: Callable[[PlaybackSnapshot], None] | None = None,
    ) -> PlaybackSnapshot:
        if not self.start <= timestamp <= self.end:
            raise ValueError("playback timestamp must be within the session bounds")
        checkpoint = self.times[max(0, bisect_right(self.times, timestamp) - 1)]
        state = reconstruct(self.service.replay, self.team_ids[0], timestamp, self.mode)
        required = [
            t
            for t in [state.target_team, *state.opponents]
            if not t.eliminated and t.team_id in self.team_ids
        ]
        if state.squads_remaining <= 1 or timestamp >= self.service.replay.data.summary.duration:
            required = []  # Resolved outcomes do not require new predictions.
        required.sort(key=lambda t: t.team_id)
        if retry:
            self.reload_cache()
            self.failures = {
                key: value for key, value in self.failures.items() if key[1] != checkpoint
            }
        forecasts = {
            t.team_id: self.records[t.team_id][checkpoint]
            for t in required
            if checkpoint in self.records[t.team_id]
        }
        snapshot = PlaybackSnapshot(state, checkpoint, required, forecasts)
        missing = [t for t in required if t.team_id not in forecasts]
        if not missing:
            return snapshot

        def stop(reason: str) -> PlaybackSnapshot:
            names = ", ".join(t.team_name for t in required if t.team_id not in forecasts)
            snapshot.loading = False
            snapshot.reason = (
                f"Replay stopped at {checkpoint:g}s: {len(forecasts)}/{len(required)} "
                f"forecasts for this checkpoint.\n{reason}\nMissing: {names}"
            )
            return snapshot

        if not self.live_predictions:
            return stop(
                "Cache incomplete; live predictions are disabled. "
                "Populate the cache or restart with --live-predictions and a request budget."
            )
        for team in missing:
            failure = self.failures.get((team.team_id, checkpoint))
            if failure:
                return stop(failure + "\n" + RETRY_HINT)
        remaining = max(0, self.service.budget.max_requests - self.service.budget.requests_used)
        # Avoid spending the last few requests on a checkpoint we cannot complete.
        if len(missing) > remaining:
            reason = (
                f"API request budget exhausted ({self.service.budget.max_requests})."
                if remaining == 0
                else "Insufficient API request budget."
            )
            return stop(
                f"{reason} Need {len(missing)} requests; {remaining} remaining. "
                "Restart with a larger --max-requests budget or populate the cache."
            )
        snapshot.loading = True
        snapshot.reason = f"Preparing complete forecasts at {checkpoint:g}s; playback is paused."
        if progress:
            progress(snapshot)
        for team in missing:
            try:
                record, _ = self.service.predict(team.team_id, checkpoint, self.mode)
            except PredictionError as exc:
                failure = f"{team.team_name} at {checkpoint:g}s: {exc}"
                self.failures[(team.team_id, checkpoint)] = failure
                stop(failure + "\n" + RETRY_HINT)
                if progress:
                    progress(snapshot)
                return snapshot
            self.records[team.team_id][checkpoint] = record
            forecasts[team.team_id] = record
            if progress:
                progress(snapshot)
        snapshot.loading = False
        snapshot.reason = ""
        return snapshot
