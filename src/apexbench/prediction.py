"""Cache-first checkpoint inference shared by all CLI commands."""

from __future__ import annotations

import math

from apexbench.cache import PredictionCache, PredictionRecord, cache_key
from apexbench.inference import (
    PROMPT_VERSION,
    BudgetExceeded,
    InferenceBudget,
    PlacementProvider,
    PredictionError,
)
from apexbench.models import ObservationMode
from apexbench.providers import build_request
from apexbench.replay import Replay
from apexbench.state import reconstruct


def checkpoints(start: float, end: float, interval: float) -> list[float]:
    if (
        not all(math.isfinite(x) for x in (start, end, interval))
        or start < 0
        or end < start
        or interval <= 0
    ):
        raise ValueError("require 0 <= start <= end and a finite positive interval")
    count = math.floor((end - start) / interval + 1e-9) + 1
    if count > 100_000:
        raise ValueError("too many checkpoints; increase the interval")
    return [round(start + i * interval, 6) for i in range(count)]


class PredictionService:
    def __init__(
        self,
        replay: Replay,
        cache: PredictionCache,
        provider: PlacementProvider,
        budget: InferenceBudget,
    ):
        self.replay, self.cache, self.provider, self.budget = replay, cache, provider, budget

    def predict(
        self,
        team: str | int,
        timestamp: float,
        mode: ObservationMode = "team",
        cache_only: bool = False,
    ) -> tuple[PredictionRecord, bool]:
        state = reconstruct(self.replay, team, timestamp, mode)
        if (
            state.target_team.eliminated
            or state.squads_remaining <= 1
            or timestamp >= self.replay.data.summary.duration
        ):
            raise PredictionError(
                "Team outcome is already resolved at this timestamp; choose an earlier checkpoint"
            )
        request = build_request(state, self.provider.model)
        key = cache_key(self.replay.fingerprint, request)
        record = self.cache.get(key)
        if record:
            return record, True
        if cache_only:
            raise PredictionError(f"No cached prediction at {timestamp:g}s")
        if self.budget.requests_used >= self.budget.max_requests:
            raise BudgetExceeded(f"API request budget exhausted ({self.budget.max_requests})")
        result = self.provider.predict(request, self.budget)
        record = PredictionRecord.create(
            self.replay.fingerprint, self.replay.data.summary.game_id, request, result
        )
        self.cache.put(record)
        return self.cache.get(key), False

    def cached(self, team: str | int, mode: ObservationMode = "team") -> list[PredictionRecord]:
        return [
            r
            for r in self.cache.records(
                self.replay.fingerprint, self.replay.team(team).team_id, mode, self.provider.model
            )
            if r.prompt_version == PROMPT_VERSION
        ]


def latest_before(records: list[PredictionRecord], timestamp: float) -> PredictionRecord | None:
    eligible = [r for r in records if r.timestamp <= timestamp]
    return max(eligible, key=lambda r: r.timestamp) if eligible else None
