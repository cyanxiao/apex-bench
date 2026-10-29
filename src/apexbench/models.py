"""Validated replay structures. Raw extras are preserved for inspection, never prompting."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

Number = FiniteFloat
Probability = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
ObservationMode = Literal["team", "oracle"]


class ReplayObject(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Position(ReplayObject):
    x: Number
    y: Number
    z: Number | None = None


class Person(ReplayObject):
    name: str
    nucleus_hash: str | None = Field(default=None, alias="nucleusHash")
    team_id: int | None = Field(default=None, alias="teamId")
    character: str | None = None


class Team(ReplayObject):
    team_id: int = Field(alias="teamId")
    team_name: str = Field(alias="teamName")
    players: list[Person] = Field(min_length=1)
    placement: int | None = Field(default=None, ge=1)


class Summary(ReplayObject):
    game_id: str = Field(alias="gameId", min_length=1)
    map: str
    map_display_name: str | None = Field(default=None, alias="mapDisplayName")
    duration: Number = Field(ge=0)
    teams: list[Team] = Field(min_length=2)

    @model_validator(mode="after")
    def unique_teams(self):
        ids = [t.team_id for t in self.teams]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate team IDs")
        for team in self.teams:
            if team.placement is not None and team.placement > len(ids):
                raise ValueError("placement exceeds team count")
            names = [p.name for p in team.players]
            if len(set(names)) != len(names):
                raise ValueError("duplicate player names within a team")
        return self


class PathPoint(Position):
    ts_game: Number = Field(alias="tsGame")


class PathPlayer(ReplayObject):
    nucleus_hash: str | None = Field(default=None, alias="nucleusHash")
    player_name: str = Field(alias="playerName")
    points: list[PathPoint] = Field(default_factory=list)


class PathTeam(ReplayObject):
    team_id: int = Field(alias="teamId")
    players: list[PathPlayer] = Field(default_factory=list)


class Pathing(ReplayObject):
    teams: list[PathTeam] = Field(default_factory=list)


class Event(ReplayObject):
    id: int = 0
    ts_game: Number = Field(alias="tsGame")
    category: str
    actor: Person | None = None
    target: Person | None = None
    pos_actor: Position | None = Field(default=None, alias="posActor")
    pos_target: Position | None = Field(default=None, alias="posTarget")
    actor_health: Number | None = Field(default=None, alias="actorHealth")
    actor_shield: Number | None = Field(default=None, alias="actorShield")
    target_health: Number | None = Field(default=None, alias="targetHealth")
    target_shield: Number | None = Field(default=None, alias="targetShield")
    damage: Number | None = None
    weapon: str | None = None
    new_weapon: str | None = Field(default=None, alias="newWeapon")
    respawned_names: list[str] = Field(default_factory=list, alias="respawnedTeammateNames")
    respawned_name: str | None = Field(default=None, alias="respawnedName")


class TimelineEntry(ReplayObject):
    ts_game: Number = Field(alias="tsGame")
    team_id: int | None = Field(default=None, alias="teamId")


class FightWindow(ReplayObject):
    start: Number = Field(alias="startTsGame")
    end: Number = Field(alias="endTsGame")

    @model_validator(mode="after")
    def ordered(self):
        if self.start > self.end:
            raise ValueError("fight window ends before it starts")
        return self


class Timeline(ReplayObject):
    kills: list[TimelineEntry] = Field(default_factory=list)
    downs: list[TimelineEntry] = Field(default_factory=list)
    squad_eliminations: list[TimelineEntry] = Field(default_factory=list, alias="squadEliminations")
    ring_events: list[TimelineEntry] = Field(default_factory=list, alias="ringEvents")
    respawns: list[TimelineEntry] = Field(default_factory=list)
    revives: list[TimelineEntry] = Field(default_factory=list)
    fight_windows: list[FightWindow] = Field(default_factory=list, alias="fightWindows")


class Deathbox(ReplayObject):
    deathbox_id: int = Field(alias="deathboxId")
    ts_game: Number = Field(alias="tsGame")
    pos: Position | None = None
    victim: Person
    inventory: list[dict[str, Any]] = Field(default_factory=list)


class Deathboxes(ReplayObject):
    deathboxes: list[Deathbox] = Field(default_factory=list)


class Ring(ReplayObject):
    type: str
    stage: int = Field(ge=0)
    ts_game: Number = Field(alias="tsGame")
    center: Position
    start_radius: Number | None = Field(default=None, alias="startRadius", ge=0)
    end_radius: Number | None = Field(default=None, alias="endRadius", ge=0)
    current_radius: Number | None = Field(default=None, alias="currentRadius", ge=0)
    shrink_duration: Number | None = Field(default=None, alias="shrinkDuration", ge=0)


class Rings(ReplayObject):
    rings: list[Ring] = Field(default_factory=list)


class Polygon(ReplayObject):
    name: str
    coordinates: list[Position] = Field(min_length=3)


class POIPayload(ReplayObject):
    polygons: list[Polygon] = Field(default_factory=list)


class POIs(ReplayObject):
    payload: POIPayload = Field(default_factory=POIPayload)


class ReplayData(ReplayObject):
    summary: Summary
    events: list[Event]
    pathing: Pathing = Field(default_factory=Pathing)
    timeline: Timeline = Field(default_factory=Timeline)
    deathboxes: Deathboxes = Field(default_factory=Deathboxes)
    rings: Rings = Field(default_factory=Rings)
    pois: POIs = Field(default_factory=POIs)


class SafeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ObservedPosition(SafeModel):
    x: float
    y: float
    observed_at: float


class PlayerState(SafeModel):
    name: str
    character: str | None = None
    status: Literal["alive", "downed", "dead"] = "alive"
    position: ObservedPosition | None = None
    health: float | None = None
    shield: float | None = None
    vitals_observed_at: float | None = None
    weapon: str | None = None
    kills: int = 0
    damage_dealt: float = 0
    revives: int = 0
    respawns: int = 0


class TeamState(SafeModel):
    team_id: int
    team_name: str
    eliminated: bool = False
    kills: int = 0
    players: list[PlayerState]


class RingState(SafeModel):
    stage: int
    phase: str
    observed_at: float
    center: dict[str, float]
    start_radius: float | None
    end_radius: float | None
    current_radius: float | None
    shrink_duration: float | None


class DeathboxState(SafeModel):
    observed_at: float
    victim_name: str
    team_id: int | None
    position: ObservedPosition | None


class GameState(SafeModel):
    state_version: str = "apexbench-state-v1"
    timestamp: float
    map: str
    observation_mode: ObservationMode
    observation_notes: list[str]
    team_count: int
    squads_remaining: int
    target_team: TeamState
    opponents: list[TeamState]
    ring: RingState | None
    deathboxes: list[DeathboxState]


class Forecast(SafeModel):
    probabilities: list[Probability] = Field(min_length=2)

    @model_validator(mode="after")
    def normalized(self):
        if not math.isclose(math.fsum(self.probabilities), 1, rel_tol=0, abs_tol=1e-6):
            raise ValueError("placement probabilities must sum to 1 (tolerance 1e-6)")
        return self

    @property
    def expected_placement(self) -> float:
        return sum(i * p for i, p in enumerate(self.probabilities, 1))
