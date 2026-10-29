"""Load local Apex Legends replay JSON files; isolate source metadata from observations."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from pathlib import Path

from apexbench.models import ReplayData, Team


class Replay:
    def __init__(self, data: ReplayData, fingerprint: str):
        self.data = data
        self.fingerprint = fingerprint
        self.events = sorted(data.events, key=lambda e: (e.ts_game, e.id))
        self.event_times = [e.ts_game for e in self.events]
        self.paths = {}
        self.warnings = []
        known_ids = {t.team_id for t in data.summary.teams}
        for section in ("pathing", "timeline", "deathboxes", "rings", "pois"):
            if section not in data.model_fields_set:
                self.warnings.append(f"Missing {section}; related observations are unavailable.")
        for team in data.pathing.teams:
            if team.team_id not in known_ids:
                raise ValueError(f"pathing references unknown team {team.team_id}")
            roster = next(t for t in data.summary.teams if t.team_id == team.team_id)
            for player in team.players:
                if player.player_name not in {p.name for p in roster.players}:
                    raise ValueError(f"pathing references unknown player {player.player_name}")
                key = (team.team_id, player.player_name)
                if key in self.paths:
                    raise ValueError(f"duplicate path for {player.player_name}")
                points = sorted(player.points, key=lambda p: p.ts_game)
                self.paths[key] = ([p.ts_game for p in points], points)
        for entry in data.timeline.squad_eliminations:
            if entry.team_id not in known_ids:
                raise ValueError("squad elimination references an unknown team")
        if not data.timeline.squad_eliminations:
            self.warnings.append(
                "No team-addressed eliminations; squads remaining may be incomplete."
            )

    @classmethod
    def load(cls, path: Path | str) -> Replay:
        content = Path(path).read_bytes()
        data = ReplayData.model_validate_json(content)
        return cls(data, hashlib.sha256(content).hexdigest())

    @classmethod
    def from_dict(cls, data: dict) -> Replay:
        content = json.dumps(data, sort_keys=True, allow_nan=False).encode()
        return cls(ReplayData.model_validate(data), hashlib.sha256(content).hexdigest())

    def team(self, name_or_id: str | int) -> Team:
        matches = [
            t
            for t in self.data.summary.teams
            if str(t.team_id) == str(name_or_id)
            or t.team_name.casefold() == str(name_or_id).casefold()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Team {name_or_id!r} not found or ambiguous; use inspect for team IDs"
            )
        return matches[0]

    def events_until(self, timestamp: float):
        return self.events[: bisect_right(self.event_times, timestamp)]

    def position_at(self, team_id: int, name: str, timestamp: float):
        times, points = self.paths.get((team_id, name), ([], []))
        index = bisect_right(times, timestamp) - 1
        return points[index] if index >= 0 else None

    def inspection(self) -> dict:
        summary = self.data.summary
        return {
            "application": "ApexBench",
            "game_id": summary.game_id,
            "sha256": self.fingerprint,
            "map": summary.map_display_name or summary.map,
            "duration_seconds": summary.duration,
            "team_count": len(summary.teams),
            "teams": [
                {"id": t.team_id, "name": t.team_name, "placement": t.placement}
                for t in sorted(summary.teams, key=lambda t: t.team_id)
            ],
            "sections": {
                "summary": sorted(summary.model_dump(by_alias=True)),
                "pathing": {
                    "players": len(self.paths),
                    "points": sum(len(p) for _, p in self.paths.values()),
                },
                "events": {
                    "count": len(self.events),
                    "categories": sorted({e.category for e in self.events}),
                    "fields": sorted(
                        {
                            k
                            for e in self.events
                            for k in e.model_dump(by_alias=True, exclude_none=True)
                        }
                    ),
                },
                "timeline": {
                    k: len(v)
                    for k, v in self.data.timeline.model_dump(by_alias=True).items()
                    if isinstance(v, list)
                },
                "deathboxes": len(self.data.deathboxes.deathboxes),
                "rings": len(self.data.rings.rings),
                "pois": [p.name for p in self.data.pois.payload.polygons],
            },
            "warnings": self.warnings,
            "notice": "Inspection includes final results. It is never sent to Jev.",
        }
