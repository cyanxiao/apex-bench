"""Deterministic, causal projection. No raw replay object enters a model request."""

from __future__ import annotations

import math

from apexbench.models import (
    DeathboxState,
    GameState,
    ObservationMode,
    ObservedPosition,
    PlayerState,
    RingState,
    TeamState,
)
from apexbench.replay import Replay


def reconstruct(
    replay: Replay, team: str | int, timestamp: float, mode: ObservationMode = "team"
) -> GameState:
    if not math.isfinite(timestamp) or not 0 <= timestamp <= replay.data.summary.duration:
        raise ValueError("timestamp must be finite and within the replay duration")
    if mode not in ("team", "oracle"):
        raise ValueError("observation mode must be team or oracle")
    selected = replay.team(team)
    teams = {
        t.team_id: TeamState(
            team_id=t.team_id,
            team_name=t.team_name,
            players=[PlayerState(name=p.name) for p in sorted(t.players, key=lambda p: p.name)],
        )
        for t in sorted(replay.data.summary.teams, key=lambda t: t.team_id)
    }
    players = {(tid, p.name): p for tid, t in teams.items() for p in t.players}
    last_resets: dict[tuple[int, str], float] = {}

    # Character selections can change during pregame. Use the latest timestamped identity,
    # never the summary's character or its order (teams there are sorted by final placement).
    for event in replay.events_until(timestamp):
        actor = players.get((event.actor.team_id, event.actor.name)) if event.actor else None
        target = players.get((event.target.team_id, event.target.name)) if event.target else None
        for person, state, pos, health, shield in (
            (event.actor, actor, event.pos_actor, event.actor_health, event.actor_shield),
            (event.target, target, event.pos_target, event.target_health, event.target_shield),
        ):
            if person is None or state is None:
                continue
            if person.character:
                state.character = person.character
            # All actor fields are observer telemetry. Team mode only sees its own telemetry.
            if mode != "oracle" and person.team_id != selected.team_id:
                continue
            if pos is not None:
                state.position = ObservedPosition(x=pos.x, y=pos.y, observed_at=event.ts_game)
            if health is not None or shield is not None:
                state.vitals_observed_at = event.ts_game
                if health is not None:
                    state.health = health
                if shield is not None:
                    state.shield = shield

        if event.category == "weaponSwitched" and actor:
            actor.weapon = event.new_weapon
        elif event.category == "playerDamaged" and actor and event.ts_game >= 0:
            actor.damage_dealt += max(0, event.damage or 0)
        elif event.category == "playerDowned" and target:
            target.status = "downed"
        elif event.category == "playerKilled" and target:
            last_resets[(event.target.team_id, event.target.name)] = event.ts_game
            target.status = "dead"
            target.health = 0
            target.shield = 0
            target.vitals_observed_at = event.ts_game
            target.weapon = None
            if actor and event.actor.team_id != event.target.team_id and event.ts_game >= 0:
                actor.kills += 1
                teams[event.actor.team_id].kills += 1
        elif event.category == "playerRevive" and target:
            target.status = "alive"
            # The event may contain the pre-revive health; do not call that current health.
            target.health = target.shield = target.vitals_observed_at = None
            if actor:
                actor.revives += 1
        elif event.category == "playerRespawnTeam" and event.actor:
            names = set(event.respawned_names)
            if event.respawned_name:
                names.add(event.respawned_name)
            for name in names:
                player = players.get((event.actor.team_id, name))
                if player:
                    last_resets[(event.actor.team_id, name)] = event.ts_game
                    player.status = "alive"
                    player.position = None
                    player.health = player.shield = player.vitals_observed_at = None
                    player.weapon = None
                    player.respawns += 1

    # Timeline squad eliminations have team IDs; raw squadEliminated events often do not.
    for elimination in replay.data.timeline.squad_eliminations:
        if elimination.ts_game <= timestamp:
            squad = teams[elimination.team_id]
            squad.eliminated = True
            for player in squad.players:
                player.status = "dead"

    for (tid, name), player in players.items():
        if mode != "oracle" and tid != selected.team_id:
            # Only publicly observed kills/eliminations and pregame roster for opponents.
            player.position = None
            player.health = player.shield = player.vitals_observed_at = None
            player.weapon = None
            player.damage_dealt = 0
            player.revives = player.respawns = 0
            player.status = "dead" if teams[tid].eliminated else "alive"
            continue
        point = replay.position_at(tid, name, timestamp)
        # Do not resurrect a pre-death/pre-respawn path sample or interpolate future points.
        last_reset = last_resets.get((tid, name), -math.inf)
        if (
            point
            and point.ts_game >= last_reset
            and (player.position is None or point.ts_game > player.position.observed_at)
        ):
            player.position = ObservedPosition(x=point.x, y=point.y, observed_at=point.ts_game)

    known_rings = [r for r in replay.data.rings.rings if r.ts_game <= timestamp]
    ring = max(known_rings, key=lambda r: r.ts_game) if known_rings else None
    ring_state = (
        None
        if ring is None
        else RingState(
            stage=ring.stage,
            phase=ring.type,
            observed_at=ring.ts_game,
            center={"x": ring.center.x, "y": ring.center.y},
            start_radius=ring.start_radius,
            end_radius=ring.end_radius,
            current_radius=ring.current_radius,
            shrink_duration=ring.shrink_duration,
        )
    )
    boxes = [
        DeathboxState(
            observed_at=b.ts_game,
            victim_name=b.victim.name,
            team_id=b.victim.team_id,
            position=ObservedPosition(x=b.pos.x, y=b.pos.y, observed_at=b.ts_game)
            if b.pos
            else None,
        )
        for b in sorted(replay.data.deathboxes.deathboxes, key=lambda b: (b.ts_game, b.deathbox_id))
        if b.ts_game <= timestamp and (mode == "oracle" or b.victim.team_id == selected.team_id)
    ]
    notes = [
        "Last observed telemetry, not continuously measured health or inventory.",
        "Positions use samples at or before the cutoff; no interpolation or future ring centers.",
        "Roster, public kill counts and squad eliminations are shared match observations.",
        "POI polygons use an unverified coordinate transform and are excluded.",
    ]
    notes += (
        ["ORACLE: all teams' timestamped observer telemetry and deathbox locations are visible."]
        if mode == "oracle"
        else [
            "TEAM: own observer telemetry; opponent telemetry is hidden.",
            "Conservative observation policy; exact first-person visibility is not simulated.",
        ]
    )
    return GameState(
        timestamp=timestamp,
        map=replay.data.summary.map,
        observation_mode=mode,
        observation_notes=notes,
        team_count=len(teams),
        squads_remaining=sum(not t.eliminated for t in teams.values()),
        target_team=teams[selected.team_id],
        opponents=[t for tid, t in teams.items() if tid != selected.team_id],
        ring=ring_state,
        deathboxes=boxes,
    )
