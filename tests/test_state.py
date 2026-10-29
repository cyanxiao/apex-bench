import copy
import json

import pytest
from pydantic import ValidationError

from apexbench.replay import Replay
from apexbench.state import reconstruct


def player(state, name="Alliance_B"):
    return next(p for p in state.target_team.players if p.name == name)


def test_parser_all_sections(replay):
    report = replay.inspection()
    assert set(report["sections"]) == {
        "summary",
        "events",
        "pathing",
        "timeline",
        "deathboxes",
        "rings",
        "pois",
    }
    assert replay.team("aLLianCE").team_id == 2
    assert replay.team("2").placement == 1
    assert report["sections"]["pois"] == ["Fixture POI"]


@pytest.mark.parametrize(
    ("timestamp", "status"),
    [(19, "alive"), (20, "downed"), (29, "downed"), (30, "dead"), (49, "dead"), (50, "alive")],
)
def test_death_respawn_boundaries(replay, timestamp, status):
    assert player(reconstruct(replay, 2, timestamp)).status == status


def test_respawn_invalidates_old_telemetry(replay):
    state = reconstruct(replay, 2, 50)
    assert player(state).position is None
    assert player(state).health is None
    assert player(state).shield is None
    assert player(state).respawns == 1
    assert player(reconstruct(replay, 2, 60)).position.observed_at == 60


def test_positions_and_ring_use_last_sample(replay):
    state = reconstruct(replay, 2, 25)
    assert player(state, "Alliance_A").position.x == 10
    assert player(state, "Alliance_A").position.observed_at == 10
    assert state.ring.center == {"x": 0, "y": 0}
    assert state.ring.current_radius is None
    assert reconstruct(replay, 2, 9).ring is None
    assert reconstruct(replay, 2, 40).ring.center == {"x": 40, "y": 40}


def test_future_mutation_cannot_change_state(raw):
    before = reconstruct(Replay.from_dict(raw), 2, 35).model_dump()
    changed = copy.deepcopy(raw)
    summary = changed["summary"]
    summary["winner"] = {"teamId": 4, "secret": "future"}
    summary["duration"] = 180
    summary["teams"].reverse()
    for team in summary["teams"]:
        team["placement"] = 3
        team["stats"] = {"kills": 12345}
        team["players"].reverse()
        for p in team["players"]:
            p["character"] = "future-summary-character"
    for event in changed["events"]:
        if event["tsGame"] > 35:
            event["actorHealth"] = 98765
            event["secret"] = "future"
    for t in changed["pathing"]["teams"]:
        for p in t["players"]:
            for point in p["points"]:
                if point["tsGame"] > 35:
                    point["x"] = 888888
    changed["rings"]["rings"][1]["center"]["x"] = 77777
    changed["timeline"]["fightWindows"][0]["endTsGame"] = 170
    changed["timeline"]["fightWindows"][0]["teamIds"] = [4]
    changed["deathboxes"]["deathboxes"][0]["hasRespawned"] = False
    changed["deathboxes"]["deathboxes"][0]["respawnTsGame"] = 160
    assert reconstruct(Replay.from_dict(changed), 2, 35).model_dump() == before


@pytest.mark.parametrize("mode", ["team", "oracle"])
def test_removing_entire_future_prefix_is_invariant(raw, mode):
    source = Replay.from_dict(raw)
    for cutoff in [0, 20, 35, 50, 70, 95]:
        past = copy.deepcopy(raw)
        past["events"] = [e for e in past["events"] if e["tsGame"] <= cutoff]
        for team in past["pathing"]["teams"]:
            for p in team["players"]:
                p["points"] = [point for point in p["points"] if point["tsGame"] <= cutoff]
        past["rings"]["rings"] = [r for r in past["rings"]["rings"] if r["tsGame"] <= cutoff]
        past["deathboxes"]["deathboxes"] = [
            b for b in past["deathboxes"]["deathboxes"] if b["tsGame"] <= cutoff
        ]
        past["timeline"]["squadEliminations"] = [
            e for e in past["timeline"]["squadEliminations"] if e["tsGame"] <= cutoff
        ]
        assert reconstruct(source, 2, cutoff, mode) == reconstruct(
            Replay.from_dict(past), 2, cutoff, mode
        )


def test_oracle_is_explicit_and_separated(replay):
    team = reconstruct(replay, 2, 15)
    oracle = reconstruct(replay, 2, 15, "oracle")
    beta = next(t for t in team.opponents if t.team_id == 3)
    oracle_beta = next(t for t in oracle.opponents if t.team_id == 3)
    assert beta.players[0].position is None
    assert beta.players[0].health is None
    assert oracle_beta.players[0].position.x == 50
    assert oracle_beta.players[0].health == 100
    assert oracle.observation_mode == "oracle"
    assert any("ORACLE" in n for n in oracle.observation_notes)


def test_no_raw_or_final_data_in_state(replay):
    state = reconstruct(replay, 2, 35, "oracle")
    text = state.model_dump_json()
    for token in [
        "winner",
        "placement",
        "stats",
        "duration",
        "fightWindows",
        "respawnTsGame",
        "inventory",
        "gameId",
        "nucleusHash",
        "future-secret",
    ]:
        # Notes can legitimately mention inventory, so examine the structural keys.
        def keys(value):
            if isinstance(value, dict):
                return set(value) | set().union(*(keys(v) for v in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(v) for v in value))
            return set()

        assert token not in keys(json.loads(text))
    assert [t.team_id for t in state.opponents] == [3, 4]
    assert "future-secret" not in text


def test_event_stats_not_summary_stats(replay):
    assert reconstruct(replay, 2, 89).target_team.kills == 0
    state = reconstruct(replay, 2, 90)
    assert state.target_team.kills == 1
    assert state.squads_remaining == 2
    assert player(state, "Alliance_A").damage_dealt == 35


@pytest.mark.parametrize("at", [-1, 121, float("nan"), float("inf")])
def test_invalid_time(replay, at):
    with pytest.raises(ValueError):
        reconstruct(replay, 2, at)


def test_malformed_replay_rejected(raw):
    raw["events"][0]["tsGame"] = "tomorrow"
    with pytest.raises(ValidationError):
        Replay.from_dict(raw)


def test_missing_optional_sections_warn(raw):
    for key in ["pathing", "timeline", "deathboxes", "rings", "pois"]:
        del raw[key]
    replay = Replay.from_dict(raw)
    assert len(replay.warnings) >= 5
    assert reconstruct(replay, 2, 10).ring is None


def test_duplicate_team_and_unknown_team_rejected(raw):
    raw["summary"]["teams"].append(raw["summary"]["teams"][0])
    with pytest.raises(ValueError, match="duplicate team"):
        Replay.from_dict(raw)


def test_unknown_team_selection(replay):
    with pytest.raises(ValueError, match="not found"):
        replay.team("not-a-team")
