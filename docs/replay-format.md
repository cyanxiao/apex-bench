# Apex Legends replay JSON format

[Back to the README](../README.md)

This guide describes the Apex Legends replay JSON file format accepted by
ApexBench v0.1. It is a JSON object containing match metadata and timestamped
observations. Exports with a different structure need conversion to this format.

Field names below use the JSON spelling accepted by the parser. The schemas in
[`models.py`](../src/apexbench/models.py) define field types; additional consistency
checks live in [`replay.py`](../src/apexbench/replay.py).

## Minimal example

Save this as a JSON file and use `apexbench inspect` or `apexbench state` to read it.
The example is synthetic and uses one player per team for brevity. Real match
files should include the full roster and available observations.

```json
{
  "summary": {
    "gameId": "example-match",
    "map": "example-map",
    "duration": 120,
    "teams": [
      {
        "teamId": 1,
        "teamName": "Team Alpha",
        "players": [{"name": "Alpha Player", "teamId": 1}],
        "placement": 1
      },
      {
        "teamId": 2,
        "teamName": "Team Beta",
        "players": [{"name": "Beta Player", "teamId": 2}],
        "placement": 2
      }
    ]
  },
  "events": [
    {
      "id": 1,
      "tsGame": 0,
      "category": "characterSelected",
      "actor": {"name": "Alpha Player", "teamId": 1, "character": "Bangalore"}
    },
    {
      "id": 2,
      "tsGame": 0,
      "category": "characterSelected",
      "actor": {"name": "Beta Player", "teamId": 2, "character": "Alter"}
    }
  ],
  "pathing": {"teams": []},
  "timeline": {"squadEliminations": []},
  "deathboxes": {"deathboxes": []},
  "rings": {"rings": []},
  "pois": {"payload": {"polygons": []}}
}
```

Only `summary` and `events` are required at the top level. An empty event list is
valid, though it supplies no event observations. Optional sections can be omitted
or supplied as empty objects. Omitting them produces inspection warnings; it does
not prevent loading. Empty or absent observations limit the reconstructed state.

## Top-level sections

| Section | Type | Required | Contents |
| --- | --- | --- | --- |
| `summary` | Object | Yes | Match identity, map, duration, roster, and optional final placements. |
| `events` | Array of event objects | Yes | Timestamped identities, combat, telemetry, and other events. |
| `pathing` | Object | No | Player position samples under `teams`. |
| `timeline` | Object | No | Derived event lists and team-addressed squad eliminations. |
| `deathboxes` | Object | No | Deathbox observations under `deathboxes`. |
| `rings` | Object | No | Ring observations under `rings`. |
| `pois` | Object | No | Named polygons under `payload.polygons`. |

Unknown fields are accepted and retained for inspection. For example, exports may
also contain `gid`, `fetchedAt`, absolute timestamps, or display metadata. These
fields are not automatically passed to the prediction model.

An optional field can be omitted; this does not necessarily mean it accepts
`null`. Use `null` only for fields marked nullable below. Lists and section objects
should be empty arrays/objects when there are no entries, rather than `null`.

## Time and identity conventions

- `tsGame` is a finite number of seconds relative to gameplay start. Integers and
  fractional seconds are supported. Negative values represent pregame observations.
- Events at exactly a requested timestamp are included. Events are sorted by
  `(tsGame, id)`; path samples are sorted by `tsGame`.
- State query times must be between zero and `summary.duration`, inclusive.
  Absolute timestamps such as `tsAbs` do not replace `tsGame`.
- Team IDs must be unique in the summary. They need not be consecutive or equal
  to placement numbers.
- Player names must be unique within each team. Event identities are matched by
  `(teamId, name)`, and pathing identities by `(teamId, playerName)`. Keep these
  values consistent with the summary roster.
- Coordinates are numeric `x` and `y`, with optional nullable `z`. ApexBench retains
  their source units; it does not assume that POI and pathing coordinates share a
  coordinate system.

## `summary`

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `gameId` | Nonempty string | Yes | Identifier for the match. |
| `map` | String | Yes | Map identifier. |
| `mapDisplayName` | String or null | No | Human-readable map name for inspection. |
| `duration` | Finite number, ≥ 0 | Yes | Match duration in seconds. |
| `teams` | Array, at least two entries | Yes | Team roster. |

Each team has:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `teamId` | Integer | Yes | Team ID, unique within the match. |
| `teamName` | String | Yes | Display name. |
| `players` | Array, at least one entry | Yes | Players belonging to this team. |
| `placement` | Integer or null | No | Actual final placement, from 1 to the number of teams. Required to evaluate this team. |

Each player requires `name`. Optional nullable fields are `teamId` (integer),
`character` (string), and `nucleusHash` (string). The same player object shape is
used for event actors/targets and deathbox victims. `nucleusHash` is preserved when
present but is not used as the reconstruction lookup key.

Summary fields such as `winner`, team `stats`, counters, and ring phases may be
present as extra metadata. They are excluded from historical prediction inputs.

## `events`

Every event requires `tsGame` and `category`. Its `id` defaults to 0 if omitted;
provide unique IDs to make ordering at equal timestamps explicit.

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | Integer | Ordering within an equal timestamp. |
| `tsGame` | Finite number | Match-relative observation time. |
| `category` | String | Event category. |
| `actor`, `target` | Player object or null | Involved players; include matching `teamId` and `name` to apply observations. |
| `posActor`, `posTarget` | Position object or null | Observed coordinates. |
| `actorHealth`, `actorShield` | Finite number or null | Last observed actor vitals. |
| `targetHealth`, `targetShield` | Finite number or null | Last observed target vitals. |
| `damage` | Finite number or null | Damage for a `playerDamaged` event. |
| `weapon` | String or null | Weapon metadata for the event. |
| `newWeapon` | String or null | Equipped weapon after `weaponSwitched`. |
| `respawnedTeammateNames` | Array of strings | Players restored by `playerRespawnTeam`; defaults to `[]`. |
| `respawnedName` | String or null | Alternative single-player respawn field. |

Actor/target, position, vitals, damage, and weapon fields default to `null`.
Reconstruction handles these categories:

| Category | Effect |
| --- | --- |
| `characterSelected` | Supplies a timestamped legend identity. Actor/target identities on other categories also update legends. |
| `weaponSwitched` | Updates the actor's equipped weapon using `newWeapon`. |
| `playerDamaged` | Adds the actor's observed damage. |
| `playerDowned` | Marks the target downed. |
| `playerKilled` | Marks the target dead and credits an identified opposing actor's team with a kill. |
| `playerRevive` | Marks the target alive, clears stale vitals, and credits the actor's revive count. |
| `playerRespawnTeam` | Restores the named players on the actor's team and clears stale telemetry. |

Actor/target positions and vitals are read whenever supplied, regardless of the
category. Unknown categories are accepted, but do not create additional counters
or status transitions. Inventory events, for example, do not build an inventory
model in v0.1. Squad elimination counts use `timeline.squadEliminations`, since
raw `squadEliminated` events can lack a team ID.

## `pathing`

```json
{
  "pathing": {
    "teams": [
      {
        "teamId": 1,
        "players": [
          {
            "playerName": "Alpha Player",
            "points": [
              {"tsGame": 0, "x": 100, "y": 200},
              {"tsGame": 10.5, "x": 120, "y": 210}
            ]
          }
        ]
      }
    ]
  }
}
```

`teams` defaults to `[]`. Each entry requires `teamId`; its `players` array defaults
to `[]`. Each player requires `playerName`, may include nullable `nucleusHash`, and
has a `points` array defaulting to `[]`. Each point requires `tsGame`, `x`, and `y`;
`z` is optional and nullable. Teams and players must exist in the summary, and
duplicate paths for the same team/player are rejected.

State reconstruction selects the latest eligible sample, never interpolating
from a future point.

## `timeline`

```json
{
  "timeline": {
    "squadEliminations": [{"tsGame": 90, "teamId": 2}],
    "kills": [],
    "downs": [],
    "ringEvents": [],
    "respawns": [],
    "revives": [],
    "fightWindows": [
      {"startTsGame": 10, "endTsGame": 20, "teamIds": [1, 2]}
    ]
  }
}
```

All lists default to `[]`. Entries in `squadEliminations`, `kills`, `downs`,
`ringEvents`, `respawns`, and `revives` require `tsGame` and may include `teamId`.
For `squadEliminations`, `teamId` must be present and refer to a summary team.
Other descriptive fields in timeline entries are retained as extra metadata.

Each fight window requires finite `startTsGame` and `endTsGame`, with start no later
than end. Additional fields such as `teamIds` are accepted. Fight windows are
excluded from predictions because they can summarize future observations.
Combat statistics come from raw events, rather than timeline kill/down summaries.

## `deathboxes`

```json
{
  "deathboxes": {
    "deathboxes": [
      {
        "deathboxId": 1,
        "tsGame": 90,
        "victim": {"name": "Beta Player", "teamId": 2},
        "pos": {"x": 300, "y": 400},
        "inventory": []
      }
    ]
  }
}
```

The inner `deathboxes` array defaults to `[]`. Each entry requires `deathboxId`
(integer), `tsGame`, and `victim` (player object). `pos` is optional and nullable.
`inventory` is an optional array of objects, defaulting to `[]`.

Prediction observations include only the deathbox timestamp, victim, and location,
subject to the observation mode. Inventory and retrospective respawn fields such
as `hasRespawned` and `respawnTsGame` are not included. Respawns are reconstructed
from timestamped events.

## `rings`

```json
{
  "rings": {
    "rings": [
      {
        "type": "startClosing",
        "stage": 0,
        "tsGame": 30,
        "center": {"x": 500, "y": 600},
        "startRadius": 1000,
        "endRadius": 500,
        "shrinkDuration": 30
      },
      {
        "type": "finishedClosing",
        "stage": 0,
        "tsGame": 60,
        "center": {"x": 550, "y": 620},
        "currentRadius": 500
      }
    ]
  }
}
```

The inner `rings` array defaults to `[]`. Every record requires `type` (string),
`stage` (nonnegative integer), `tsGame`, and `center` (position). Common types are
`startClosing` and `finishedClosing`. Optional nullable numeric fields are
`startRadius`, `endRadius`, `currentRadius`, and `shrinkDuration`; when present,
they must be finite and nonnegative.

Only records already observed at the cutoff can enter a state. The future
finished circle is not used to infer a closing circle's current center.

## `pois`

```json
{
  "pois": {
    "payload": {
      "polygons": [
        {
          "name": "Example POI",
          "coordinates": [
            {"x": 0, "y": 0},
            {"x": 100, "y": 0},
            {"x": 0, "y": 100}
          ]
        }
      ]
    }
  }
}
```

`payload` defaults to an empty object and `polygons` to `[]`. Each polygon requires
a `name` string and at least three coordinate objects with `x` and `y`; `z` is
optional and nullable. POIs are available for inspection. They are not used to
assign player locations without a verified coordinate transform.

## Validate a file

```sh
uv run apexbench inspect /path/to/replay.json --json
uv run apexbench state /path/to/replay.json --team 1 --at 10
```

Loading rejects missing required fields, malformed typed records, nonfinite
timestamps/coordinates, duplicate team IDs or within-team player names, invalid
placements, unknown pathing teams/players, duplicate player paths, and unknown
teams in squad eliminations. Missing optional observations may instead produce
warnings. Unknown extra fields do not automatically become prediction features.

For how observations become model inputs and how final placements are scored,
see [Methodology](methodology.md).
