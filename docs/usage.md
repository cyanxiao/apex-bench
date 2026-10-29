# Usage guide

[Back to the README](../README.md)

All commands take a local Apex Legends replay JSON file in the
[supported format](replay-format.md). Examples use:

```sh
REPLAY="/path/to/replay.json"
TEAM=2  # Replace with an ID listed by inspect.
```

Times accept seconds, `MM:SS`, or `HH:MM:SS`, measured from gameplay start.
Team selectors accept numeric IDs or exact, case-insensitive team names.

## Explore a replay

```sh
uv run apexbench inspect "$REPLAY"
uv run apexbench inspect "$REPLAY" --json
uv run apexbench state "$REPLAY" --team "$TEAM" --at 01:00
uv run apexbench state "$REPLAY" --team "$TEAM" --at 60 \
  --mode oracle --output exports/state.json
```

`inspect` includes final results. `state` returns observations available at the
requested time. See [Methodology](methodology.md) for observation policies.

## Generate a forecast

Export an OpenRouter key in the shell running ApexBench:

```sh
export OPENROUTER_API_KEY="your-openrouter-api-key"
uv run apexbench predict "$REPLAY" --team "$TEAM" --at 60 \
  --output exports/prediction.json
uv run apexbench predict "$REPLAY" --team "$TEAM" --at 60 --cache-only
```

ApexBench reads `OPENROUTER_API_KEY` from its process environment. It does not load
`.env` files automatically. Cached forecasts work without a key.

The default model is `typesafe/jev-1.13`. Use `--model` to select another pinned
Jev version. Version 0.1 accepts Jev models only and excludes router/latest aliases.

## Terminal replay

### Single team

```sh
uv run apexbench replay "$REPLAY" --team "$TEAM" \
  --step 60 --live-predictions --max-requests 20
```

The panel shows players, legends, survival state, last observed health and shields,
positions, kills, and the team's placement forecast.

### Top-N leaderboard

```sh
uv run apexbench replay "$REPLAY" --top 10 \
  --step 30 --speed 10 --live-predictions --max-requests 120
```

`--top` replaces the single-team view and overrides `--team`. The leaderboard
excludes eliminated squads and shows up to N active teams. Forecasts rank by
expected final placement, then win probability, observed kills, and team ID.
All active teams need forecasts to determine a complete ranking.

| Option | Meaning |
| --- | --- |
| `--team` | Team name or ID for the single-team view. Defaults to Alliance. |
| `--top N` | Display up to N active teams in a leaderboard. |
| `--start`, `--end` | Playback bounds. Defaults to the full match. |
| `--step` | Forecast checkpoint interval in game seconds; defaults to 30. |
| `--speed` | Playback speed multiplier; defaults to 10. |
| `--live-predictions` | Allow API calls for missing forecasts. Off by default. |
| `--max-requests` | Maximum API attempts for this invocation; replay defaults to 0. |
| `--max-input-bytes` | Maximum serialized request size; defaults to 32,000 bytes. |
| `--mode` | `team` or `oracle`; defaults to `team`. |
| `--cache` | SQLite cache path; defaults to `.apexbench/predictions.sqlite3`. |
| `--model` | Pinned Jev model version. |
| `--plain` | Print checkpoints without waiting or terminal controls. |

`APEXBENCH_CACHE` also sets the cache path. Use the same cache, model, and observation
mode when generating, replaying, and evaluating a set of forecasts.

### Controls

| Key | Action |
| --- | --- |
| Space / `p` | Pause or resume. Resuming a stopped checkpoint also retries it. |
| `c` | Reload the cache and retry the current checkpoint. |
| `n` / `l` | Next checkpoint. |
| `b` / `h` | Previous checkpoint. |
| `r` | Return to the beginning of the playback range. |
| `+` / `-` | Increase or decrease playback speed. |
| `q` | Quit. |

The playhead pauses at the end. Non-TTY and `TERM=dumb` sessions use plain output.
Compact terminals show three likely placements in single-team mode; taller
terminals show five. Use a taller terminal or `--plain` for large leaderboards.

### Checkpoints and missing forecasts

Checkpoints are anchored at `--start`. For example, `--start 60 --step 30` requests
forecasts at 60, 90, 120 seconds, and so on. Each required team needs a forecast
for that exact checkpoint. Those forecasts cover playback until the next one.
Older or future forecasts do not substitute for a missing checkpoint.

If live inference is enabled, ApexBench fills the missing entries. It first checks
that the remaining request budget can cover all of them. If it cannot, playback
pauses before making more calls. A completely cached checkpoint can play with a
zero request budget. Resolved team or match outcomes need no new predictions.

A stopped replay shows the checkpoint, missing teams, and reason. Rankings stay
hidden until every active team has a valid forecast. Successful forecasts remain
cached after a failure. Press `c` to retry missing entries or reload forecasts
generated in another terminal. Requests are not retried automatically.

Plain playback stops at the first incomplete checkpoint. Interactive playback
allows you to retry, rewind, or quit. Increasing playback speed does not skip
checkpoint checks.

### Read the counters

- `Forecasts @ 60s: 16/16` means all 16 active squads have a forecast for the
  60-second checkpoint. This counts all active squads, even with `--top 10`.
- `API requests: 34/120 used · 86 remaining` counts HTTP attempts in this invocation.
  Cache hits are free; failed attempts count. Pausing, retrying, or rewinding does
  not reset this counter. A new command starts a new budget.
- `60s / 1.25s` in the forecast/age column means a forecast from match time 60
  seconds, currently 1.25 game seconds old. Ages always use two decimal places.

## Precompute and export

```sh
uv run apexbench precompute "$REPLAY" --team "$TEAM" --interval 60 --dry-run
uv run apexbench precompute "$REPLAY" --team "$TEAM" \
  --interval 60 --max-requests 20 --output exports/predictions.jsonl
uv run apexbench replay "$REPLAY" --team "$TEAM" --step 60
uv run apexbench evaluate "$REPLAY" --team "$TEAM" \
  --predictions exports/predictions.jsonl --output exports/evaluation.json
uv run apexbench evaluate "$REPLAY" --team "$TEAM" \
  --predictions exports/predictions.jsonl --output exports/evaluation.csv
```

`precompute` supports `--start` and `--end`, skips resolved outcomes, and resumes
from exact cache hits. Use the same `--interval` and replay `--step` when preparing
a playback cache. Prediction exports use `.json` or `.jsonl`; evaluations use
`.json` or `.csv`. Without `--predictions`, evaluation reads the cache.

## Budgets and troubleshooting

`predict` allows one API attempt by default, `precompute` allows ten, and `replay`
allows zero. Replay also requires `--live-predictions` before making calls.
`--max-requests` applies across all teams and checkpoints in a command. With 20
active teams, one uncached top-N checkpoint needs 20 attempts. Budgets count
requests, not timestamps. Input-size and request-count limits are not dollar caps.
Provider usage/cost is retained in successful prediction records.

| Situation | What to do |
| --- | --- |
| Key is not configured | Export `OPENROUTER_API_KEY` in the shell launching ApexBench. |
| Cache is incomplete | Generate the missing checkpoints or enable live predictions. |
| Request budget is insufficient | Rerun with a larger budget; cached entries are reused. |
| Request exceeds the input limit | Adjust `--max-input-bytes` if appropriate for the experiment. |
| Provider response fails validation | Read the specific error and retry with `c`. Repeated failures remain stopped. |
| OpenRouter returns an HTTP error | Check access, model availability, credits, or rate limits before retrying. |

Validation failures identify the team, checkpoint, and check that failed. The CLI
saves the request and response text under `diagnostics/` beside the cache and
prints the file path. `predict` and `precompute` accept `--timeout` in seconds,
with a default of 30.

Exit status is 0 for success, 1 for errors, and 2 when a precompute budget prevents
completion or replay stops at an incomplete checkpoint. Quitting interactive
playback while blocked also exits 2.
