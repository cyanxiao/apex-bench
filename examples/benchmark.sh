#!/bin/sh
# ApexBench: local workflow. Supply a replay path and a team name or ID.
set -eu
REPLAY=${1:?Usage: sh examples/benchmark.sh /path/to/replay.json TEAM}
TEAM=${2:?Usage: sh examples/benchmark.sh /path/to/replay.json TEAM}
uv run apexbench inspect "$REPLAY"
uv run apexbench state "$REPLAY" --team "$TEAM" --at 0 --output exports/state.json
uv run apexbench precompute "$REPLAY" --team "$TEAM" --interval 120 --dry-run
uv run apexbench precompute "$REPLAY" --team "$TEAM" --interval 120 --max-requests 10 \
  --output exports/predictions.jsonl
uv run apexbench replay "$REPLAY" --team "$TEAM" --plain --step 120
uv run apexbench evaluate "$REPLAY" --team "$TEAM" --output exports/evaluation.json
uv run apexbench evaluate "$REPLAY" --team "$TEAM" --output exports/evaluation.csv
