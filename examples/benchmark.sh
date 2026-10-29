#!/bin/sh
# ApexBench: local workflow. Supply a replay path, team, and optional model ID.
set -eu
REPLAY=${1:?Usage: sh examples/benchmark.sh /path/to/replay.json TEAM [MODEL]}
TEAM=${2:?Usage: sh examples/benchmark.sh /path/to/replay.json TEAM [MODEL]}
MODEL=${3:-typesafe/jev-1.13}
uv run apexbench inspect "$REPLAY"
uv run apexbench state "$REPLAY" --team "$TEAM" --at 0 --output exports/state.json
uv run apexbench precompute "$REPLAY" --team "$TEAM" --model "$MODEL" --interval 120 --dry-run
uv run apexbench precompute "$REPLAY" --team "$TEAM" --model "$MODEL" --interval 120 --max-requests 10 \
  --output exports/predictions.jsonl
uv run apexbench replay "$REPLAY" --team "$TEAM" --model "$MODEL" --plain --step 120
uv run apexbench evaluate "$REPLAY" --team "$TEAM" --model "$MODEL" --output exports/evaluation.json
uv run apexbench evaluate "$REPLAY" --team "$TEAM" --model "$MODEL" --output exports/evaluation.csv
