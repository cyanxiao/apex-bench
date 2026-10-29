# ApexBench

https://github.com/user-attachments/assets/8cc109f9-d807-4497-bf08-6c24cae93bd9

ApexBench is an open-source benchmark for predicting team placements in competitive
Apex Legends. Load a historical ALGS replay, get forecasts from Jev, GPT, Claude,
Gemini, DeepSeek, or other compatible models through OpenRouter, and compare them
with the match results.

Watch forecasts change during terminal playback, follow a single squad or a top-N
leaderboard, and export results for your own experiments.

## Get started

You need **Python 3.12+**, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and an Apex Legends replay JSON file in the [supported format](docs/replay-format.md).
New predictions also require an OpenRouter API key.

From a local checkout:

```sh
uv sync
uv run apexbench --help
```

Choose a replay and inspect its teams:

```sh
REPLAY="/path/to/replay.json"
uv run apexbench inspect "$REPLAY"
```

Set `TEAM` to a team ID from that output, then generate a forecast:

```sh
TEAM=2
export OPENROUTER_API_KEY="your-openrouter-api-key"

uv run apexbench predict "$REPLAY" --team "$TEAM" --at 60
```

Successful predictions are cached, so repeating the same request does not make
another API call. Inspection, state reconstruction, cached playback, and evaluation
work without a key.

## Choose a model

Jev remains the default. Select another model with `--model`:

```sh
uv run apexbench models
uv run apexbench predict "$REPLAY" --team "$TEAM" --at 60 \
  --model openai/gpt-4.1-mini
uv run apexbench predict "$REPLAY" --team "$TEAM" --at 60 \
  --model anthropic/claude-sonnet-4.5
uv run apexbench predict "$REPLAY" --team "$TEAM" --at 60 \
  --model google/gemini-2.5-flash
```

Other OpenRouter chat model IDs are accepted when they support
[JSON Schema structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs).
All models use the same `OPENROUTER_API_KEY`. Pass the same `--model` when running
`predict`, `precompute`, `replay`, and `evaluate`; each model has its own cached forecasts.
See the [usage guide](docs/usage.md#choose-a-model) for more examples and requirements.

## Watch a match

Show the top ten active teams with live predictions:

```sh
uv run apexbench replay "$REPLAY" \
  --top 10 --step 30 --live-predictions --max-requests 120
```

The leaderboard shows kills, expected placement, win probability, and forecast age.
Teams rank by expected placement, with lower values ranked first. Top-N mode
forecasts all active teams to determine the ranking; `--max-requests` sets the
total API attempt budget for the session.

Playback pauses with an explanation if a checkpoint cannot be completed. Press
**space/p** to pause or resume, **c** to retry a stopped checkpoint, or **q** to quit.
See the [usage guide](docs/usage.md) for every option and control.

## Run a benchmark

Generate forecasts every 60 game seconds, then score them against final placement:

```sh
uv run apexbench precompute "$REPLAY" --team "$TEAM" \
  --interval 60 --max-requests 20 --output exports/predictions.jsonl

uv run apexbench evaluate "$REPLAY" --team "$TEAM" \
  --predictions exports/predictions.jsonl --output exports/evaluation.json
```

ApexBench reports **Brier score** and **log loss**; lower scores are better. Prediction
exports support JSON and JSONL, and evaluation exports support JSON and CSV.
If a run reaches its budget, cached progress is preserved for the next run.

## Commands

| Command | What it does |
| --- | --- |
| `inspect` | Explore replay metadata, teams, and available fields. |
| `state` | View a team's observations at a chosen match time. |
| `models` | List example model IDs. |
| `predict` | Get or reuse a placement forecast from the selected model. |
| `replay` | Watch a match with forecasts in the terminal. |
| `precompute` | Build a cache of forecasts at regular intervals. |
| `evaluate` | Score predictions and export benchmark results. |

Run `uv run apexbench COMMAND --help` for command-specific options.

## Documentation

- [Usage guide](docs/usage.md) — commands, playback controls, budgets, and troubleshooting.
- [Replay format](docs/replay-format.md) — JSON sections, fields, timestamps, and examples.
- [Methodology](docs/methodology.md) — observation modes, forecasting inputs, and scoring.
- [Architecture](docs/architecture.md) — parser, model adapters, caching, and export formats.
- [Contributing](docs/development.md) — development setup, tests, and project structure.

Version 0.1 supports local replay JSON and OpenRouter model inference. See the
methodology notes for the scope and limitations of the benchmark.

## Contributing

Contributions are welcome. Start with the [development guide](docs/development.md)
for setup and verification commands.

ApexBench is released under the [Apache License 2.0](LICENSE).
