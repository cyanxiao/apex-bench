# Contributing to ApexBench

[Back to the README](../README.md)

Contributions to replay support, forecast quality, terminal usability, evaluation,
tests, and documentation are welcome.

## Set up a checkout

Use Python 3.12+ and uv. The repository's `.python-version` selects Python 3.13
for local development, and `uv.lock` pins dependencies.

```sh
uv sync --dev
uv run apexbench --help
```

## Verify changes

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

Run checks appropriate to your changes. Describe the problem, the resulting
behavior, and how you verified it when submitting a contribution. Keep user-facing
examples and help text up to date when behavior changes.

## Package contents

Source distributions use an explicit allowlist in `pyproject.toml`. Wheels contain
the Python application and generated package metadata, including `LICENSE` and
`NOTICE`. Add new public documentation, examples, or fixtures to the source
allowlist when needed; local exports and diagnostic files are not release inputs.

Keep generated results in `exports/`. Root-level JSON, JSONL, and CSV files,
diagnostics, environment files, and common credential files are ignored by Git.
An `.env.example` template may be tracked when it contains placeholders only.
The publication tests build distributions with dummy local files, verify that
those files are excluded, and check that the application and public fixture remain
included. These checks also run without a `.gitignore` file so package exclusions
do not rely solely on Git rules.

## Tests

The default tests use the small replay fixture in `tests/fixtures/replay.json`
and mocked HTTP responses. They do not require an OpenRouter key or paid API calls.
Coverage includes reconstruction boundaries, respawn state, future-information
exclusion, provider validation, caching, budgets, exports, and terminal controls.
CI checks Python 3.12 and 3.13.

Optional full-replay integration tests can be enabled with:

```sh
APEXBENCH_TEST_REPLAY="/path/to/compatibility-replay.json" \
  uv run pytest tests/test_supplied_replay.py
```

These tests assert fixed observations from a separate compatibility replay.
Arbitrary replay files will not satisfy those assertions. They are skipped when
the environment variable is unset. New replay support should include focused
fixtures and tests for the relevant format and behavior.

## Where to start

- Read [Architecture](architecture.md) for the module layout and data flow.
- Read [Methodology](methodology.md) before changing observations or scoring.
- Use [Usage](usage.md) when updating command options, replay behavior, or examples.
- Treat replay fields as input data. Keep prediction observations separate from
  final results and future events.
- Keep prompt, state, and record versions in mind when changing cache identity or
  exported inputs. Existing results must remain interpretable.

The [example benchmark script](../examples/benchmark.sh) runs inspection, state
reconstruction, precomputation, cached playback, and evaluation:

```sh
sh examples/benchmark.sh /path/to/replay.json 2
```

The second argument is a team ID or name from `apexbench inspect`.

## License

ApexBench uses the [Apache License 2.0](../LICENSE).
