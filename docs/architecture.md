# Architecture

[Back to the README](../README.md)

ApexBench v0.1 uses Python, Typer, Rich, Pydantic, httpx, and SQLite. Its required
match-data source is a local Apex Legends replay JSON file.

## Modules

| Module | Responsibility |
| --- | --- |
| `models.py` | Replay schemas, observation models, and forecast validation. |
| `replay.py` | File loading, fingerprinting, metadata inspection, and indexed observations. |
| `state.py` | Deterministic projection into timestamped team or oracle observations. |
| `jev.py` | Request construction, OpenRouter calls, budgets, response checks, and diagnostics. |
| `cache.py` | SQLite persistence and portable prediction records. |
| `prediction.py` | Cache-first inference and checkpoint generation. |
| `playback_session.py` | Complete-checkpoint checks, generation, stopping, and retry state. |
| `playback.py` | Terminal views, controls, and playhead progression. |
| `evaluation.py` | Scoring and prediction/evaluation imports and exports. |
| `cli.py` | Command definitions and configuration. |

## Replay parsing

The [replay format guide](replay-format.md) describes the accepted JSON structure,
required fields, and examples.

The parser validates `summary` and `events`, and supports `pathing`, `timeline`,
`deathboxes`, `rings`, and `pois`. Optional missing sections produce inspection
warnings. Unknown raw fields remain available for inspection.

Prediction inputs use separate, allowlisted models with extra fields forbidden.
No raw replay section is passed directly to Jev. Events are ordered by timestamp
and event ID; path samples are indexed for lookups at a cutoff time. See
[Methodology](methodology.md) for the observation policy.

## Jev integration

ApexBench calls OpenRouter's native
[`POST /api/alpha/decisions`](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request)
endpoint using a Choice question with one alternative per final placement.
The adapter reads `answers.placement.probabilities`, rather than parsing a
generated explanation. The [Jev tutorial](https://openrouter.ai/docs/guides/community/jev-tutorial)
describes this response format.

Validation requires exact class coverage, numeric finite probabilities in [0,1],
a sum of 1 within an absolute tolerance of 1e-6, a valid choice, and a matching
resolved Jev model. Failed responses are rejected, not silently normalized or
replaced with another model. The CLI can save response diagnostics for investigation.

`PlacementProvider` is the extension point for future model adapters. Only pinned
Jev versions are accepted in v0.1. Requests have a size limit, timeout, and attempt
budget; failures consume attempts and there are no hidden HTTP retries.

## Prediction cache

The default cache is `.apexbench/predictions.sqlite3`. A cache key includes:

- The SHA256 fingerprint of the entire replay file.
- The canonical request: model, observations, and question.
- The prompt version.

Changing the replay, timestamp, observation mode, model, state schema, or prompt
therefore changes the cache identity. Writes are transactional and successful
identical requests are immutable. Separate processes can race on a cache miss
and each make a request; budgets apply per invocation.

## Playback completeness

A playback session looks up exact forecasts for the scheduled checkpoint. In
top-N mode it requires all active teams, even when only N rows are displayed.
Before inference, it checks whether the remaining attempt budget can cover every
missing forecast. Successful results are saved individually; a provider failure
stops the session without discarding earlier results.

The terminal withholds rankings while a checkpoint is loading or incomplete.
Automatic progression cannot skip unchecked checkpoints. An explicit retry
reloads the cache, clears that checkpoint's saved failure, and fills remaining
gaps. Older and future forecasts cannot satisfy a missing checkpoint. Outcomes
already resolved do not require inference.

## Export formats

JSON and JSONL prediction exports contain the full request, raw response, requested
and resolved model, prompt/state versions, UTC creation time, replay fingerprint,
match/team identifiers, timestamp, and validated distribution. On import,
provenance, cache identity, and consistency between the forecast and raw response
are validated.

JSON evaluations contain prediction records, metric definitions, per-checkpoint
scores, and aggregate means. CSV evaluations contain per-checkpoint provenance,
scores, and `p_1` through `p_N` columns. Keep a JSON prediction or evaluation export
alongside CSV when full request/response reproducibility is needed.
