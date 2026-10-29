# Benchmark methodology

[Back to the README](../README.md)

ApexBench measures probabilistic final-placement forecasts for a chosen team at
historical match checkpoints. A forecast contains one probability per placement,
from first to the number of teams in the match.

## Observation modes

`--mode team` is the default. It includes the selected team's timestamped observer
telemetry, roster identities, public kill counts, team-addressed squad eliminations,
and observed ring updates. Opponent health, positions, weapons, individual survival
state, and damage are withheld. Opponent players display `alive` until their squad
is eliminated; here it means not known eliminated, rather than continuously observed
alive.

`--mode oracle` exposes every team's past observer telemetry and deathbox locations.
It is explicitly labeled and uses a separate cache namespace. It still excludes
future observations. Neither mode is an exact first-person visibility simulation.

## Reconstructing observations

The state includes events at or before the requested timestamp. Negative-time
pregame events initialize identities and last observed telemetry. Teams and players
are sorted by ID and name, avoiding the summary's final-placement ordering.
Legends come from timestamped identities, rather than final summary characters.

Positions use the last observed path sample at or before the cutoff. ApexBench
does not interpolate between samples or use future points. Death and respawn
invalidate stale path samples, and respawn clears old health, shields, and weapons.
Vitals are last observed values, not continuously measured current values.

Combat statistics come from raw events. Timeline squad eliminations provide team
IDs that may be absent from raw elimination events. Ring state uses records
already emitted; the center of a future finished ring is never used to infer its
current position. Deathboxes expose only their timestamp, victim, and location.
Retrospective respawn annotations and inventory are excluded.

## Information excluded from forecasts

Prediction inputs omit final placement, winner, cumulative summary statistics,
summary counters, match ID, date/header, exact duration/end time, fetch time,
and future ring phases. Record provenance can retain match identifiers outside
the request sent to the model.

Aggregated fight windows are parsed but excluded because their endpoints and
participants can depend on future events. POI polygons are inspectable but are
not used to assign locations without a verified transform to pathing coordinates.
Inventory and other incompletely observed systems are not fabricated.

Predictions after target elimination, one squad remaining, or match end are
rejected and excluded from forecasting metrics. Replay and state inspection can
still show resolved outcomes.

## Scoring

For placement probabilities `p_k` and actual final placement `y`:

| Metric | Definition | Interpretation |
| --- | --- | --- |
| Brier score | `sum_k (p_k - 1[k=y])²` | Unnormalized multiclass score, from 0 to 2. Lower is better. |
| Log loss | `-ln(max(p_y, 1e-15))` | Natural logarithm; lower is better. |

The evaluation report counts probabilities clipped by the log-loss floor. Means
give each unique checkpoint equal weight. Duplicate prediction imports do not
increase their weight. Teams, observation modes, models, and prompt versions are
evaluated separately.

Imported records are checked against the replay fingerprint and freshly
reconstructed input before scoring. See [Architecture](architecture.md) for
record and cache formats.

## Interpretation and limitations

- Checkpoints within one match are correlated. Their average scores do not establish
  benchmark-wide calibration or generalization.
- Team and player names remain in observations, so recognition of historical matches
  by a model is a possible source of contamination.
- Observer telemetry can be incomplete or stale. Complete forecast coverage means
  that every required team has a prediction, not that every game-state field is known.
- Raw events and derived timeline summaries may disagree. The reconstruction policy
  above defines which source supplies each observation.
- Placement probabilities describe the selected model's forecasts; they are not a guarantee of
  calibrated uncertainty for competitive Apex Legends.
- Jev returns native Decisions probabilities. Chat models generate probability
  estimates under a JSON Schema using the same observations and forecasting
  instruction. These are different inference methods; record the model and adapter
  when comparing their scores. ApexBench never silently normalizes either output.

Record the replay, model version, observation mode, checkpoint schedule, and
prediction export when comparing experiments.
