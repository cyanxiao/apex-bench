import copy
import math

import pytest

from apexbench.cache import PredictionCache, PredictionRecord, cache_key
from apexbench.evaluation import (
    EPSILON,
    evaluate,
    export_evaluation,
    export_predictions,
    import_predictions,
    scores,
)
from apexbench.jev import InferenceBudget
from apexbench.models import Forecast
from apexbench.prediction import PredictionService
from apexbench.replay import Replay


def test_scoring_conventions():
    assert scores(Forecast(probabilities=[1.0, 0.0]), 1) == {"brier_score": 0, "log_loss": 0}
    uniform = scores(Forecast(probabilities=[0.5, 0.5]), 1)
    assert uniform["brier_score"] == 0.5
    assert uniform["log_loss"] == pytest.approx(math.log(2))
    wrong = scores(Forecast(probabilities=[0.0, 1.0]), 1)
    assert wrong["brier_score"] == 2
    assert wrong["log_loss"] == -math.log(EPSILON)
    with pytest.raises(ValueError):
        scores(Forecast(probabilities=[0.5, 0.5]), 3)


def test_export_roundtrip_and_reproducible_evaluation(replay, mock_api, tmp_path):
    provider, calls = mock_api
    with PredictionCache(tmp_path / "cache.sqlite3") as cache:
        service = PredictionService(replay, cache, provider, InferenceBudget(2))
        records = [service.predict(2, t)[0] for t in [25, 60]]
    for suffix in ["json", "jsonl"]:
        path = tmp_path / f"predictions.{suffix}"
        export_predictions(records, path)
        imported = import_predictions(path)
        assert imported == records
        report = evaluate(replay, imported + imported)
        assert report["count"] == 2
        assert report["mean_brier_score"] == pytest.approx(2 / 3)
        assert report["mean_log_loss"] == pytest.approx(math.log(3))
        assert report["rows"][0]["actual_placement"] == 1
        for extension in ["csv", "json"]:
            output = tmp_path / f"evaluation.{extension}"
            export_evaluation(report, output)
            assert "replay_sha256" in output.read_text()
        assert len(calls) == 2  # Scoring/import/export is offline.
    assert import_predictions(tmp_path / "evaluation.json") == records


def test_wrong_replay_rejected(replay, mock_api, tmp_path, raw):
    provider, _ = mock_api
    with PredictionCache(tmp_path / "c.sqlite3") as cache:
        record = PredictionService(replay, cache, provider, InferenceBudget(1)).predict(2, 25)[0]
    raw["summary"]["map"] = "different_map"
    with pytest.raises(ValueError, match="SHA256"):
        evaluate(Replay.from_dict(raw), [record])


def test_reconstructed_input_verification(replay, mock_api, tmp_path):
    provider, _ = mock_api
    with PredictionCache(tmp_path / "c.sqlite3") as cache:
        record = PredictionService(replay, cache, provider, InferenceBudget(1)).predict(2, 25)[0]
    altered = copy.deepcopy(record.model_dump())
    altered["request"]["state"]["target_team"]["kills"] = 999
    altered["cache_key"] = cache_key(replay.fingerprint, altered["request"])
    record = PredictionRecord.model_validate(altered)
    with pytest.raises(ValueError, match="historical state"):
        evaluate(replay, [record])


def test_mixed_observation_policies_rejected(replay, mock_api, tmp_path):
    provider, _ = mock_api
    with PredictionCache(tmp_path / "c.sqlite3") as cache:
        service = PredictionService(replay, cache, provider, InferenceBudget(2))
        records = [service.predict(2, 25, mode)[0] for mode in ["team", "oracle"]]
    with pytest.raises(ValueError, match="one team"):
        evaluate(replay, records)
