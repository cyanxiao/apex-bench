import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from apexbench.cli import app, parse_time

FIXTURE = Path(__file__).parent / "fixtures" / "replay.json"
runner = CliRunner()


def run(*args):
    return runner.invoke(app, [str(arg) for arg in args])


def test_command_help_and_version():
    result = run("--help")
    assert result.exit_code == 0
    for command in ["inspect", "state", "predict", "precompute", "replay", "evaluate"]:
        assert command in result.stdout
        assert run(command, "--help").exit_code == 0
    assert run("--version").stdout.strip() == "ApexBench 0.1.0"


def test_inspect_and_state():
    result = run("inspect", FIXTURE, "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["team_count"] == 3
    result = run("state", FIXTURE, "--at", "00:25", "--team", "ALLIANCE")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["timestamp"] == 25


def test_precompute_predict_replay_evaluate_flow(mock_api, monkeypatch, tmp_path):
    provider, calls = mock_api
    monkeypatch.setattr("apexbench.cli.JevProvider", lambda *a, **kw: provider)
    cache = tmp_path / "cache.sqlite3"
    output = tmp_path / "predictions.jsonl"
    result = run(
        "precompute",
        FIXTURE,
        "--start",
        "0",
        "--end",
        "100",
        "--interval",
        "25",
        "--max-requests",
        "2",
        "--cache",
        cache,
        "--output",
        output,
    )
    assert result.exit_code == 2, result.output
    assert len(calls) == 2
    assert len(output.read_text().splitlines()) == 2
    # Resume fills only missing checkpoints.
    result = run(
        "precompute",
        FIXTURE,
        "--end",
        "100",
        "--interval",
        "25",
        "--max-requests",
        "3",
        "--cache",
        cache,
        "--output",
        output,
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 5
    result = run("predict", FIXTURE, "--at", "25", "--cache-only", "--cache", cache, "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["cache_hit"] is True
    assert len(calls) == 5
    result = run(
        "replay",
        FIXTURE,
        "--start",
        "0",
        "--end",
        "50",
        "--step",
        "25",
        "--plain",
        "--cache",
        cache,
    )
    assert result.exit_code == 0, result.output
    assert "Jev @ 0s" in result.stdout
    assert "Jev @ 25s" in result.stdout
    assert "Jev @ 50s" in result.stdout
    report = tmp_path / "evaluation.json"
    result = run("evaluate", FIXTURE, "--predictions", output, "--output", report, "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["count"] == 5
    assert json.loads(report.read_text())["rows"][0]["actual_placement"] == 1
    assert len(calls) == 5


def test_precompute_dry_run_and_default_replay_no_api(mock_api, monkeypatch, tmp_path):
    provider, calls = mock_api
    monkeypatch.setattr("apexbench.cli.JevProvider", lambda *a, **kw: provider)
    cache = tmp_path / "cache.sqlite3"
    result = run("precompute", FIXTURE, "--dry-run", "--cache", cache)
    assert result.exit_code == 0, result.output
    assert "missing" in result.stdout
    result = run("replay", FIXTURE, "--plain", "--step", "60", "--cache", cache)
    assert result.exit_code == 2, result.output
    assert "Replay stopped at 0s" in result.stdout
    assert "Cache incomplete" in result.stdout
    assert calls == []


def test_live_replay_obeys_budget(mock_api, monkeypatch, tmp_path):
    provider, calls = mock_api
    monkeypatch.setattr("apexbench.cli.JevProvider", lambda *a, **kw: provider)
    result = run(
        "replay",
        FIXTURE,
        "--plain",
        "--step",
        "25",
        "--end",
        "100",
        "--live-predictions",
        "--max-requests",
        "1",
        "--cache",
        tmp_path / "c.sqlite3",
    )
    assert result.exit_code == 2, result.output
    assert len(calls) == 1
    assert "budget exhausted" in result.stdout
    assert "API requests: 1/1 used · 0 remaining" in result.stdout
    assert "Replay stopped at 25s" in result.stdout
    assert "00050.0s" not in result.stdout


def test_missing_key_and_errors(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    result = run("predict", FIXTURE, "--at", "25", "--cache", tmp_path / "c.sqlite3")
    assert result.exit_code == 1
    assert "OPENROUTER_API_KEY" in result.output
    assert run("state", FIXTURE, "--at", "nan").exit_code == 1
    assert run("state", FIXTURE, "--team", "unknown").exit_code == 1
    result = run("precompute", FIXTURE, "--max-requests", "0", "--cache", tmp_path / "c.sqlite3")
    assert result.exit_code == 2
    assert "budget exhausted" in result.output


def test_invalid_export_fails_before_spending(mock_api, monkeypatch, tmp_path):
    provider, calls = mock_api
    monkeypatch.setattr("apexbench.cli.JevProvider", lambda *a, **kw: provider)
    for command in ["predict", "precompute"]:
        result = run(
            command,
            FIXTURE,
            "--output",
            tmp_path / "invalid.csv",
            "--cache",
            tmp_path / "c.sqlite3",
        )
        assert result.exit_code == 1
    assert calls == []


@pytest.mark.parametrize(
    ("value", "expected"), [("123.5", 123.5), ("07:30", 450), ("1:02:03", 3723)]
)
def test_timestamp_formats(value, expected):
    assert parse_time(value) == expected


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "1:60", "1.5:30", "1:2:3:4", ""])
def test_invalid_timestamp_formats(value):
    with pytest.raises(ValueError):
        parse_time(value)
