"""Exercise real terminal controls and a small terminal's forecast layout."""

import io
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest
from rich.console import Console

from apexbench.cache import PredictionCache
from apexbench.jev import InferenceBudget
from apexbench.playback import frame
from apexbench.prediction import PredictionService
from apexbench.replay import Replay
from apexbench.state import reconstruct


def test_compact_forecast_fits_80_by_24(replay, mock_api, tmp_path):
    provider, _ = mock_api
    with PredictionCache(tmp_path / "layout.sqlite3") as cache:
        record, _ = PredictionService(replay, cache, provider, InferenceBudget(1)).predict(2, 25)
    output = io.StringIO()
    console = Console(file=output, width=80, height=24, color_system=None)
    console.print(
        frame(
            reconstruct(replay, 2, 25),
            record,
            1,
            True,
            compact=True,
            budget=InferenceBudget(120, requests_used=34),
        )
    )
    assert len(output.getvalue().splitlines()) <= 24
    assert "Jev @ 25s" in output.getvalue()
    assert "q quit" in output.getvalue()
    assert "API requests: 34/120 used · 86 remaining" in output.getvalue()


@pytest.mark.skipif(os.name == "nt", reason="POSIX pseudo-terminal test")
@pytest.mark.parametrize("top", [None, 10])
@pytest.mark.parametrize("cached", [True, False])
def test_pause_step_rewind_quit_restore_terminal(tmp_path, top, mock_api, cached):
    import pty
    import termios

    master, slave = pty.openpty()
    initial = termios.tcgetattr(slave)
    termios.tcsetwinsize(slave, (24, 80))
    fixture = Path(__file__).parent / "fixtures" / "replay.json"
    cache_path = tmp_path / "terminal.sqlite3"
    provider, _ = mock_api
    with PredictionCache(cache_path) as cache:
        service = PredictionService(Replay.load(fixture), cache, provider, InferenceBudget(9))
        for timestamp in [20, 40, 60] if cached else []:
            for team in [2, 3, 4] if top else [2]:
                service.predict(team, timestamp)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "apexbench",
            "replay",
            str(fixture),
            "--start",
            "20",
            "--end",
            "60",
            "--step",
            "20",
            "--speed",
            "0.01",
            "--cache",
            str(cache_path),
            *(["--top", str(top)] if top is not None else []),
        ],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env={**os.environ, "TERM": "xterm-256color"},
        close_fds=True,
    )

    def until(marker):
        output = b""
        deadline = time.monotonic() + 5
        while marker not in output and time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                output += os.read(master, 65536)
        assert marker in output, output.decode(errors="replace")[-2000:]

    try:
        until(b"PLAYING" if cached else b"Replay stopped at 20s")
        os.write(master, b"p")
        until(b"PAUSED" if cached else b"Replay stopped at 20s")
        os.write(master, b"n")
        until(b"00040.0s")
        os.write(master, b"b")
        until(b"00020.0s")
        # A burst also checks that reading one key cannot strand another in a
        # Python text buffer. Drain the PTY while awaiting the final screen.
        os.write(master, b"pq")
        deadline = time.monotonic() + 5
        while process.poll() is None and time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                os.read(master, 65536)
        process.wait(timeout=1)
        assert process.returncode == (0 if cached else 2)
        restored = termios.tcgetattr(slave)
        # macOS may mark queued input for reprocessing during restoration.
        # PENDIN is kernel bookkeeping, not a changed terminal configuration.
        restored[3] &= ~getattr(termios, "PENDIN", 0)
        initial[3] &= ~getattr(termios, "PENDIN", 0)
        assert restored == initial
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        os.close(master)
        os.close(slave)
