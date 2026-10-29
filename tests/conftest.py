import json
from pathlib import Path

import httpx
import pytest

from apexbench.jev import DEFAULT_MODEL, JevProvider
from apexbench.replay import Replay

FIXTURE = Path(__file__).parent / "fixtures" / "replay.json"


@pytest.fixture
def raw():
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def replay(raw):
    return Replay.from_dict(raw)


@pytest.fixture
def mock_api():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        count = len(body["questions"]["placement"]["criteria"])
        return httpx.Response(
            200,
            json={
                "id": "test-only-mock",
                "model": DEFAULT_MODEL + "-20260917",
                "provider": "TypeSafe",
                "answers": {
                    "placement": {
                        "type": "choice",
                        "choice": "1",
                        "confidence": 0.5,
                        "probabilities": {str(i): 1.0 / count for i in range(1, count + 1)},
                    }
                },
                "usage": {"input_tokens": 100, "output_tokens": 0, "cost": 0.0000042},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        yield JevProvider("test-key-never-real", client=client), calls
