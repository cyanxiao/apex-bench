"""Keep local credentials and experiment data out of commits and distributions."""

import os
import shutil
import subprocess
import sys
import tarfile
import uuid
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GIT_PROTECTED_FILES = [
    ".env",
    ".env.local",
    ".env.production",
    ".netrc",
    ".pypirc",
    "private-key.pem",
    "private-key.key",
    "credentials.p12",
    "credentials.pfx",
    "predictions.json",
    "predictions.jsonl",
    "evaluation.csv",
    "diagnostics/response.json",
    ".apexbench/diagnostics/response.json",
    "exports/predictions.json",
    "experiment.log",
    "src/apexbench/.env.local",
    "src/apexbench/private-key.pem",
    "src/apexbench/diagnostics/response.json",
    ".aws/credentials",
    ".ssh/id_ed25519",
    "logs/run.log",
]
LOCAL_FILES = GIT_PROTECTED_FILES + [
    "src/apexbench/predictions.json",
    "tests/fixtures/customer-replay.json",
    "docs/local-notes.md",
    "private-notes.txt",
]


def test_git_ignores_local_files_but_keeps_public_inputs(tmp_path):
    git = shutil.which("git")
    assert git, "Git is required for publication tests"
    shutil.copy2(ROOT / ".gitignore", tmp_path / ".gitignore")
    subprocess.run([git, "-c", "init.defaultBranch=main", "init", "-q", str(tmp_path)], check=True)
    # Unknown files outside the documented output paths are guarded by the
    # package allowlist; Git rules protect the usual local credential/output files.
    protected = GIT_PROTECTED_FILES
    public = [".env.example", "tests/fixtures/replay.json", "docs/usage.md", "README.md"]
    result = subprocess.run(
        [git, "-C", str(tmp_path), "check-ignore", "--stdin"],
        input="\n".join([*protected, *public]) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    assert set(result.stdout.splitlines()) == set(protected)


@pytest.mark.parametrize("with_gitignore", [True, False])
def test_distributions_exclude_local_files_and_preserve_public_inputs(tmp_path, with_gitignore):
    uv = shutil.which("uv")
    assert uv, "uv is required for publication tests"
    project = tmp_path / "project"
    project.mkdir()
    for name in ["src", "tests", "docs", "examples", ".github"]:
        shutil.copytree(
            ROOT / name, project / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    for name in ["pyproject.toml", "uv.lock", "README.md", "LICENSE", "NOTICE", ".python-version"]:
        shutil.copy2(ROOT / name, project / name)
    if with_gitignore:
        shutil.copy2(ROOT / ".gitignore", project / ".gitignore")
    marker = f"local-publication-test-{uuid.uuid4().hex}"
    for name in LOCAL_FILES:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker + "\n")
    output = tmp_path / "dist"
    environment = dict(os.environ)
    environment.pop("OPENROUTER_API_KEY", None)
    # Build the wheel directly too: a clean source archive must not hide a
    # broader wheel selection rule when publishing directly from a checkout.
    for target in ["--sdist", "--wheel"]:
        result = subprocess.run(
            [uv, "build", "--offline", target, "--out-dir", str(output)],
            cwd=project,
            env=environment,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
    with tarfile.open(next(output.glob("*.tar.gz"))) as archive:
        members = {
            str(Path(member.name).relative_to(Path(member.name).parts[0])): archive.extractfile(
                member
            ).read()
            for member in archive.getmembers()
            if member.isfile()
        }
        assert all(name not in members for name in LOCAL_FILES)
        assert all(marker.encode() not in value for value in members.values())
        for name in [
            "src/apexbench/cli.py",
            "tests/fixtures/replay.json",
            "docs/usage.md",
            "docs/replay-format.md",
            "docs/development.md",
            "pyproject.toml",
            "uv.lock",
            "LICENSE",
            "NOTICE",
        ]:
            assert members[name] == (ROOT / name).read_bytes()
    wheel = next(output.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
        expected_modules = {
            str(path.relative_to(ROOT / "src")) for path in (ROOT / "src/apexbench").glob("*.py")
        }
        assert {name for name in members if name.startswith("apexbench/")} == expected_modules
        assert all(marker.encode() not in value for value in members.values())
        for filename in ["LICENSE", "NOTICE"]:
            entry = next(name for name in members if name.endswith("/licenses/" + filename))
            assert members[entry] == (ROOT / filename).read_bytes()
    # Import the built wheel rather than accidentally using the editable checkout.
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import apexbench.cli; "
            "assert apexbench.cli.__file__.startswith(sys.argv[1])",
            str(wheel),
        ],
        check=True,
        env=environment,
    )
