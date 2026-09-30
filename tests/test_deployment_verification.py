"""The production verifier must accept only the commit that passed CI."""

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/verify_deployment.py"
spec = importlib.util.spec_from_file_location("verify_deployment", SCRIPT)
assert spec is not None and spec.loader is not None
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def test_verifies_exact_live_commit(monkeypatch, tmp_path):
    sha = "a" * 40
    responses = {
        "/version": (
            200,
            json.dumps(
                {"commit_sha": sha, "build_time": "2026-09-30T00:00:00+00:00"}
            ).encode(),
        ),
        "/healthz": (200, b"ok"),
        "/": (200, b"<title>Streamlit</title>"),
        "/api/health": (200, b'{"status":"healthy"}'),
    }
    monkeypatch.setattr(
        verifier,
        "fetch",
        lambda url: responses[url.removeprefix("https://example.com")],
    )
    report = tmp_path / "report.json"

    verifier.verify("https://example.com", sha, report)

    data = json.loads(report.read_text())
    assert data["status"] == "verified"
    assert data["observed_sha"] == sha


def test_rejects_wrong_live_commit_and_writes_report(monkeypatch, tmp_path):
    sha = "a" * 40
    monkeypatch.setattr(
        verifier,
        "fetch",
        lambda url: (
            200,
            json.dumps(
                {"commit_sha": "b" * 40, "build_time": "2026-09-30T00:00:00+00:00"}
            ).encode(),
        ),
    )
    checks = iter((0, 0, 1801))
    monkeypatch.setattr(verifier.time, "monotonic", lambda: next(checks))
    monkeypatch.setattr(verifier.time, "sleep", lambda _: None)
    report = tmp_path / "report.json"

    with pytest.raises(TimeoutError):
        verifier.verify("https://example.com", sha, report)

    data = json.loads(report.read_text())
    assert data["status"] == "failed"
    assert data["observed_sha"] == "b" * 40
