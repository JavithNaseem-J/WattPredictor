"""Poll the public Render service until it serves the exact tested commit."""

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen


def fetch(url: str) -> tuple[int, bytes]:
    with urlopen(url, timeout=10) as response:
        return response.status, response.read()


def verify(base_url: str, expected_sha: str, report_path: Path) -> None:
    report: dict[str, object] = {
        "expected_sha": expected_sha,
        "production_url": base_url,
        "status": "failed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        parsed = urlparse(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Production URL must be an HTTPS origin")
        if not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
            raise ValueError("Expected SHA must be 40 lowercase hex characters")
        origin = base_url.rstrip("/")
        deadline = time.monotonic() + 30 * 60
        while time.monotonic() < deadline:
            try:
                _, body = fetch(f"{origin}/version")
                version = json.loads(body)
                report["observed_sha"] = version.get("commit_sha")
                report["build_time"] = version.get("build_time")
                if version.get("commit_sha") != expected_sha:
                    raise ValueError("Live commit differs from tested commit")
                built_at = datetime.fromisoformat(version["build_time"])
                if built_at.utcoffset() != timezone.utc.utcoffset(built_at):
                    raise ValueError("Build time is not UTC")

                health_status, _ = fetch(f"{origin}/healthz")
                frontend_status, frontend = fetch(f"{origin}/")
                api_status, api_body = fetch(f"{origin}/api/health")
                if b"streamlit" not in frontend.lower():
                    raise ValueError("Dashboard HTML is unavailable")
                if json.loads(api_body).get("status") != "healthy":
                    raise ValueError("API health is degraded")
                report.update(
                    status="verified",
                    health_status=health_status,
                    frontend_status=frontend_status,
                    api_status=api_status,
                    verified_at=datetime.now(timezone.utc).isoformat(),
                )
                return
            except (
                HTTPError,
                URLError,
                TimeoutError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                report["last_error"] = str(exc)
                time.sleep(10)
        raise TimeoutError("Production did not serve the tested SHA within 30 minutes")
    except (TimeoutError, ValueError) as exc:
        report["error"] = str(exc)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: verify_deployment.py URL SHA REPORT_PATH")
    verify(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
