#!/usr/bin/env python3
"""Run full test suite locally and update GitHub Gist badge.

Usage:
    # All tests (requires live ChatGPT auth + running server)
    uv run python scripts/update_test_badge.py

    # Unit tests only (no auth required)
    uv run python scripts/update_test_badge.py --unit-only

Requirements:
    - GIST_TOKEN env var (fine-grained PAT with gist scope)
    - For integration tests: running gptmock server with valid auth
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from urllib.request import Request, urlopen

GIST_ID = "255a945245d92c731d002ee3be93a74c"
GIST_FILENAME = "gptmock-tests.json"

UNIT_TESTS = [
    "tests/test_env_precedence.py",
    "tests/test_bootstrap.py",
]


def run_pytest(paths: list[str]) -> tuple[int, int, int]:
    """Run pytest and return (collected, passed, failed)."""
    result = subprocess.run(
        ["uv", "run", "pytest", *paths, "-v", "--tb=short", "-q"],
        capture_output=True,
        text=True,
    )
    stdout = result.stdout

    # Parse "N passed, M failed" from last line
    passed = failed = 0
    for line in stdout.splitlines():
        if "passed" in line or "failed" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "passed" or part == "passed,":
                    passed = int(parts[i - 1])
                elif part == "failed" or part == "failed,":
                    failed = int(parts[i - 1])

    collected = 0
    for line in stdout.splitlines():
        if "collected" in line:
            for word in line.split():
                if word.isdigit():
                    collected = int(word)
                    break

    if collected == 0:
        collected = passed + failed

    print(stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return collected, passed, failed


def update_gist(data: dict) -> None:
    """Update a GitHub Gist file via API."""
    token = os.environ.get("GIST_TOKEN", "")
    if not token:
        print("GIST_TOKEN not set — skipping badge update", file=sys.stderr)
        print(f"Badge data: {json.dumps(data, indent=2)}")
        return

    payload = json.dumps({"files": {GIST_FILENAME: {"content": json.dumps(data)}}}).encode()

    req = Request(
        f"https://api.github.com/gists/{GIST_ID}",
        data=payload,
        method="PATCH",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
    )
    with urlopen(req) as resp:
        if resp.status == 200:
            print(f"Badge updated: {data['message']}")
        else:
            print(f"Gist update failed: {resp.status}", file=sys.stderr)


def main() -> None:
    unit_only = "--unit-only" in sys.argv

    if unit_only:
        paths = UNIT_TESTS
        label = "tests (unit)"
    else:
        paths = ["tests/"]
        label = "tests"

    print(f"Running: {' '.join(paths)}")
    print("=" * 60)

    collected, passed, failed = run_pytest(paths)

    if failed > 0:
        color = "red"
        message = f"{passed}/{collected} passed"
    else:
        color = "brightgreen"
        message = f"{passed} passed"

    badge = {
        "schemaVersion": 1,
        "label": label,
        "message": message,
        "color": color,
    }

    print("=" * 60)
    print(f"Result: {passed} passed, {failed} failed (of {collected})")

    update_gist(badge)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
