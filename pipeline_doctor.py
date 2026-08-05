"""Read-only environment preflight for the archival pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class Check:
    name: str
    status: str
    detail: str
    required: bool = False


def module_check(import_name: str, label: str, *, required: bool = True) -> Check:
    available = importlib.util.find_spec(import_name) is not None
    return Check(
        label,
        "ok" if available else ("error" if required else "warning"),
        "installed" if available else "not installed",
        required,
    )


def executable_output(command: list[str], timeout: int = 8) -> Optional[str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0:
            return (result.stdout or result.stderr).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def run_checks(*, cleanup_required: bool = True) -> list[Check]:
    checks: list[Check] = []
    version = sys.version_info
    tested = (3, 11) <= version[:2] <= (3, 12)
    supported = version >= (3, 10)
    checks.append(
        Check(
            "Python",
            "ok" if tested else ("warning" if supported else "error"),
            f"{version.major}.{version.minor}.{version.micro}; tested target is 3.11/3.12",
            True,
        )
    )

    ffmpeg = REPO_ROOT / "ffmpeg.exe"
    ffmpeg_version = (
        executable_output([str(ffmpeg), "-version"]) if ffmpeg.is_file() else None
    )
    checks.append(
        Check(
            "FFmpeg",
            "ok" if ffmpeg_version else "error",
            (ffmpeg_version or "bundled ffmpeg.exe is missing/unusable").splitlines()[0],
            True,
        )
    )

    for import_name, label in (
        ("torch", "PyTorch"),
        ("faster_whisper", "Faster-Whisper"),
        ("whisper", "OpenAI Whisper fallback"),
        ("docx", "python-docx"),
        ("psutil", "psutil"),
    ):
        checks.append(module_check(import_name, label))

    gpu = executable_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    checks.append(
        Check(
            "NVIDIA GPU",
            "ok" if gpu else "warning",
            gpu or "nvidia-smi unavailable; CPU fallback may be used",
            False,
        )
    )

    settings = REPO_ROOT / ".transcribe_settings.json"
    if settings.is_file():
        try:
            json.loads(settings.read_text(encoding="utf-8"))
            checks.append(Check("Settings", "ok", "valid local JSON"))
        except (OSError, ValueError) as exc:
            checks.append(
                Check(
                    "Settings",
                    "warning",
                    f"invalid local settings ignored: {exc}",
                )
            )
    else:
        checks.append(Check("Settings", "ok", "no local settings yet"))

    terms = REPO_ROOT / "special_words.txt"
    checks.append(
        Check(
            "Whisper prompt terms",
            "ok" if terms.is_file() else "warning",
            str(terms) if terms.is_file() else "special_words.txt is absent",
        )
    )

    token_present = False
    token_source = None
    try:
        from cleanup_client import resolve_access_credentials

        _client_id, _client_secret, token_source = resolve_access_credentials()
        token_present = bool(_client_id and _client_secret)
    except Exception:
        token_present = False
    checks.append(
        Check(
            "Cleanup service token",
            (
                "ok"
                if token_present
                else ("error" if cleanup_required else "ok")
            ),
            (
                f"present via {token_source}"
                if token_present
                else (
                    "run configure_cleanup_credentials.py or set both Access environment variables"
                    if cleanup_required
                    else "not required for the selected local-only/render-only run"
                )
            ),
            cleanup_required,
        )
    )
    return checks


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check archival pipeline prerequisites")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not require Cloudflare cleanup credentials",
    )
    args = parser.parse_args(argv)
    checks = run_checks(cleanup_required=not args.no_cleanup)
    if args.json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        symbols = {"ok": "OK", "warning": "WARN", "error": "ERROR"}
        for check in checks:
            print(f"[{symbols[check.status]:5}] {check.name}: {check.detail}")
    return 1 if any(check.status == "error" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
