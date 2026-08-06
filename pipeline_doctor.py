"""Read-only environment preflight for the archival pipeline.

The doctor deliberately does not download models or mutate the environment.  It
checks the pinned Windows/Pascal package lane and verifies that both PyTorch and
CTranslate2 can see the installed CUDA device.  A reviewed real-audio canary is
still required before an archive-wide run.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.metadata
import json
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parent

EXPECTED_VERSIONS = {
    "torch": "2.6.0+cu124",
    "faster-whisper": "1.2.1",
    "ctranslate2": "4.8.1",
    "openai-whisper": "20250625",
    "python-docx": "1.2.0",
    "moviepy": "2.2.1",
    "psutil": "7.2.0",
    "tqdm": "4.67.1",
    "keyring": "25.6.0",
}
EXPECTED_PYTHON = (3, 12)
EXPECTED_TORCH_CUDA = "12.4"
VALID_MODES = ("full", "transcribe", "cleanup-only", "render-only", "inventory")
PARAKEET_MODEL_PREFIX = "nvidia/parakeet-"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    required: bool = False


@dataclass(frozen=True)
class ModeRequirements:
    transcribe: bool
    cleanup: bool
    render: bool


def requirements_for_mode(mode: str, *, cleanup_required: bool) -> ModeRequirements:
    if mode not in VALID_MODES:
        raise ValueError(f"unsupported doctor mode: {mode}")
    transcribe = mode in {"full", "transcribe"}
    cleanup = cleanup_required and mode in {"full", "cleanup-only"}
    render = mode in {"full", "transcribe", "cleanup-only", "render-only"}
    return ModeRequirements(transcribe=transcribe, cleanup=cleanup, render=render)


def _status_for_problem(required: bool) -> str:
    return "error" if required else "warning"


def _exception_detail(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {message}"[:500]


def skipped_check(name: str, mode: str) -> Check:
    return Check(name, "ok", f"not required for {mode} mode", False)


def package_check(
    import_name: str,
    distribution_name: str,
    label: str,
    *,
    required: bool,
    mode: str,
) -> tuple[Optional[Any], Check]:
    if not required:
        return None, skipped_check(label, mode)
    try:
        module = importlib.import_module(import_name)
        actual = importlib.metadata.version(distribution_name)
    except Exception as exc:
        return None, Check(
            label,
            "error",
            f"cannot import {distribution_name}: {_exception_detail(exc)}",
            True,
        )

    expected = EXPECTED_VERSIONS[distribution_name]
    if actual != expected:
        return module, Check(
            label,
            "error",
            f"{actual} installed; pinned version is {expected}",
            True,
        )
    return module, Check(label, "ok", f"{actual} (import succeeded)", True)


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


def find_ffmpeg() -> Optional[Path]:
    bundled = REPO_ROOT / "ffmpeg.exe"
    if bundled.is_file():
        return bundled
    discovered = shutil.which("ffmpeg")
    return Path(discovered).resolve() if discovered else None


def check_python() -> Check:
    version = sys.version_info
    bits = struct.calcsize("P") * 8
    matches = version[:2] == EXPECTED_PYTHON and bits == 64
    return Check(
        "Python",
        "ok" if matches else "error",
        (
            f"{version.major}.{version.minor}.{version.micro}, {bits}-bit; "
            "pinned lane is Python 3.12 x64"
        ),
        True,
    )


def check_torch_cuda(torch_module: Any, *, require_gpu: bool) -> Check:
    required = require_gpu
    try:
        built_cuda = str(torch_module.version.cuda or "none")
        if built_cuda != EXPECTED_TORCH_CUDA:
            return Check(
                "PyTorch CUDA",
                _status_for_problem(required),
                f"wheel runtime is CUDA {built_cuda}; pinned runtime is {EXPECTED_TORCH_CUDA}",
                required,
            )
        if not torch_module.cuda.is_available():
            return Check(
                "PyTorch CUDA",
                _status_for_problem(required),
                "torch.cuda.is_available() is false; CPU fallback only",
                required,
            )

        device_index = int(torch_module.cuda.current_device())
        name = str(torch_module.cuda.get_device_name(device_index))
        capability = tuple(torch_module.cuda.get_device_capability(device_index))
        properties = torch_module.cuda.get_device_properties(device_index)
        memory = getattr(properties, "total_memory", None)
        if memory is None:
            memory = getattr(properties, "total_mem", 0)
        memory_gib = float(memory) / (1024**3)
        architectures = set(torch_module.cuda.get_arch_list())

        if capability == (6, 1) and "sm_61" not in architectures:
            return Check(
                "PyTorch CUDA",
                _status_for_problem(required),
                "GTX 1070 Ti is compute 6.1 but this torch wheel lacks sm_61",
                required,
            )
        return Check(
            "PyTorch CUDA",
            "ok",
            (
                f"CUDA {built_cuda}; {name}; {memory_gib:.1f} GiB; "
                f"compute {capability[0]}.{capability[1]}; sm_61 present="
                f"{'sm_61' in architectures}"
            ),
            required,
        )
    except Exception as exc:
        return Check(
            "PyTorch CUDA",
            _status_for_problem(required),
            f"CUDA probe failed: {_exception_detail(exc)}",
            required,
        )


def check_windows_cuda_dlls(torch_module: Any, *, require_gpu: bool) -> Check:
    if os.name != "nt":
        return Check("CUDA runtime DLLs", "ok", "Windows-only check skipped", False)

    torch_lib = Path(torch_module.__file__).resolve().parent / "lib"
    names = (
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudnn64_9.dll",
        "cudnn_ops64_9.dll",
    )
    missing = [name for name in names if not (torch_lib / name).is_file()]
    if missing:
        return Check(
            "CUDA runtime DLLs",
            _status_for_problem(require_gpu),
            f"pinned torch runtime is missing: {', '.join(missing)}",
            require_gpu,
        )

    try:
        # Load the exact pinned files. Importing torch first also makes these
        # versioned CUDA DLLs available to CTranslate2 in this proven lane.
        for name in names:
            ctypes.WinDLL(str(torch_lib / name))
    except OSError as exc:
        return Check(
            "CUDA runtime DLLs",
            _status_for_problem(require_gpu),
            f"could not load pinned torch CUDA DLLs: {_exception_detail(exc)}",
            require_gpu,
        )
    return Check(
        "CUDA runtime DLLs",
        "ok",
        "cuBLAS 12 and cuDNN 9 loaded from pinned torch 2.6.0+cu124 runtime",
        require_gpu,
    )


def check_ctranslate2_cuda(
    ctranslate2_module: Any,
    *,
    require_gpu: bool,
) -> Check:
    try:
        device_count = int(ctranslate2_module.get_cuda_device_count())
        if device_count < 1:
            return Check(
                "CTranslate2 CUDA",
                _status_for_problem(require_gpu),
                "no CUDA device visible to CTranslate2; CPU fallback only",
                require_gpu,
            )
        compute_types = {
            str(value)
            for value in ctranslate2_module.get_supported_compute_types("cuda", 0)
        }
        int8_supported = bool({"int8", "int8_float32"} & compute_types)
        if not int8_supported:
            return Check(
                "CTranslate2 CUDA",
                _status_for_problem(require_gpu),
                f"device 0 lacks Pascal-safe INT8; supported={sorted(compute_types)}",
                require_gpu,
            )
        return Check(
            "CTranslate2 CUDA",
            "ok",
            f"{device_count} device(s); supported compute types={sorted(compute_types)}",
            require_gpu,
        )
    except Exception as exc:
        return Check(
            "CTranslate2 CUDA",
            _status_for_problem(require_gpu),
            f"CUDA probe failed: {_exception_detail(exc)}",
            require_gpu,
        )


def check_model_cache(*, required: bool, mode: str) -> Check:
    if not required:
        return skipped_check("Faster-Whisper model cache", mode)
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        repositories = sorted(
            repo.repo_id
            for repo in cache.repos
            if "faster-whisper" in repo.repo_id.casefold()
        )
    except Exception as exc:
        return Check(
            "Faster-Whisper model cache",
            "warning",
            (
                "no readable local Faster-Whisper cache; setup never downloads models "
                f"automatically ({_exception_detail(exc)})"
            ),
            False,
        )
    if not repositories:
        return Check(
            "Faster-Whisper model cache",
            "warning",
            "no cached model; run a reviewed real-audio CUDA/int8 canary before archive work",
            False,
        )
    shown = ", ".join(repositories[:3])
    suffix = "" if len(repositories) <= 3 else f" (+{len(repositories) - 3} more)"
    return Check(
        "Faster-Whisper model cache",
        "ok",
        f"cached: {shown}{suffix}; cache presence is not an inference canary",
        False,
    )


def check_parakeet_environment(*, required: bool) -> list[Check]:
    """Verify the isolated NeMo/CUDA lane without importing it into this venv."""

    interpreter = REPO_ROOT / ".parakeet-venv" / "Scripts" / "python.exe"
    if not interpreter.is_file():
        return [
            Check(
                "NVIDIA Parakeet environment",
                _status_for_problem(required),
                f"isolated interpreter is missing: {interpreter}",
                required,
            )
        ]
    probe = (
        "import json, torch, nemo; "
        "ok=torch.cuda.is_available(); "
        "device=torch.cuda.current_device() if ok else None; "
        "print(json.dumps({'torch':torch.__version__, 'nemo':nemo.__version__, "
        "'cuda':str(torch.version.cuda), 'available':ok, 'name': "
        "torch.cuda.get_device_name(device) if ok else None, 'capability': "
        "torch.cuda.get_device_capability(device) if ok else None}))"
    )
    try:
        result = subprocess.run(
            [str(interpreter), "-B", "-c", probe],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            detail = (result.stderr or result.stdout or "unknown failure").strip()
            return [
                Check(
                    "NVIDIA Parakeet environment",
                    _status_for_problem(required),
                    f"isolated import probe failed: {detail[:400]}",
                    required,
                )
            ]
        value = json.loads(result.stdout.strip())
    except Exception as exc:
        return [
            Check(
                "NVIDIA Parakeet environment",
                _status_for_problem(required),
                f"isolated import probe failed: {_exception_detail(exc)}",
                required,
            )
        ]
    cuda_ok = bool(value.get("available")) and value.get("cuda") == EXPECTED_TORCH_CUDA
    cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--nvidia--parakeet-tdt-0.6b-v3"
    return [
        Check(
            "NVIDIA Parakeet environment",
            "ok",
            f"NeMo {value.get('nemo')}; torch {value.get('torch')}; isolated interpreter {interpreter}",
            required,
        ),
        Check(
            "NVIDIA Parakeet CUDA",
            "ok" if cuda_ok else _status_for_problem(required),
            (
                f"CUDA {value.get('cuda')}; {value.get('name')}; "
                f"compute {value.get('capability')}"
                if cuda_ok
                else f"CUDA unavailable or incompatible: {value}"
            ),
            required,
        ),
        Check(
            "NVIDIA Parakeet model cache",
            "ok" if cache.is_dir() else "warning",
            str(cache) if cache.is_dir() else "model is not cached; the first run may download it",
            False,
        ),
    ]


def check_cleanup_credentials(*, required: bool) -> Check:
    token_present = False
    token_source = None
    try:
        from cleanup_client import resolve_access_credentials

        client_id, client_secret, token_source = resolve_access_credentials()
        token_present = bool(client_id and client_secret)
    except Exception:
        token_present = False
    return Check(
        "Cleanup service token",
        "ok" if token_present or not required else "error",
        (
            f"present via {token_source}"
            if token_present
            else (
                "run configure_cleanup_credentials.py or set both Access environment variables"
                if required
                else "not required for the selected mode"
            )
        ),
        required,
    )


def check_settings(*, required: bool, mode: str) -> Check:
    if not required:
        return skipped_check("Settings", mode)
    settings = REPO_ROOT / ".transcribe_settings.json"
    if not settings.is_file():
        return Check("Settings", "ok", "no local settings yet", False)
    try:
        value = json.loads(settings.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return Check(
            "Settings",
            "error",
            f"invalid local settings must be repaired: {_exception_detail(exc)}",
            True,
        )
    if not isinstance(value, dict):
        return Check(
            "Settings",
            "error",
            "local settings must contain a JSON object at the top level",
            True,
        )
    return Check("Settings", "ok", "valid local JSON object", True)


def run_checks(
    *,
    cleanup_required: bool = True,
    mode: str = "full",
    require_gpu: bool = False,
    stt_model: Optional[str] = None,
) -> list[Check]:
    needs = requirements_for_mode(mode, cleanup_required=cleanup_required)
    checks: list[Check] = [check_python()]
    uses_parakeet = bool(stt_model and stt_model.casefold().startswith(PARAKEET_MODEL_PREFIX))

    if needs.transcribe:
        ffmpeg = find_ffmpeg()
        ffmpeg_version = executable_output([str(ffmpeg), "-version"]) if ffmpeg else None
        checks.append(
            Check(
                "FFmpeg",
                "ok" if ffmpeg_version else "error",
                (
                    f"{ffmpeg_version.splitlines()[0]} [{ffmpeg}]"
                    if ffmpeg_version and ffmpeg
                    else "bundled ffmpeg.exe and ffmpeg on PATH are missing/unusable"
                ),
                True,
            )
        )
    else:
        checks.append(skipped_check("FFmpeg", mode))

    if needs.transcribe and uses_parakeet:
        checks.extend(check_parakeet_environment(required=True))
        _module, docx_check = package_check(
            "docx", "python-docx", "python-docx", required=needs.render, mode=mode
        )
        checks.append(docx_check)
        terms = REPO_ROOT / "special_words.txt"
        checks.append(
            Check(
                "GLM glossary source",
                "ok" if terms.is_file() else "warning",
                str(terms) if terms.is_file() else "special_words.txt is absent; server glossary remains authoritative",
                False,
            )
        )
        checks.append(check_settings(required=True, mode=mode))
        checks.append(check_cleanup_credentials(required=needs.cleanup))
        return checks

    torch_module, torch_package = package_check(
        "torch", "torch", "PyTorch", required=needs.transcribe, mode=mode
    )
    checks.append(torch_package)
    if torch_module is not None:
        torch_cuda = check_torch_cuda(torch_module, require_gpu=require_gpu)
        checks.append(torch_cuda)
        checks.append(
            check_windows_cuda_dlls(torch_module, require_gpu=require_gpu)
            if torch_cuda.status == "ok"
            else Check(
                "CUDA runtime DLLs",
                _status_for_problem(require_gpu),
                "not checked because the PyTorch CUDA probe did not pass",
                require_gpu,
            )
        )
    else:
        checks.extend(
            [
                skipped_check("PyTorch CUDA", mode)
                if not needs.transcribe
                else Check(
                    "PyTorch CUDA",
                    _status_for_problem(require_gpu),
                    "not checked because PyTorch could not be imported",
                    require_gpu,
                ),
                skipped_check("CUDA runtime DLLs", mode)
                if not needs.transcribe
                else Check(
                    "CUDA runtime DLLs",
                    _status_for_problem(require_gpu),
                    "not checked because PyTorch could not be imported",
                    require_gpu,
                ),
            ]
        )

    ctranslate2_module, ctranslate2_package = package_check(
        "ctranslate2",
        "ctranslate2",
        "CTranslate2",
        required=needs.transcribe,
        mode=mode,
    )
    checks.append(ctranslate2_package)
    if ctranslate2_module is not None:
        checks.append(
            check_ctranslate2_cuda(ctranslate2_module, require_gpu=require_gpu)
        )
    else:
        checks.append(
            skipped_check("CTranslate2 CUDA", mode)
            if not needs.transcribe
            else Check(
                "CTranslate2 CUDA",
                _status_for_problem(require_gpu),
                "not checked because CTranslate2 could not be imported",
                require_gpu,
            )
        )

    package_specs = (
        ("faster_whisper", "faster-whisper", "Faster-Whisper", needs.transcribe),
        ("whisper", "openai-whisper", "OpenAI Whisper fallback", needs.transcribe),
        ("docx", "python-docx", "python-docx", needs.render),
        ("moviepy", "moviepy", "MoviePy", needs.transcribe),
        ("psutil", "psutil", "psutil", needs.transcribe),
        ("tqdm", "tqdm", "tqdm", needs.transcribe),
        ("keyring", "keyring", "keyring", needs.cleanup),
    )
    for import_name, distribution, label, required in package_specs:
        _module, check = package_check(
            import_name,
            distribution,
            label,
            required=required,
            mode=mode,
        )
        checks.append(check)

    if needs.transcribe:
        gpu = executable_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        )
        checks.append(
            Check(
                "NVIDIA driver",
                "ok" if gpu else _status_for_problem(require_gpu),
                gpu or "nvidia-smi unavailable; CPU fallback only",
                require_gpu,
            )
        )
    else:
        checks.append(skipped_check("NVIDIA driver", mode))

    checks.append(check_settings(required=needs.transcribe, mode=mode))

    terms = REPO_ROOT / "special_words.txt"
    if needs.transcribe:
        checks.append(
            Check(
                "Whisper prompt terms",
                "ok" if terms.is_file() else "warning",
                str(terms) if terms.is_file() else "special_words.txt is absent",
                False,
            )
        )
    else:
        checks.append(skipped_check("Whisper prompt terms", mode))

    checks.append(check_model_cache(required=needs.transcribe, mode=mode))
    checks.append(check_cleanup_credentials(required=needs.cleanup))
    return checks


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check archival pipeline prerequisites")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="full",
        help="Only require dependencies used by this pipeline mode",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Treat missing or incompatible NVIDIA CUDA acceleration as an error",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not require Cloudflare cleanup credentials",
    )
    parser.add_argument(
        "--stt-model",
        help="Selected local STT model; enables the isolated Parakeet checks when applicable",
    )
    args = parser.parse_args(argv)
    checks = run_checks(
        cleanup_required=not args.no_cleanup,
        mode=args.mode,
        require_gpu=args.require_gpu,
        stt_model=args.stt_model,
    )
    if args.json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        symbols = {"ok": "OK", "warning": "WARN", "error": "ERROR"}
        for check in checks:
            print(f"[{symbols[check.status]:5}] {check.name}: {check.detail}")
    return 1 if any(check.status == "error" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
