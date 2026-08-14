"""Canonical, dependency-free configuration resolution for AudioProcessor.

The GUI retains its historic settings-file keys for backwards compatibility,
but all new callers should use the canonical names documented here. Resolution
is deterministic: defaults < environment < saved settings < CLI < GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping


DEFAULT_STT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_CLEANUP_ENDPOINT = "https://pg.objectiveartefacts.com.au/api/tooling/cleanup-chunk"
DEFAULT_CLEANUP_MODEL = "@cf/zai-org/glm-4.7-flash"
DEFAULT_GLM_WORKERS = 30
EXISTING_DOCX_MODES = frozenset({"skip", "all", "before"})


class PipelineConfigValidationError(ValueError):
    """Raised when one configuration surface supplies an unsafe value."""


@dataclass(frozen=True)
class ResolvedPipelineSettings:
    stt_model: str
    existing_docx_mode: str
    replace_before_date: str | None
    cleanup_endpoint: str
    cleanup_model: str
    glm_workers: int
    sources: Mapping[str, str]

    def startup_log(self) -> str:
        fields = (
            "stt_model",
            "existing_docx_mode",
            "replace_before_date",
            "cleanup_endpoint",
            "cleanup_model",
            "glm_workers",
        )
        values = {field: getattr(self, field) for field in fields}
        detail = ", ".join(f"{field}={values[field]!r} ({self.sources[field]})" for field in fields)
        return f"Effective pipeline configuration: {detail}"


def resolve_pipeline_settings(
    *,
    environment: Mapping[str, str] | None = None,
    saved_settings: Mapping[str, Any] | None = None,
    cli_values: Mapping[str, Any] | None = None,
    gui_values: Mapping[str, Any] | None = None,
) -> ResolvedPipelineSettings:
    """Resolve canonical settings and record the winning source for each value."""

    values: dict[str, Any] = {
        "stt_model": DEFAULT_STT_MODEL,
        "existing_docx_mode": "all",
        "replace_before_date": None,
        "cleanup_endpoint": DEFAULT_CLEANUP_ENDPOINT,
        "cleanup_model": DEFAULT_CLEANUP_MODEL,
        "glm_workers": DEFAULT_GLM_WORKERS,
    }
    sources = {key: "default" for key in values}
    env = os.environ if environment is None else environment
    _apply(values, sources, _environment_values(env), "environment")
    _apply(values, sources, _canonicalize(saved_settings or {}), "settings")
    _apply(values, sources, _canonicalize(cli_values or {}), "cli")
    _apply(values, sources, _canonicalize(gui_values or {}), "gui")

    for key in ("stt_model", "cleanup_endpoint", "cleanup_model"):
        value = values[key]
        if not isinstance(value, str) or not value.strip():
            raise PipelineConfigValidationError(f"{key} must be a non-empty string")
        values[key] = value.strip()
    mode = values["existing_docx_mode"]
    if mode not in EXISTING_DOCX_MODES:
        raise PipelineConfigValidationError(
            f"existing_docx_mode must be one of {sorted(EXISTING_DOCX_MODES)}"
        )
    cutoff = values["replace_before_date"]
    if cutoff is not None and (not isinstance(cutoff, str) or not cutoff.strip()):
        cutoff = None
    values["replace_before_date"] = cutoff.strip() if isinstance(cutoff, str) else None
    try:
        values["glm_workers"] = int(values["glm_workers"])
    except (TypeError, ValueError) as exc:
        raise PipelineConfigValidationError("glm_workers must be a positive integer") from exc
    if values["glm_workers"] < 1:
        raise PipelineConfigValidationError("glm_workers must be a positive integer")
    return ResolvedPipelineSettings(**values, sources=dict(sources))


def _environment_values(environment: Mapping[str, str]) -> dict[str, Any]:
    return {
        "stt_model": environment.get("TRANSCRIBE_MODEL_NAME"),
        "cleanup_endpoint": environment.get("PG_CLEANUP_ENDPOINT"),
        "cleanup_model": environment.get("PG_CLEANUP_MODEL"),
        "glm_workers": environment.get("TRANSCRIPT_GLM_WORKERS"),
        "existing_docx_mode": environment.get("TRANSCRIPT_EXISTING_DOCX_MODE"),
        "replace_before_date": environment.get("TRANSCRIPT_REPLACE_BEFORE_DATE"),
    }


def _canonicalize(values: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    aliases = {
        "whisper_model": "stt_model",
        "replace_mode": "existing_docx_mode",
    }
    for key, value in values.items():
        canonical = aliases.get(key, key)
        if canonical not in {
            "stt_model", "existing_docx_mode", "replace_before_date",
            "cleanup_endpoint", "cleanup_model", "glm_workers",
        } or value is None:
            continue
        if canonical in result and result[canonical] != value:
            raise PipelineConfigValidationError(
                f"conflicting values for {canonical}: use only the canonical key"
            )
        result[canonical] = value
    return result


def _apply(values: dict[str, Any], sources: dict[str, str], candidates: Mapping[str, Any], source: str) -> None:
    for key, value in candidates.items():
        if value is not None:
            values[key] = value
            sources[key] = source
