"""Standalone desktop UI for exact, source-grounded context compilations."""

from __future__ import annotations

import importlib
import inspect
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
import tkinter as tk
from typing import Any, Callable, Sequence

from docx import Document  # type: ignore

from context_finder import (
    COMPILATION_MARKER,
    DEFAULT_CONTEXT_WORDS_EACH_SIDE,
    ContextRegion,
    SearchOptions,
    SearchResult,
    apply_boundary_selection,
    create_compilation_docx,
    find_contexts,
    read_result_records,
    validate_query,
    write_result_records,
)
from gui_components import (
    ACCENT,
    AMBER,
    BG,
    CARD_BG,
    FG,
    FG_DIM,
    FONT,
    FONT_LG,
    FONT_SM,
    FONT_TTL,
    GREEN,
    RED,
    _styled_btn,
)


REPO_ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = REPO_ROOT / ".transcribe_settings.json"
SETTINGS_KEY = "context_finder"
_AUTO_REFINER = object()


@dataclass(frozen=True, slots=True)
class ContextFinderJobConfig:
    folder: Path
    query: str
    output_path: Path
    refine_with_glm: bool = True
    keep_jsonl: bool = False
    context_words_each_side: int = DEFAULT_CONTEXT_WORDS_EACH_SIDE


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    phase: str
    message: str
    completed: int = 0
    total: int = 0


@dataclass(frozen=True, slots=True)
class ContextFinderJobOutcome:
    result: SearchResult
    output_path: Path
    records_path: Path | None
    refinement_requested: bool
    refined_regions: int
    resumed_regions: int
    fallback_regions: int
    warnings: tuple[str, ...]

    @property
    def occurrence_count(self) -> int:
        return self.result.occurrence_count

    @property
    def region_count(self) -> int:
        return len(self.result.regions)

    @property
    def source_count(self) -> int:
        return self.result.source_count


class ContextFinderCancelled(RuntimeError):
    """Raised when a run is stopped before its atomic DOCX publication."""


def load_context_finder_settings(path: Path | str | None = None) -> dict[str, Any]:
    """Read the Context Finder section without disturbing transcription settings."""

    settings_path = Path(path) if path is not None else SETTINGS_PATH
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("top-level settings value must be an object")
        section = data.get(SETTINGS_KEY, {})
        if not isinstance(section, dict):
            raise ValueError(f"{SETTINGS_KEY!r} settings must be an object")
        return dict(section)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        warnings.warn(
            f"Could not load Context Finder settings from {settings_path}: {exc}",
            RuntimeWarning,
        )
        return {}


def save_context_finder_settings(
    values: dict[str, Any], path: Path | str | None = None
) -> bool:
    """Atomically merge Context Finder settings into the shared settings file."""

    settings_path = Path(path) if path is not None else SETTINGS_PATH
    temporary: Path | None = None
    try:
        if settings_path.exists():
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("top-level settings value must be an object")
        else:
            data = {}
        existing = data.get(SETTINGS_KEY, {})
        if not isinstance(existing, dict):
            existing = {}
        merged = dict(existing)
        merged.update(values)
        data[SETTINGS_KEY] = merged

        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{settings_path.name}.",
            suffix=".tmp",
            dir=settings_path.parent,
            delete=False,
        ) as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, settings_path)
        temporary = None
        return True
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        warnings.warn(
            f"Could not save Context Finder settings to {settings_path}: {exc}",
            RuntimeWarning,
        )
        return False
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def default_context_output_path(folder: Path | str, query: str) -> Path:
    """Return a safe compilation path beside, rather than inside, the library."""

    root = Path(folder).expanduser().resolve()
    spec = validate_query(query)
    safe_query = re.sub(r"[^\w .'-]+", "_", spec.text, flags=re.UNICODE).strip(" .")
    safe_query = safe_query[:80] or "search"
    return root.parent / f"{root.name} - Context - {safe_query}.docx"


def validate_job_config(config: ContextFinderJobConfig) -> ContextFinderJobConfig:
    """Resolve paths and protect existing non-Context-Finder documents."""

    folder = Path(config.folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Search folder does not exist: {folder}")
    query = validate_query(config.query).text
    output = Path(config.output_path).expanduser().resolve()
    if output.suffix.casefold() != ".docx":
        raise ValueError("Output must be a Word document ending in .docx")
    if config.context_words_each_side < 0:
        raise ValueError("Context word allowance cannot be negative")
    if output.exists() and not _is_generated_compilation(output):
        raise FileExistsError(
            "Refusing to overwrite an existing document that was not created by "
            "Context Finder. Choose another output filename."
        )
    return replace(config, folder=folder, query=query, output_path=output)


def run_context_finder_job(
    config: ContextFinderJobConfig,
    *,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[ProgressUpdate], None] | None = None,
    refiner: Any = _AUTO_REFINER,
) -> ContextFinderJobOutcome:
    """Run search, optional GLM boundary selection, and atomic publication.

    The optional client is deliberately invoked with a one-region SearchResult.
    A failure therefore falls back only that region. Returned paragraphs are
    never trusted: only validated boundary fields are copied onto the original
    immutable source region.
    """

    config = validate_job_config(config)
    cancelled = cancel_check or (lambda: False)
    progress = progress_callback or (lambda _update: None)
    warnings_found: list[str] = []
    records_path = config.output_path.with_suffix(".jsonl")

    _raise_if_cancelled(cancelled)
    progress(
        ProgressUpdate(
            "scan",
            "Scanning Word, text and Markdown documents recursively...",
        )
    )
    result = find_contexts(
        config.folder,
        config.query,
        options=SearchOptions(
            context_words_each_side=config.context_words_each_side
        ),
        exclude_paths=(config.output_path,),
    )
    progress(
        ProgressUpdate(
            "scan_complete",
            f"Found {result.occurrence_count} exact hit(s) in "
            f"{result.source_count} source(s), grouped into "
            f"{len(result.regions)} context region(s).",
            completed=result.scanned_files,
            total=result.scanned_files,
        )
    )
    if result.issues:
        warnings_found.append(
            f"{len(result.issues)} source document(s) could not be read; details "
            "are available in the retained JSONL when that option is enabled."
        )
    _raise_if_cancelled(cancelled)

    resumed_regions = 0
    if config.keep_jsonl and records_path.is_file():
        result, resumed_regions, resume_warning = _reuse_saved_selections(
            result, records_path
        )
        if resume_warning:
            warnings_found.append(resume_warning)
        elif resumed_regions:
            progress(
                ProgressUpdate(
                    "resume",
                    f"Reused {resumed_regions} previously validated GLM boundary "
                    f"selection(s) from {records_path.name}.",
                    completed=resumed_regions,
                    total=len(result.regions),
                )
            )

    refined_regions = 0
    fallback_regions = 0
    temporary_checkpoint_dir: Path | None = None
    pending_regions = tuple(
        region
        for region in result.regions
        if region.selection.method == "deterministic_context_window"
    )
    if config.keep_jsonl:
        # Establish the resumable baseline before the first network request.
        write_result_records(result, records_path)
    if config.refine_with_glm and pending_regions:
        if refiner is _AUTO_REFINER:
            refiner, unavailable_reason = _load_optional_refiner()
        elif refiner is None:
            unavailable_reason = "context_finder_client is unavailable"
        else:
            unavailable_reason = None

        if refiner is None:
            fallback_regions = len(pending_regions)
            warning = (
                "GLM-4.7-Flash boundary refinement was requested but is unavailable "
                f"({unavailable_reason}). All pending sections use deterministic "
                "page-sized paragraph boundaries."
            )
            warnings_found.append(warning)
            progress(ProgressUpdate("glm_fallback", warning))
        else:
            updated_by_id = {region.region_id: region for region in result.regions}
            total = len(pending_regions)
            original_by_id = {
                region.region_id: region for region in pending_regions
            }
            if config.keep_jsonl:
                client_checkpoint_dir = retained_checkpoint_dir(config.output_path)
            else:
                temporary_checkpoint_dir = operational_checkpoint_dir(
                    config.output_path
                )
                client_checkpoint_dir = temporary_checkpoint_dir
            checkpoint_lock = threading.Lock()

            def client_progress(*args, **kwargs) -> None:
                completed, reported_total, candidate, status = _parse_client_progress(
                    args, kwargs, default_total=total
                )
                if isinstance(candidate, ContextRegion):
                    original = original_by_id.get(candidate.region_id)
                    if original is not None:
                        try:
                            safe = _copy_validated_selection(original, candidate)
                        except Exception:
                            safe = None
                        if safe is not None:
                            with checkpoint_lock:
                                updated_by_id[original.region_id] = safe
                                if config.keep_jsonl:
                                    checkpoint_result = result.with_regions(
                                        tuple(updated_by_id.values())
                                    )
                                    write_result_records(
                                        checkpoint_result, records_path
                                    )
                progress(
                    ProgressUpdate(
                        "glm",
                        f"GLM-4.7-Flash context {completed} of {reported_total}: "
                        f"{_clean_error(status)}",
                        completed=completed,
                        total=reported_total,
                    )
                )

            pending_result = replace(result, regions=pending_regions)
            progress(
                ProgressUpdate(
                    "glm",
                    f"GLM-4.7-Flash is refining {total} context region(s) with "
                    "bounded parallel requests...",
                    completed=0,
                    total=total,
                )
            )
            try:
                returned = _invoke_refiner(
                    refiner,
                    pending_result,
                    cancel_check=cancelled,
                    progress_callback=client_progress,
                    checkpoint_dir=client_checkpoint_dir,
                    retain_checkpoints=config.keep_jsonl,
                )
                returned_result = _extract_refined_result(returned)
                candidate_by_id = {
                    region.region_id: region for region in returned_result.regions
                }
                for original_region in pending_regions:
                    candidate = candidate_by_id.get(original_region.region_id)
                    if candidate is None:
                        fallback_regions += 1
                        warnings_found.append(
                            f"GLM returned no result for "
                            f"{original_region.source_relative_path}; deterministic "
                            "boundaries were retained."
                        )
                        updated_by_id[original_region.region_id] = original_region
                        continue
                    try:
                        safe_region = _copy_validated_selection(
                            original_region, candidate
                        )
                    except Exception as exc:
                        fallback_regions += 1
                        warnings_found.append(
                            f"GLM result failed local validation for "
                            f"{original_region.source_relative_path}; deterministic "
                            f"boundaries were retained ({type(exc).__name__}: "
                            f"{_clean_error(str(exc))})."
                        )
                        updated_by_id[original_region.region_id] = original_region
                        continue
                    updated_by_id[original_region.region_id] = safe_region
                    if _selection_is_fallback(safe_region):
                        fallback_regions += 1
                        detail = safe_region.selection.note or "client selected fallback"
                        warnings_found.append(
                            f"Deterministic boundaries retained for "
                            f"{original_region.source_relative_path}: {_clean_error(detail)}"
                        )
                    else:
                        refined_regions += 1
            except Exception as exc:
                if cancelled():
                    raise ContextFinderCancelled(
                        "Context Finder stopped during GLM boundary refinement."
                    ) from exc
                fallback_regions = len(pending_regions)
                warning = (
                    "GLM refinement failed before a complete result could be "
                    f"validated; all pending regions retained deterministic boundaries "
                    f"({type(exc).__name__}: {_clean_error(str(exc))})."
                )
                warnings_found.append(warning)
                progress(ProgressUpdate("glm_fallback", warning))
                for original_region in pending_regions:
                    updated_by_id[original_region.region_id] = original_region
            result = result.with_regions(tuple(updated_by_id.values()))
    elif not config.refine_with_glm:
        fallback_regions = len(pending_regions)
        progress(
            ProgressUpdate(
                "deterministic",
                "GLM refinement is off; using deterministic page-sized boundaries.",
            )
        )

    _raise_if_cancelled(cancelled)
    if config.keep_jsonl:
        progress(ProgressUpdate("records", f"Writing resume records: {records_path}"))
        write_result_records(result, records_path)

    progress(ProgressUpdate("publish", "Creating the exact-quotation Word compilation..."))
    _raise_if_cancelled(cancelled)
    create_compilation_docx(result, config.output_path)
    if temporary_checkpoint_dir is not None:
        shutil.rmtree(temporary_checkpoint_dir, ignore_errors=True)
    progress(
        ProgressUpdate(
            "complete",
            f"Context compilation ready: {config.output_path}",
            completed=len(result.regions),
            total=len(result.regions),
        )
    )
    return ContextFinderJobOutcome(
        result=result,
        output_path=config.output_path,
        records_path=records_path if config.keep_jsonl else None,
        refinement_requested=config.refine_with_glm,
        refined_regions=refined_regions,
        resumed_regions=resumed_regions,
        fallback_regions=fallback_regions,
        warnings=tuple(warnings_found),
    )


def _reuse_saved_selections(
    current: SearchResult, records_path: Path
) -> tuple[SearchResult, int, str | None]:
    try:
        previous = read_result_records(records_path)
        previous_by_id = {region.region_id: region for region in previous.regions}
        updated: list[ContextRegion] = []
        reused = 0
        for region in current.regions:
            saved = previous_by_id.get(region.region_id)
            if saved is None or saved.selection.method == "deterministic_context_window":
                updated.append(region)
                continue
            updated.append(_copy_validated_selection(region, saved))
            reused += 1
        return current.with_regions(updated), reused, None
    except Exception as exc:
        return (
            current,
            0,
            f"Existing resume records were ignored because they did not validate "
            f"against the current sources ({type(exc).__name__}: "
            f"{_clean_error(str(exc))}).",
        )


def _load_optional_refiner() -> tuple[Callable[..., Any] | None, str | None]:
    try:
        module = importlib.import_module("context_finder_client")
        refiner = getattr(module, "refine_result_with_glm")
        if not callable(refiner):
            raise TypeError("refine_result_with_glm is not callable")
        return refiner, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {_clean_error(str(exc))}"


def _invoke_refiner(
    refiner: Callable[..., Any],
    singleton: SearchResult,
    *,
    cancel_check: Callable[[], bool],
    progress_callback: Callable[..., None],
    checkpoint_dir: Path,
    retain_checkpoints: bool,
) -> Any:
    """Pass optional control callbacks only when supported by the client."""

    try:
        signature = inspect.signature(refiner)
    except (TypeError, ValueError):
        return refiner(singleton)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if accepts_kwargs or "cancel_check" in signature.parameters:
        kwargs["cancel_check"] = cancel_check
    if accepts_kwargs or "progress_callback" in signature.parameters:
        kwargs["progress_callback"] = progress_callback
    if accepts_kwargs or "checkpoint_dir" in signature.parameters:
        kwargs["checkpoint_dir"] = checkpoint_dir
    if accepts_kwargs or "retain_checkpoints" in signature.parameters:
        kwargs["retain_checkpoints"] = retain_checkpoints
    return refiner(singleton, **kwargs)


def _extract_refined_result(value: Any) -> SearchResult:
    if isinstance(value, tuple) and value:
        value = value[0]
    if isinstance(value, SearchResult):
        return value
    raise TypeError("Refiner must return a SearchResult")


def _copy_validated_selection(
    original: ContextRegion, candidate: ContextRegion
) -> ContextRegion:
    if candidate.region_id != original.region_id:
        raise ValueError("Refiner returned a different region_id")
    selection = candidate.selection
    return apply_boundary_selection(
        original,
        selection.start_paragraph,
        selection.end_paragraph,
        method=selection.method,
        model=selection.model,
        confidence=selection.confidence,
        note=selection.note,
    )


def _selection_is_fallback(region: ContextRegion) -> bool:
    method = region.selection.method.casefold()
    return method == "deterministic_context_window" or "fallback" in method


def _parse_client_progress(
    args: Sequence[Any],
    kwargs: dict[str, Any],
    *,
    default_total: int,
) -> tuple[int, int, Any, str]:
    completed = int(kwargs.get("completed", args[0] if args else 0))
    total = int(kwargs.get("total", args[1] if len(args) >= 2 else default_total))
    region = kwargs.get("region", args[2] if len(args) >= 3 else None)
    status = str(kwargs.get("status", args[3] if len(args) >= 4 else "completed"))
    return completed, max(1, total), region, status


def retained_checkpoint_dir(output_path: Path | str) -> Path:
    output = Path(output_path).expanduser().resolve()
    return output.parent / f"{output.stem} - Context Finder Checkpoints"


def operational_checkpoint_dir(output_path: Path | str) -> Path:
    """Return a resumable TEMP location that is removed after publication."""

    output = Path(output_path).expanduser().resolve()
    identity = str(output).casefold() if os.name == "nt" else str(output)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return (
        Path(tempfile.gettempdir())
        / "AudioProcessor"
        / "context-finder-active"
        / digest
    )


def _raise_if_cancelled(cancel_check: Callable[[], bool]) -> None:
    if cancel_check():
        raise ContextFinderCancelled(
            "Context Finder stopped before publication; no incomplete DOCX was written."
        )


def _is_generated_compilation(path: Path) -> bool:
    try:
        properties = Document(str(path)).core_properties
        return (
            properties.subject == COMPILATION_MARKER
            or properties.keywords == COMPILATION_MARKER
        )
    except Exception:
        return False


def _clean_error(message: str, limit: int = 260) -> str:
    cleaned = " ".join(str(message).split())
    return cleaned[:limit] + ("..." if len(cleaned) > limit else "")


def open_local_output(path: Path | str) -> None:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Output document does not exist: {target}")
    if os.name == "nt":
        os.startfile(str(target))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])


class ContextFinderWindow:
    """Tk window whose worker communicates exclusively through a queue."""

    def __init__(self, window: tk.Tk | tk.Toplevel):
        self.window = window
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.close_pending = False
        self.output_is_automatic = True
        self.last_outcome: ContextFinderJobOutcome | None = None

        saved = load_context_finder_settings()
        saved_folder = str(saved.get("last_folder") or REPO_ROOT)
        if not Path(saved_folder).is_dir():
            saved_folder = str(REPO_ROOT)
        saved_query = str(saved.get("query") or "awakening")
        try:
            validate_query(saved_query)
        except ValueError:
            saved_query = "awakening"
        saved_output = str(saved.get("last_output") or "")

        self.folder_var = tk.StringVar(value=saved_folder)
        self.query_var = tk.StringVar(value=saved_query)
        self.output_var = tk.StringVar(
            value=saved_output
            or str(default_context_output_path(saved_folder, saved_query))
        )
        self.output_is_automatic = not bool(saved_output)
        self.refine_var = tk.BooleanVar(
            value=bool(saved.get("refine_with_glm", True))
        )
        self.keep_jsonl_var = tk.BooleanVar(
            value=bool(saved.get("keep_jsonl", False))
        )
        self.status_var = tk.StringVar(value="Ready")
        self.counts_var = tk.StringVar(
            value="Exact hits: 0   |   Context regions: 0   |   Sources: 0"
        )

        self._build()
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self.window.after(120, self._poll_events)

    def _build(self) -> None:
        self.window.title("Context Finder - Exact Source Research")
        self.window.geometry("960x760")
        self.window.minsize(820, 650)
        self.window.configure(bg=BG)
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)

        outer = tk.Frame(self.window, bg=BG, padx=24, pady=20)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(5, weight=1)

        tk.Label(
            outer,
            text="Context Finder",
            bg=BG,
            fg="#1a365d",
            font=FONT_TTL,
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            outer,
            text=(
                "Find an exact word or short phrase throughout a transcript library, "
                "then compile verbatim, source-linked context sections."
            ),
            bg=BG,
            fg=FG_DIM,
            font=FONT,
            anchor="w",
            justify="left",
            wraplength=880,
        ).grid(row=1, column=0, sticky="ew", pady=(4, 14))

        input_card = tk.Frame(outer, bg=CARD_BG, padx=16, pady=14)
        input_card.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        input_card.columnconfigure(1, weight=1)
        self._label(input_card, "Transcript library folder:", 0)
        folder_entry = self._entry(input_card, self.folder_var, 0)
        folder_entry.grid(row=0, column=1, sticky="ew", padx=(10, 8), pady=4)
        _styled_btn(input_card, "Browse...", self._browse_folder).grid(
            row=0, column=2, pady=4
        )

        self._label(input_card, "Exact word or phrase:", 1)
        query_entry = self._entry(input_card, self.query_var, 1)
        query_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=4)
        query_entry.bind("<KeyRelease>", lambda _event: self._refresh_default_output())

        self._label(input_card, "Output Word document:", 2)
        output_entry = self._entry(input_card, self.output_var, 2)
        output_entry.grid(row=2, column=1, sticky="ew", padx=(10, 8), pady=4)
        output_entry.bind("<Key>", lambda _event: self._mark_output_manual())
        output_buttons = tk.Frame(input_card, bg=CARD_BG)
        output_buttons.grid(row=2, column=2, sticky="e", pady=4)
        _styled_btn(output_buttons, "Save as...", self._browse_output).pack(
            side="left", padx=(0, 4)
        )
        _styled_btn(
            output_buttons,
            "Default",
            self._use_default_output,
            bg="#64748b",
        ).pack(side="left")

        options_card = tk.Frame(outer, bg=CARD_BG, padx=16, pady=12)
        options_card.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        tk.Checkbutton(
            options_card,
            text="Refine thematic boundaries with Cloudflare GLM-4.7-Flash",
            variable=self.refine_var,
            bg=CARD_BG,
            fg=FG,
            activebackground=CARD_BG,
            font=FONT,
            anchor="w",
        ).pack(anchor="w")
        tk.Checkbutton(
            options_card,
            text="Keep resumable JSONL and troubleshooting checkpoints beside the Word document",
            variable=self.keep_jsonl_var,
            bg=CARD_BG,
            fg=FG,
            activebackground=CARD_BG,
            font=FONT,
            anchor="w",
        ).pack(anchor="w", pady=(5, 0))
        tk.Label(
            options_card,
            text=(
                "GLM selects exact source boundaries only; it cannot paraphrase the "
                "quoted text. If refinement is unavailable, deterministic context is "
                "published and reported clearly."
            ),
            bg=CARD_BG,
            fg=FG_DIM,
            font=FONT_SM,
            justify="left",
            wraplength=850,
        ).pack(anchor="w", pady=(7, 0))

        controls = tk.Frame(outer, bg=BG)
        controls.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        self.start_btn = _styled_btn(
            controls,
            "Find Contexts",
            self._start,
            font=FONT_LG,
            bg=ACCENT,
        )
        self.start_btn.pack(side="left", padx=(0, 8))
        self.stop_btn = _styled_btn(
            controls, "Stop", self._stop, font=FONT_LG, bg=RED
        )
        self.stop_btn.pack(side="left", padx=(0, 8))
        self.stop_btn.configure(state="disabled", disabledforeground="#f8fafc")
        self.open_btn = _styled_btn(
            controls,
            "Open Output",
            self._open_output,
            font=FONT_LG,
            bg=GREEN,
        )
        self.open_btn.pack(side="left", padx=(0, 8))
        self.open_btn.configure(disabledforeground="#f8fafc")
        self.open_btn.configure(
            state="normal" if Path(self.output_var.get()).is_file() else "disabled"
        )

        progress_card = tk.Frame(outer, bg=CARD_BG, padx=16, pady=12)
        progress_card.grid(row=5, column=0, sticky="nsew")
        progress_card.columnconfigure(0, weight=1)
        progress_card.rowconfigure(4, weight=1)
        tk.Label(
            progress_card,
            textvariable=self.status_var,
            bg=CARD_BG,
            fg=FG,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        self.progress = ttk.Progressbar(
            progress_card, mode="determinate", maximum=1, value=0
        )
        self.progress.grid(row=1, column=0, sticky="ew", pady=(7, 8))
        tk.Label(
            progress_card,
            textvariable=self.counts_var,
            bg=CARD_BG,
            fg=FG_DIM,
            font=FONT_SM,
            anchor="w",
        ).grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Separator(progress_card, orient="horizontal").grid(
            row=3, column=0, sticky="ew", pady=(0, 8)
        )
        self.log = scrolledtext.ScrolledText(
            progress_card,
            wrap="word",
            height=12,
            font=("Consolas", 9),
            bg="#f8fafc",
            fg="#1f2937",
            relief="flat",
            padx=8,
            pady=8,
            state="disabled",
        )
        self.log.grid(row=4, column=0, sticky="nsew")

    def _label(self, parent: tk.Widget, text: str, row: int) -> None:
        tk.Label(
            parent,
            text=text,
            bg=CARD_BG,
            fg=FG,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).grid(row=row, column=0, sticky="w", pady=4)

    def _entry(self, parent: tk.Widget, variable: tk.StringVar, row: int) -> tk.Entry:
        _ = row
        return tk.Entry(
            parent,
            textvariable=variable,
            font=FONT,
            relief="flat",
            bg="#f9fafb",
            fg="#111827",
            insertbackground="#111827",
        )

    def _browse_folder(self) -> None:
        selected = filedialog.askdirectory(
            title="Select transcript library folder",
            initialdir=self.folder_var.get() or str(REPO_ROOT),
            parent=self.window,
        )
        if selected:
            self.folder_var.set(selected)
            self.output_is_automatic = True
            self._refresh_default_output()

    def _browse_output(self) -> None:
        current = Path(self.output_var.get() or str(REPO_ROOT / "contexts.docx"))
        selected = filedialog.asksaveasfilename(
            title="Save Context Finder compilation",
            initialdir=str(current.parent),
            initialfile=current.name,
            defaultextension=".docx",
            filetypes=(("Word document", "*.docx"),),
            parent=self.window,
        )
        if selected:
            self.output_var.set(selected)
            self.output_is_automatic = False

    def _mark_output_manual(self) -> None:
        self.output_is_automatic = False

    def _use_default_output(self) -> None:
        self.output_is_automatic = True
        self._refresh_default_output()

    def _refresh_default_output(self) -> None:
        if not self.output_is_automatic:
            return
        try:
            self.output_var.set(
                str(
                    default_context_output_path(
                        self.folder_var.get(), self.query_var.get()
                    )
                )
            )
        except (OSError, ValueError):
            pass

    def _start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        config = ContextFinderJobConfig(
            folder=Path(self.folder_var.get()),
            query=self.query_var.get(),
            output_path=Path(self.output_var.get()),
            refine_with_glm=self.refine_var.get(),
            keep_jsonl=self.keep_jsonl_var.get(),
        )
        try:
            config = validate_job_config(config)
        except Exception as exc:
            messagebox.showerror("Cannot start Context Finder", str(exc), parent=self.window)
            return

        save_context_finder_settings(
            {
                "last_folder": str(config.folder),
                "query": config.query,
                "last_output": str(config.output_path),
                "refine_with_glm": config.refine_with_glm,
                "keep_jsonl": config.keep_jsonl,
            }
        )
        self.cancel_event.clear()
        self.last_outcome = None
        self._clear_log()
        self._append_log(f"Search folder: {config.folder}\n")
        self._append_log(f"Exact query: {config.query}\n")
        self._append_log(f"Output: {config.output_path}\n\n")
        self.status_var.set("Starting exact recursive search...")
        self.counts_var.set(
            "Exact hits: 0   |   Context regions: 0   |   Sources: 0"
        )
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.open_btn.configure(state="disabled")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)

        def worker() -> None:
            try:
                outcome = run_context_finder_job(
                    config,
                    cancel_check=self.cancel_event.is_set,
                    progress_callback=lambda update: self.events.put(
                        ("progress", update)
                    ),
                )
                self.events.put(("complete", outcome))
            except ContextFinderCancelled as exc:
                self.events.put(("cancelled", str(exc)))
            except Exception as exc:
                self.events.put(
                    (
                        "error",
                        (
                            f"{type(exc).__name__}: {exc}",
                            traceback.format_exc(),
                        ),
                    )
                )

        self.worker = threading.Thread(
            target=worker,
            name="context-finder-worker",
            daemon=True,
        )
        self.worker.start()

    def _stop(self) -> None:
        self.cancel_event.set()
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Stop requested - waiting for the current safe boundary...")
        self._append_log(
            "\nStop requested. An active source scan, GLM request, or atomic Word "
            "publication may need to finish before the worker exits.\n"
        )

    def _poll_events(self) -> None:
        try:
            while True:
                event, value = self.events.get_nowait()
                if event == "progress":
                    self._on_progress(value)
                elif event == "complete":
                    self._on_complete(value)
                elif event == "cancelled":
                    self._finish_idle()
                    self.status_var.set("Stopped safely")
                    self._append_log(f"\n{value}\n")
                elif event == "error":
                    self._finish_idle()
                    summary, details = value
                    self.status_var.set("Context Finder failed")
                    self._append_log(f"\nError: {summary}\n{details}\n")
                    messagebox.showerror("Context Finder failed", summary, parent=self.window)
        except queue.Empty:
            pass
        if self.close_pending and not (self.worker and self.worker.is_alive()):
            self.window.destroy()
            return
        try:
            self.window.after(120, self._poll_events)
        except tk.TclError:
            pass

    def _on_progress(self, update: ProgressUpdate) -> None:
        self.status_var.set(update.message)
        self._append_log(update.message + "\n")
        if update.total > 0:
            self.progress.stop()
            self.progress.configure(
                mode="determinate",
                maximum=max(1, update.total),
                value=min(update.completed, update.total),
            )
        else:
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
        if update.phase == "scan_complete":
            match = re.search(
                r"Found (\d+) exact hit\(s\) in (\d+) source\(s\).*into (\d+)",
                update.message,
            )
            if match:
                hits, sources, regions = match.groups()
                self.counts_var.set(
                    f"Exact hits: {hits}   |   Context regions: {regions}   |   "
                    f"Sources: {sources}"
                )

    def _on_complete(self, outcome: ContextFinderJobOutcome) -> None:
        self.last_outcome = outcome
        self._finish_idle()
        self.counts_var.set(
            f"Exact hits: {outcome.occurrence_count}   |   Context regions: "
            f"{outcome.region_count}   |   Sources: {outcome.source_count}"
        )
        self.open_btn.configure(state="normal")
        if outcome.refinement_requested:
            refinement = (
                f"GLM refined {outcome.refined_regions}; resumed "
                f"{outcome.resumed_regions}; deterministic fallback "
                f"{outcome.fallback_regions}."
            )
        else:
            refinement = "GLM refinement was switched off."
        self.status_var.set("Complete - Word compilation is ready")
        self._append_log(f"\n{refinement}\n")
        for warning in outcome.warnings:
            self._append_log(f"Warning: {warning}\n")
        if outcome.records_path is not None:
            self._append_log(f"Resume JSONL: {outcome.records_path}\n")
        self._append_log(f"Open output: {outcome.output_path}\n")

    def _finish_idle(self) -> None:
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def _open_output(self) -> None:
        target = (
            self.last_outcome.output_path
            if self.last_outcome is not None
            else Path(self.output_var.get())
        )
        try:
            open_local_output(target)
        except Exception as exc:
            messagebox.showerror("Could not open output", str(exc), parent=self.window)

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _on_close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            if self.close_pending:
                return
            if not messagebox.askyesno(
                "Context Finder is still running",
                "Request a safe stop and close when the current operation finishes?",
                parent=self.window,
            ):
                return
            self.close_pending = True
            self._stop()
            return
        self.window.destroy()


def launch_context_finder() -> None:
    root = tk.Tk()
    ContextFinderWindow(root)
    root.mainloop()


def main() -> int:
    launch_context_finder()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
