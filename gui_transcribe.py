"""AudioProcessor v2.0 — Speech-to-Text Transcription Tool

Clean GUI for converting audio/video to text with local speech models.
  - NVIDIA Parakeet TDT 0.6B v3 (GPU) -- archive default
  - Faster-Whisper large-v3 (GPU, CTranslate2 int8) -- retained comparison option
  - Single file or recursive batch processing
  - Skip / replace / replace-before-date for existing outputs
"""
import argparse
import datetime
import gc
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import warnings
from pathlib import Path
from typing import Callable, List, Optional

import tkinter as tk
from tkinter import ttk, messagebox

from gui_components import (
    BG, CARD_BG, FG, ACCENT, GREEN, RED, AMBER,
    FONT, FONT_LG, FONT_TTL, SUPPORTED_EXTS,
    InputPanel, SettingsPanel, LogPanel, ModelPreloadDialog, _styled_btn,
)

# ── Paths & constants ────────────────────────────────────────────────
REPO_ROOT     = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(REPO_ROOT, ".transcribe_settings.json")
STOP_FLAG     = threading.Event()


# ── Settings persistence ─────────────────────────────────────────────
def _load_settings() -> dict:
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            warnings.warn(
                f"Ignoring settings in {SETTINGS_PATH}: the top-level value must be an object.",
                RuntimeWarning,
            )
            return {}
        return data
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(
            f"Could not load settings from {SETTINGS_PATH}: {exc}",
            RuntimeWarning,
        )
        return {}


def _save_settings(data: dict) -> bool:
    """Atomically persist settings, retaining the previous file on failure."""
    tmp_path = f"{SETTINGS_PATH}.tmp"
    try:
        os.makedirs(os.path.dirname(SETTINGS_PATH) or ".", exist_ok=True)
        if os.path.isfile(SETTINGS_PATH):
            try:
                with open(SETTINGS_PATH, "r", encoding="utf-8") as existing:
                    existing_data = json.load(existing)
                if not isinstance(existing_data, dict):
                    raise ValueError("top-level value is not an object")
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                backup_path = f"{SETTINGS_PATH}.invalid-backup"
                if not os.path.exists(backup_path):
                    shutil.copy2(SETTINGS_PATH, backup_path)
                warnings.warn(
                    f"Backed up invalid settings to {backup_path} before replacing them: {exc}",
                    RuntimeWarning,
                )
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data or {}, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, SETTINGS_PATH)
        return True
    except (OSError, TypeError, ValueError) as exc:
        warnings.warn(
            f"Could not save settings to {SETTINGS_PATH}: {exc}",
            RuntimeWarning,
        )
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False


def _load_project(folder: str) -> dict:
    settings = _load_settings()
    projects = settings.get("projects", {})
    if not isinstance(projects, dict):
        return {}
    key = _project_key(folder)
    direct = projects.get(key)
    if isinstance(direct, dict):
        return dict(direct)
    # Read older settings whose keys used a different slash or case style.
    for saved_path, value in projects.items():
        if (
            isinstance(saved_path, str)
            and isinstance(value, dict)
            and _project_key(saved_path) == key
        ):
            return dict(value)
    return {}


def _save_project(folder: str, proj: dict) -> None:
    all_s = _load_settings()
    projects = all_s.get("projects")
    if not isinstance(projects, dict):
        projects = {}
        all_s["projects"] = projects
    key = _project_key(folder)
    merged: dict = {}
    for saved_path in list(projects):
        value = projects.get(saved_path)
        if isinstance(saved_path, str) and _project_key(saved_path) == key:
            if isinstance(value, dict):
                merged.update(value)
            if saved_path != key:
                del projects[saved_path]
    merged.update(proj)
    projects[key] = merged
    _save_settings(all_s)


def _project_folder(path: str) -> str:
    absolute = os.path.abspath(os.path.normpath(os.path.expanduser(path)))
    return os.path.dirname(absolute) if os.path.isfile(absolute) else absolute


def _project_key(path: str) -> str:
    """Return one stable settings key for equivalent Windows path spellings."""

    return os.path.normcase(_project_folder(path))


def _validate_replace_policy(mode: str, cutoff_date: str) -> Optional[str]:
    """Validate replacement selection and return its normalized cutoff."""

    if mode not in {"skip", "all", "before"}:
        raise ValueError("Existing-output selection is invalid.")
    if mode != "before":
        return None
    value = str(cutoff_date or "").strip()
    try:
        parsed = datetime.datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            "Refresh-transcripts-before requires a valid date in YYYY-MM-DD format."
        ) from exc
    if parsed.strftime("%Y-%m-%d") != value:
        raise ValueError(
            "Refresh-transcripts-before requires a valid date in YYYY-MM-DD format."
        )
    return value


def _validate_polished_selection(settings: dict) -> Optional[str]:
    """Validate the GUI's polished-pipeline selection before saving or running."""

    mode = str(settings.get("replace_mode", "skip"))
    cutoff = _validate_replace_policy(
        mode, str(settings.get("replace_before_date", ""))
    )
    if bool(settings.get("existing_transcripts_only", False)) and mode == "skip":
        raise ValueError(
            "Use existing Word transcripts (skip Whisper) cannot be combined with "
            "Skip existing. Choose Refresh all, or Refresh transcripts before a date."
        )
    return cutoff


def _default_terms_file() -> Optional[str]:
    p = os.path.join(REPO_ROOT, "special_words.txt")
    return p if os.path.isfile(p) else None


def _launch_context_finder_process():
    """Open the standalone research tool without coupling its worker to this GUI."""

    script = os.path.join(REPO_ROOT, "context_finder_gui.py")
    if not os.path.isfile(script):
        raise FileNotFoundError(f"Context Finder module is missing: {script}")
    return subprocess.Popen([sys.executable, script], cwd=REPO_ROOT)


# ── Transcription helpers ────────────────────────────────────────────
def _run_single(path: str, outdir: Optional[str], q: queue.Queue,
                *, threads: Optional[int] = None) -> bool:
    """Transcribe one file, writing progress to *q* and reporting success."""
    if STOP_FLAG.is_set():
        q.put("Cancelled.\n")
        return False
    try:
        from transcribe_optimised import transcribe_file_simple_auto
        target = outdir or os.path.dirname(path)
        out = transcribe_file_simple_auto(path, output_dir=target,
                                          threads_override=threads)
        if out and os.path.isfile(out):
            q.put(f"Done -> {out}\n")
            return True
        else:
            q.put("Warning: no output generated.\n")
            return False
    except Exception as e:
        # Check if this was a stop request (don't print traceback for user-initiated stops)
        if STOP_FLAG.is_set() or "Stop requested" in str(e):
            q.put("Stopped.\n")
            return False
        import traceback
        q.put(f"Error: {e}\n{traceback.format_exc()}")
        return False


def _run_batch(paths: List[str], q: queue.Queue,
               *, threads: Optional[int] = None):
    total = len(paths)
    q.put(f"Batch: {total} file(s) queued.\n")
    ok = fail = 0
    for i, p in enumerate(paths, 1):
        if STOP_FLAG.is_set():
            q.put(f"\nCancelled after {ok} done, {fail} failed.\n")
            return
        q.put(f"\n[{i}/{total}] {os.path.basename(p)}\n")
        try:
            succeeded = _run_single(p, None, q, threads=threads)
            if succeeded:
                ok += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1
            q.put(f"Error ({os.path.basename(p)}): {e}\n")
    q.put(f"\nBatch complete -- {ok} succeeded, {fail} failed.\n")


def _run_polished_pipeline(input_path: str, settings: dict, q: queue.Queue) -> int:
    """Run the resumable local STT -> GLM -> Word path used by the GUI."""

    from archive_pipeline import PipelineConfig, default_output_root, execute_pipeline

    source = Path(input_path).expanduser().resolve()
    if source.is_file():
        # Source publication treats the file's parent as its protected scope,
        # so every file canary receives a dedicated output outside that scope.
        extension = source.suffix.lower().lstrip(".") or "audio"
        output_root = (
            source.parent.parent
            / f"{source.parent.name} - Polished Single Files"
            / f"{source.stem}__{extension}"
        )
    else:
        output_root = default_output_root(source)
    existing_transcripts_only = bool(
        settings.get("existing_transcripts_only", False)
    )
    retain_troubleshooting_artifacts = bool(
        settings.get("retain_troubleshooting_artifacts", True)
    )
    selected_stt_model = str(
        settings.get("whisper_model", "nvidia/parakeet-tdt-0.6b-v3")
    )
    cutoff = _validate_polished_selection(settings)
    config = PipelineConfig(
        input_path=source,
        output_root=output_root,
        stt_model=selected_stt_model,
        force=bool(settings.get("force_reprocess", False)),
        publish_source_docx=True,
        recursive=bool(settings.get("recursive", True)),
        existing_docx_mode=str(settings.get("replace_mode", "skip")),
        replace_before_date=cutoff,
        existing_transcripts_only=existing_transcripts_only,
        retain_troubleshooting_artifacts=retain_troubleshooting_artifacts,
        glm_workers=10,
        progress_callback=lambda lane, message: q.put(("progress", lane, message)),
    )
    q.put(f"Polished artifacts: {output_root}\n")
    if not retain_troubleshooting_artifacts:
        q.put(
            "Optional per-job event logging is off; compact hash-bound resume and "
            "provenance metadata will still be retained.\n"
        )
    if existing_transcripts_only:
        q.put(
            "Pipeline: existing speech Word -> protected GLM-4.7-Flash -> "
            "separate '<name> - GLM Review.docx' (Whisper and audio skipped; "
            "source Word remains unchanged)\n"
        )
    else:
        if selected_stt_model.casefold().startswith("nvidia/parakeet-"):
            q.put(
                "Pipeline: one local Parakeet GPU worker -> durable raw '<name>.docx'; "
                "ten protected GLM-4.7-Flash workers review the completed queue independently.\n"
            )
        else:
            q.put(
                "Pipeline: local Faster-Whisper -> raw '<name>.docx' plus protected "
                "GLM-4.7-Flash '<name> - GLM Review.docx'\n"
            )
    return int(execute_pipeline(config, cancel_check=STOP_FLAG.is_set))


def _run_polished_preflight(
    q: queue.Queue,
    *,
    existing_transcripts_only: bool = False,
    stt_model: str = "nvidia/parakeet-tdt-0.6b-v3",
) -> bool:
    """Fail before opening audio when the pinned production lane is incomplete."""

    from pipeline_doctor import run_checks

    if existing_transcripts_only:
        q.put("Checking document and protected-cleanup prerequisites (no GPU needed)...\n")
    else:
        q.put("Checking local GPU, document, and protected-cleanup prerequisites...\n")
    checks = run_checks(
        cleanup_required=True,
        mode="cleanup-only" if existing_transcripts_only else "full",
        require_gpu=not existing_transcripts_only,
        stt_model=None if existing_transcripts_only else stt_model,
    )
    symbols = {"ok": "OK", "warning": "WARN", "error": "ERROR"}
    for check in checks:
        q.put(f"[{symbols.get(check.status, check.status):5}] {check.name}: {check.detail}\n")
    errors = [check for check in checks if check.status == "error"]
    if errors:
        q.put("Preflight failed; no audio or existing transcript was changed.\n")
        return False
    q.put("Preflight passed.\n\n")
    return True


def _handle_window_close(
    worker_state: dict,
    *,
    confirm_close: Callable[[], bool],
    request_stop: Callable[[], None],
    destroy: Callable[[], None],
) -> str:
    """Close immediately when idle, otherwise keep Tk alive for safe shutdown."""

    if not worker_state.get("active"):
        destroy()
        return "closed"
    if worker_state.get("close_pending"):
        return "waiting"
    if not confirm_close():
        return "running"
    # A Tk message box runs its own event loop.  The worker may have completed
    # while the confirmation was open, in which case it is now safe to close.
    if not worker_state.get("active"):
        destroy()
        return "closed"
    worker_state["close_pending"] = True
    request_stop()
    return "stopping"


# ── Output-skip logic ────────────────────────────────────────────────
def _should_process(src: str, mode: str, cutoff_date: str) -> bool:
    """Return True if this file should be (re-)transcribed."""
    normalized_cutoff = _validate_replace_policy(mode, cutoff_date)
    docx = os.path.splitext(src)[0] + ".docx"
    if not os.path.isfile(docx):
        return True  # no existing output
    if mode == "all":
        return True
    if mode == "before":
        cutoff = datetime.datetime.strptime(str(normalized_cutoff), "%Y-%m-%d")
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(docx))
        return mtime < cutoff
    # mode == "skip"
    return False


def _collect_files(folder: str, recursive: bool, replace_mode: str,
                   cutoff_date: str, q: queue.Queue) -> List[str]:
    files: List[str] = []
    skipped = 0
    walker = os.walk(folder) if recursive else [(folder, [], os.listdir(folder))]
    for dp, _, names in walker:
        for n in sorted(names):
            full = os.path.join(dp, n)
            if not os.path.isfile(full):
                continue
            if os.path.splitext(n)[1].lower() not in SUPPORTED_EXTS:
                continue
            if _should_process(full, replace_mode, cutoff_date):
                files.append(full)
            else:
                skipped += 1
    if skipped:
        q.put(f"Skipped {skipped} file(s) with existing outputs.\n")
    return files


# ── Queue-based stdout redirect ──────────────────────────────────────
class _QueueWriter:
    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, s):
        if s:
            self.q.put(str(s))

    def flush(self):
        pass


# ═══════════════════════════════════════════════════════════════════
#  Main GUI
# ═══════════════════════════════════════════════════════════════════
def launch_gui():
    root = tk.Tk()
    root.title("AudioProcessor v2.0 — Speech-to-Text")
    root.geometry("1060x760")
    root.minsize(900, 640)
    root.configure(bg=BG)
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    try:
        root.state("zoomed")
    except Exception:
        pass

    style = ttk.Style()
    style.configure("Clean.TFrame", background=BG)

    outer = ttk.Frame(root, style="Clean.TFrame", padding="24 20 24 20")
    outer.grid(row=0, column=0, sticky="nsew")
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(4, weight=1)  # log panel expands

    # ── Title ────────────────────────────────────────────────────────
    tk.Label(outer, text="Speech-to-Text Transcription", bg=BG, fg="#1a365d",
             font=FONT_TTL).grid(row=0, column=0, sticky="w", pady=(0, 16))

    # ── Load last-used project settings ──────────────────────────────
    last_settings = _load_settings()
    last_input = str(
        last_settings.get("last_input")
        or last_settings.get("last_folder")
        or REPO_ROOT
    )
    if not os.path.exists(last_input):
        last_input = REPO_ROOT
    proj = _load_project(last_input)

    def on_source_selected(source):
        nonlocal proj
        proj = _load_project(source)
        settings_panel.apply(proj)

    # ── Input panel ──────────────────────────────────────────────────
    input_panel = InputPanel(
        outer,
        on_source_selected=on_source_selected,
        initial_path=last_input,
    )
    input_panel.grid(row=1, column=0, sticky="ew", pady=(0, 10))

    # ── Settings panel ───────────────────────────────────────────────
    settings_panel = SettingsPanel(outer, proj_settings=proj)
    settings_panel.grid(row=2, column=0, sticky="ew", pady=(0, 10))

    # ── Buttons ──────────────────────────────────────────────────────
    btn_bar = tk.Frame(outer, bg=BG)
    btn_bar.grid(row=3, column=0, sticky="w", pady=(0, 10))

    q: queue.Queue = queue.Queue()
    worker_state = {
        "active": False,
        "close_pending": False,
        "existing_transcripts_only": False,
    }

    def finish_worker():
        worker_state["active"] = False
        worker_state["existing_transcripts_only"] = False
        run_btn.configure(state="normal")
        stop_btn.configure(state="disabled")
        clear_btn.configure(state="normal")
        download_btn.configure(state="normal")
        access_btn.configure(state="normal")
        if worker_state["close_pending"]:
            root.destroy()

    def start():
        inp = input_panel.get_path()
        if not inp or not os.path.exists(inp):
            messagebox.showerror("No input",
                                 "Select a valid file or folder first.")
            return
        snap = settings_panel.snapshot()
        try:
            normalized_cutoff = (
                _validate_polished_selection(snap)
                if snap.get("polished_pipeline", 1)
                else _validate_replace_policy(
                    str(snap.get("replace_mode", "skip")),
                    str(snap.get("replace_before_date", "")),
                )
            )
        except ValueError as exc:
            messagebox.showerror("Cannot start pipeline", str(exc))
            return
        if normalized_cutoff is not None:
            snap["replace_before_date"] = normalized_cutoff
        existing_only_run = bool(
            snap.get("polished_pipeline", 1)
            and snap.get("existing_transcripts_only", False)
        )

        stt_log.clear()
        glm_log.clear()
        stt_log.append("Starting speech-to-text lane...\n")
        glm_log.append("Starting protected GLM review queue...\n")
        run_btn.configure(state="disabled")
        stop_btn.configure(state="normal")
        clear_btn.configure(state="disabled")
        download_btn.configure(state="disabled")
        access_btn.configure(state="disabled")
        worker_state["active"] = True
        worker_state["existing_transcripts_only"] = existing_only_run
        STOP_FLAG.clear()
        # Also reset the engine's internal stop event
        if not existing_only_run:
            try:
                from transcribe_optimised import clear_stop
                clear_stop()
            except Exception:
                pass

        # Persist project settings
        folder = _project_folder(inp)
        _save_project(folder, snap)
        s = _load_settings()
        s["last_folder"] = folder
        s["last_input"] = os.path.abspath(os.path.normpath(inp))
        _save_settings(s)

        def worker():
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = _QueueWriter(q)
            sys.stderr = _QueueWriter(q)
            try:
                # Existing-transcript mode deliberately avoids even loading the
                # transcription engine; only archive_pipeline's DOCX import,
                # protected cleanup, and rendering paths are entered.
                if not existing_only_run:
                    os.environ["TRANSCRIBE_MODEL_NAME"] = snap["whisper_model"]
                    if snap["whisper_model"].casefold().startswith("nvidia/parakeet-"):
                        os.environ.pop("TRANSCRIBE_FORCE_NATIVE_WHISPER", None)
                    elif snap["whisper_model"].startswith("faster-whisper-"):
                        os.environ.pop("TRANSCRIBE_FORCE_NATIVE_WHISPER", None)
                    else:
                        os.environ["TRANSCRIBE_FORCE_NATIVE_WHISPER"] = "1"

                    os.environ["TRANSCRIBE_QUALITY_MODE"] = (
                        "1" if snap["quality_mode"] else "0"
                    )
                    os.environ["TRANSCRIBE_MAX_PERF"] = "1"
                    os.environ["TRANSCRIBE_ALLOW_PROMPT"] = "1"
                    terms = _default_terms_file()
                    if terms:
                        os.environ["TRANSCRIBE_AWKWARD_FILE"] = terms

                if snap.get("polished_pipeline", 1):
                    existing_transcripts_only = bool(
                        snap.get("existing_transcripts_only", False)
                    )
                    if not _run_polished_preflight(
                        q,
                        existing_transcripts_only=existing_transcripts_only,
                        stt_model=str(snap.get("whisper_model", "nvidia/parakeet-tdt-0.6b-v3")),
                    ):
                        return
                    exit_code = _run_polished_pipeline(inp, snap, q)
                    if STOP_FLAG.is_set():
                        q.put("\nStopped safely. Completed checkpoints were preserved.\n")
                    elif exit_code == 0:
                        q.put(
                            "\nPipeline completed. Raw speech documents and "
                            "separate GLM Review copies are ready.\n"
                        )
                    elif exit_code == 3:
                        q.put(
                            "\nPipeline completed. One or more GLM Review documents are "
                            "flagged for human checking; original speech documents "
                            "remain untouched.\n"
                        )
                    else:
                        q.put(
                            "\nPipeline stopped or failed. Run it again with the same "
                            "folder to resume preserved checkpoints.\n"
                        )
                elif os.path.isdir(inp):
                    files = _collect_files(
                        inp,
                        recursive=bool(snap["recursive"]),
                        replace_mode=snap["replace_mode"],
                        cutoff_date=snap["replace_before_date"],
                        q=q,
                    )
                    if files:
                        _run_batch(files, q)
                    else:
                        q.put("No eligible files found.\n")
                else:
                    if _should_process(inp, snap["replace_mode"],
                                       snap["replace_before_date"]):
                        _run_single(inp, None, q)
                    else:
                        q.put("Output already exists (skipped).\n")
                if not snap.get("polished_pipeline", 1):
                    q.put("\nDone.\n")
            except Exception as e:
                # Clean message for user-initiated stops (no traceback)
                if STOP_FLAG.is_set() or "Stop requested" in str(e):
                    q.put("\nStopped by user.\n")
                else:
                    import traceback
                    q.put(f"Error: {e}\n{traceback.format_exc()}")
            finally:
                sys.stdout, sys.stderr = old_out, old_err
                root.after(0, finish_worker)

        threading.Thread(target=worker, daemon=True).start()

    def stop():
        STOP_FLAG.set()
        # Also signal the engine's internal stop event for mid-transcription abort
        if not worker_state.get("existing_transcripts_only"):
            try:
                from transcribe_optimised import request_stop
                request_stop()
            except Exception:
                pass
        stop_btn.configure(state="disabled")
        stt_log.append(
            "\nCancellation requested. The current speech-to-text operation will stop safely; "
            "completed raw transcripts and GLM checkpoints are preserved.\n"
        )
        glm_log.append(
            "\nCancellation requested. No new GLM review will start; completed chunks are preserved.\n"
        )

    def clear_cache():
        # Clear cached model objects (frees VRAM without requiring re-download)
        try:
            from transcribe_optimised import _clear_model_cache
            _clear_model_cache()
        except Exception:
            pass
        gc.collect()
        gc.collect()  # second pass for cyclic references
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                torch.cuda.empty_cache()  # reclaim after ipc_collect
        except Exception:
            pass
        stt_log.append("Model cache cleared.\n")

    def configure_cleanup_access():
        try:
            subprocess.Popen(
                [sys.executable, os.path.join(REPO_ROOT, "configure_cleanup_credentials_gui.py")],
                cwd=REPO_ROOT,
            )
        except Exception as exc:
            messagebox.showerror("Could not open cleanup access", str(exc))

    def open_context_finder():
        try:
            _launch_context_finder_process()
        except Exception as exc:
            messagebox.showerror("Could not open Context Finder", str(exc))

    run_btn = _styled_btn(btn_bar, "  Start Polished Pipeline", start,
                          font=FONT_LG, bg=ACCENT)
    run_btn.pack(side="left", padx=(0, 8))
    stop_btn = _styled_btn(btn_bar, "  Stop", stop, font=FONT_LG, bg=RED)
    stop_btn.pack(side="left", padx=(0, 8))
    stop_btn.configure(state="disabled")
    clear_btn = _styled_btn(btn_bar, "Clear Cache", clear_cache,
                            font=FONT_LG, bg=AMBER)
    clear_btn.pack(side="left", padx=(0, 8))
    download_btn = _styled_btn(
        btn_bar, "Download Models", lambda: ModelPreloadDialog(root),
        font=FONT_LG, bg="#6366f1",
    )
    download_btn.pack(side="left", padx=(0, 8))
    access_btn = _styled_btn(
        btn_bar, "Cleanup Access", configure_cleanup_access,
        font=FONT_LG, bg="#475569",
    )
    access_btn.pack(side="left", padx=(0, 8))
    context_btn = _styled_btn(
        btn_bar, "Context Finder", open_context_finder,
        font=FONT_LG, bg="#0f766e",
    )
    context_btn.pack(side="left")

    def update_run_label(*_args):
        run_btn.configure(
            text=(
                (
                    "  Start GLM + Word (Skip Whisper)"
                    if settings_panel.existing_transcripts_var.get()
                    else "  Start Polished Pipeline"
                )
                if settings_panel.pipeline_var.get()
                else "  Start Local Transcription"
            )
        )

    settings_panel.pipeline_var.trace_add("write", update_run_label)
    settings_panel.existing_transcripts_var.trace_add("write", update_run_label)
    update_run_label()

    # ── Live pipeline lanes ──────────────────────────────────────────
    # The two panes expose the actual producer/consumer hand-off: one GPU
    # Parakeet worker continues with the next recording while ten GLM workers
    # independently review already-durable raw transcripts.
    logs = tk.Frame(outer, bg=BG)
    logs.grid(row=4, column=0, sticky="nsew", pady=(0, 0))
    logs.columnconfigure(0, weight=1)
    logs.columnconfigure(1, weight=1)
    logs.rowconfigure(1, weight=1)
    tk.Label(
        logs,
        text="Parakeet speech-to-text (one local GPU worker)",
        bg=BG,
        fg="#1d4ed8",
        font=("Segoe UI", 9, "bold"),
    ).grid(row=0, column=0, sticky="w", pady=(0, 4))
    tk.Label(
        logs,
        text="GLM review queue (ten protected workers)",
        bg=BG,
        fg="#0f766e",
        font=("Segoe UI", 9, "bold"),
    ).grid(row=0, column=1, sticky="w", padx=(12, 0), pady=(0, 4))
    stt_log = LogPanel(logs)
    stt_log.grid(row=1, column=0, sticky="nsew")
    glm_log = LogPanel(logs)
    glm_log.grid(row=1, column=1, sticky="nsew", padx=(12, 0))

    def on_window_close():
        outcome = _handle_window_close(
            worker_state,
            confirm_close=lambda: messagebox.askyesno(
                "Pipeline is still running",
                "Request a safe stop and close after the current operation finishes?\n\n"
                "The window must remain open until checkpointing or any active Word "
                "publication transaction completes.",
                parent=root,
            ),
            request_stop=stop,
            destroy=root.destroy,
        )
        if outcome == "stopping":
            stt_log.append(
                "The window will close automatically after the safe stop completes.\n"
            )

    root.protocol("WM_DELETE_WINDOW", on_window_close)

    # ── Queue poller ─────────────────────────────────────────────────
    def poll():
        try:
            while True:
                msg = q.get_nowait()
                if (
                    isinstance(msg, tuple)
                    and len(msg) == 3
                    and msg[0] == "progress"
                ):
                    _marker, lane, text = msg
                    (glm_log if lane == "glm" else stt_log).append(str(text))
                else:
                    stt_log.append(str(msg))
        except queue.Empty:
            pass
        root.after(120, poll)

    poll()
    root.mainloop()


# ═══════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Speech-to-Text Transcription Tool")
    parser.add_argument("--input",
                        help="Audio/video file or folder (headless mode)")
    parser.add_argument("--outdir", help="Output folder override")
    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--threads", type=int, help="CPU thread override")
    parser.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3",
                        help="Model: nvidia/parakeet-tdt-0.6b-v3 | faster-whisper-large-v3 | large-v3")
    args = parser.parse_args()

    if args.gui or not args.input:
        launch_gui()
        return

    # ── Headless ─────────────────────────────────────────────────────
    os.environ["TRANSCRIBE_MODEL_NAME"] = args.model
    if not args.model.startswith("faster-whisper-"):
        os.environ["TRANSCRIBE_FORCE_NATIVE_WHISPER"] = "1"
    os.environ["TRANSCRIBE_MAX_PERF"] = "1"
    os.environ["TRANSCRIBE_ALLOW_PROMPT"] = "1"
    terms = _default_terms_file()
    if terms:
        os.environ["TRANSCRIBE_AWKWARD_FILE"] = terms

    q: queue.Queue = queue.Queue()

    def runner():
        p = args.input
        if os.path.isdir(p):
            files = sorted(
                os.path.join(root, f)
                for root, _dirs, fnames in os.walk(p)
                for f in fnames
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
            )
            if files:
                _run_batch(files, q, threads=args.threads)
            else:
                q.put("No supported files found.\n")
        else:
            _run_single(p, args.outdir, q, threads=args.threads)

    t = threading.Thread(target=runner)
    t.start()
    while t.is_alive() or not q.empty():
        try:
            print(q.get(timeout=0.2), end="")
        except queue.Empty:
            pass


if __name__ == "__main__":
    main()
