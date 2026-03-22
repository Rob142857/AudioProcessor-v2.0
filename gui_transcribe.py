"""AudioProcessor v2.0 — Speech-to-Text Transcription Tool

Clean GUI for converting audio/video to text using Whisper AI.
  - Faster-Whisper large-v3 (GPU, CTranslate2 int8) -- recommended
  - Native Whisper large-v3 (GPU / CPU fallback)
  - Single file or recursive batch processing
  - Skip / replace / replace-before-date for existing outputs
"""
import argparse
import datetime
import gc
import json
import os
import queue
import subprocess
import sys
import threading
from typing import List, Optional

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
            return json.load(f)
    except Exception:
        return {}


def _save_settings(data: dict) -> None:
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data or {}, f, indent=2)
    except Exception:
        pass


def _load_project(folder: str) -> dict:
    return _load_settings().get("projects", {}).get(folder, {})


def _save_project(folder: str, proj: dict) -> None:
    all_s = _load_settings()
    all_s.setdefault("projects", {})[folder] = proj
    _save_settings(all_s)


def _default_terms_file() -> Optional[str]:
    p = os.path.join(REPO_ROOT, "special_words.txt")
    return p if os.path.isfile(p) else None


# ── Transcription helpers ────────────────────────────────────────────
def _run_single(path: str, outdir: Optional[str], q: queue.Queue,
                *, threads: Optional[int] = None):
    """Transcribe one file, writing progress to *q*."""
    if STOP_FLAG.is_set():
        q.put("Cancelled.\n")
        return
    try:
        from transcribe_optimised import transcribe_file_simple_auto
        target = outdir or os.path.dirname(path)
        out = transcribe_file_simple_auto(path, output_dir=target,
                                          threads_override=threads)
        if out and os.path.isfile(out):
            q.put(f"Done -> {out}\n")
        else:
            q.put("Warning: no output generated.\n")
    except Exception as e:
        import traceback
        q.put(f"Error: {e}\n{traceback.format_exc()}")


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
            _run_single(p, None, q, threads=threads)
            ok += 1
        except Exception as e:
            fail += 1
            q.put(f"Error ({os.path.basename(p)}): {e}\n")
    q.put(f"\nBatch complete -- {ok} succeeded, {fail} failed.\n")


# ── Output-skip logic ────────────────────────────────────────────────
def _should_process(src: str, mode: str, cutoff_date: str) -> bool:
    """Return True if this file should be (re-)transcribed."""
    docx = os.path.splitext(src)[0] + ".docx"
    if not os.path.isfile(docx):
        return True  # no existing output
    if mode == "all":
        return True
    if mode == "before":
        try:
            cutoff = datetime.datetime.strptime(cutoff_date, "%Y-%m-%d")
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(docx))
            return mtime < cutoff
        except Exception:
            return True
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
    last_folder = last_settings.get("last_folder", REPO_ROOT)
    proj = _load_project(last_folder)

    def on_folder_selected(folder):
        nonlocal proj
        proj = _load_project(folder)
        settings_panel.apply(proj)

    # ── Input panel ──────────────────────────────────────────────────
    input_panel = InputPanel(outer, on_folder_selected=on_folder_selected)
    input_panel.grid(row=1, column=0, sticky="ew", pady=(0, 10))

    # ── Settings panel ───────────────────────────────────────────────
    settings_panel = SettingsPanel(outer, proj_settings=proj)
    settings_panel.grid(row=2, column=0, sticky="ew", pady=(0, 10))

    # ── Buttons ──────────────────────────────────────────────────────
    btn_bar = tk.Frame(outer, bg=BG)
    btn_bar.grid(row=3, column=0, sticky="w", pady=(0, 10))

    q: queue.Queue = queue.Queue()

    def start():
        inp = input_panel.get_path()
        if not inp or not os.path.exists(inp):
            messagebox.showerror("No input",
                                 "Select a valid file or folder first.")
            return
        log.clear()
        log.append("Starting...\n")
        run_btn.configure(state="disabled")
        stop_btn.configure(state="normal")
        STOP_FLAG.clear()

        snap = settings_panel.snapshot()

        # Persist project settings
        folder = os.path.dirname(inp) if os.path.isfile(inp) else inp
        _save_project(folder, snap)
        s = _load_settings()
        s["last_folder"] = folder
        _save_settings(s)

        def worker():
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = _QueueWriter(q)
            sys.stderr = _QueueWriter(q)
            try:
                # Apply env vars from settings snapshot
                os.environ["TRANSCRIBE_MODEL_NAME"] = snap["whisper_model"]
                if snap["whisper_model"].startswith("faster-whisper-"):
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

                if os.path.isdir(inp):
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
                q.put("\nDone.\n")
            except Exception as e:
                import traceback
                q.put(f"Error: {e}\n{traceback.format_exc()}")
            finally:
                sys.stdout, sys.stderr = old_out, old_err
                root.after(0, lambda: run_btn.configure(state="normal"))
                root.after(0, lambda: stop_btn.configure(state="disabled"))

        threading.Thread(target=worker, daemon=True).start()

    def stop():
        STOP_FLAG.set()
        stop_btn.configure(state="disabled")
        log.append("\nStopping...\n")

    def clear_cache():
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        log.append("Cache cleared.\n")

    run_btn = _styled_btn(btn_bar, "  Start Transcription", start,
                          font=FONT_LG, bg=ACCENT)
    run_btn.pack(side="left", padx=(0, 8))
    stop_btn = _styled_btn(btn_bar, "  Stop", stop, font=FONT_LG, bg=RED)
    stop_btn.pack(side="left", padx=(0, 8))
    stop_btn.configure(state="disabled")
    _styled_btn(btn_bar, "Clear Cache", clear_cache,
                font=FONT_LG, bg=AMBER).pack(side="left", padx=(0, 8))
    _styled_btn(btn_bar, "Download Models", lambda: ModelPreloadDialog(root),
                font=FONT_LG, bg="#6366f1").pack(side="left")

    # ── Log panel ────────────────────────────────────────────────────
    log = LogPanel(outer)
    log.grid(row=4, column=0, sticky="nsew", pady=(0, 0))

    # ── Queue poller ─────────────────────────────────────────────────
    def poll():
        try:
            while True:
                msg = q.get_nowait()
                log.append(msg)
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
    parser.add_argument("--model", default="faster-whisper-large-v3",
                        help="Model: faster-whisper-large-v3 | large-v3")
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
