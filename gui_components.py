"""Reusable GUI components for the transcription tool."""
import os
import datetime
import threading
import tkinter as tk
from tkinter import ttk, filedialog

SUPPORTED_EXTS = (
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma",
    ".aiff", ".aif", ".3gp",
    ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm",
)

# ── Colours & fonts ──────────────────────────────────────────────────
BG       = "#f8f9fa"
CARD_BG  = "white"
FG       = "#374151"
FG_DIM   = "#6b7280"
ACCENT   = "#007acc"
GREEN    = "#059669"
RED      = "#dc2626"
AMBER    = "#f59e0b"
ENTRY_BG = "#f9fafb"
ENTRY_FG = "#111827"
FONT     = ("Segoe UI", 10)
FONT_SM  = ("Segoe UI", 9)
FONT_LG  = ("Segoe UI", 12, "bold")
FONT_TTL = ("Segoe UI", 18, "bold")


def _styled_btn(parent, text, command, *, bg=ACCENT, fg="white", font=FONT_SM, **kw):
    return tk.Button(
        parent, text=text, command=command, font=font,
        bg=bg, fg=fg, relief="flat", bd=0, padx=14, pady=6,
        activebackground=bg, activeforeground=fg, cursor="hand2", **kw,
    )


# ─────────────────────────────────────────────────────────────────────
# Input panel: file/folder selection
# ─────────────────────────────────────────────────────────────────────
class InputPanel(tk.Frame):
    """File or folder picker with status line."""

    def __init__(
        self,
        parent,
        on_folder_selected=None,
        *,
        on_source_selected=None,
        initial_path: str = "",
    ):
        super().__init__(parent, bg=CARD_BG)
        # ``on_folder_selected`` is retained for callers of the original GUI.
        self._on_source_selected = on_source_selected or on_folder_selected
        self.columnconfigure(1, weight=1)

        tk.Label(self, text="Audio / Video source:", bg=CARD_BG, fg=FG,
                 font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 4))

        self.path_var = tk.StringVar(value=initial_path)
        tk.Entry(self, textvariable=self.path_var, font=FONT, relief="flat",
                 bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=ENTRY_FG
                 ).grid(row=0, column=1, sticky="ew", padx=(8, 4), pady=(14, 4))

        btn_frame = tk.Frame(self, bg=CARD_BG)
        btn_frame.grid(row=0, column=2, padx=(4, 16), pady=(14, 4))
        _styled_btn(btn_frame, "File…", self._browse_file).pack(side="left", padx=(0, 4))
        _styled_btn(btn_frame, "Folder…", self._browse_folder, bg="#0ea5e9").pack(side="left")

        self.status = tk.Label(self, text="No source selected", bg=CARD_BG, fg=FG_DIM, font=FONT_SM)
        self.status.grid(row=1, column=0, columnspan=3, sticky="w", padx=16, pady=(0, 10))
        if initial_path and os.path.exists(initial_path):
            self._show_status(initial_path)

    # ── helpers ──
    def _browse_file(self):
        types = [
            ("Audio files", "*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.wma *.aiff *.aif *.3gp"),
            ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
            ("All files", "*.*"),
        ]
        p = filedialog.askopenfilename(title="Select audio / video file", filetypes=types)
        if p:
            self.path_var.set(p)
            self._show_status(p)
            if self._on_source_selected:
                self._on_source_selected(p)

    def _browse_folder(self):
        d = filedialog.askdirectory(title="Select folder for batch processing")
        if d:
            self.path_var.set(d)
            self._show_status(d)
            if self._on_source_selected:
                self._on_source_selected(d)

    def _show_status(self, path: str) -> None:
        if os.path.isfile(path):
            self.status.config(text=f"File: {os.path.basename(path)}", fg=GREEN)
            return
        if os.path.isdir(path):
            try:
                n = sum(
                    1
                    for f in os.listdir(path)
                    if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
                )
            except OSError:
                n = 0
            self.status.config(
                text=f"Folder: {os.path.basename(path)}  ({n} files at this level)",
                fg=GREEN,
            )

    def get_path(self) -> str:
        return self.path_var.get().strip()


# ─────────────────────────────────────────────────────────────────────
# Settings panel: model, recursive, replace-outputs, quality
# ─────────────────────────────────────────────────────────────────────
class SettingsPanel(tk.Frame):
    """All transcription settings in a compact card."""

    MODEL_OPTIONS = [
        ("nvidia/parakeet-tdt-0.6b-v3",     "NVIDIA Parakeet TDT 0.6B v3  (GPU, fast archive default)"),
        ("faster-whisper-large-v3",       "Faster-Whisper Large-v3  (GPU 4GB+, best quality)"),
        ("faster-whisper-large-v3-turbo", "Faster-Whisper Large-v3-turbo  (GPU 4GB+, 2x faster)"),
        ("faster-whisper-medium",         "Faster-Whisper Medium  (GPU 2GB+, good quality)"),
        ("faster-whisper-small",          "Faster-Whisper Small  (GPU 1GB+, fast)"),
        ("large-v3",                      "Native Whisper Large-v3  (GPU / CPU fallback)"),
    ]

    def __init__(self, parent, proj_settings: dict | None = None):
        super().__init__(parent, bg=CARD_BG)
        ps = proj_settings or {}
        self.columnconfigure(1, weight=1)
        row = 0

        # Valid model keys for validation
        valid_models = {k for k, _ in self.MODEL_OPTIONS}

        # Polished pipeline mode is deliberately the safe, default path.  The
        # description is informational rather than an editable model selector:
        # cleanup remains pinned by archive_pipeline.py.
        mode_frame = tk.Frame(self, bg=CARD_BG)
        mode_frame.grid(
            row=row, column=0, columnspan=3, sticky="ew", padx=16, pady=(14, 4)
        )
        self.pipeline_var = tk.IntVar(value=int(bool(ps.get("polished_pipeline", 1))))
        tk.Checkbutton(
            mode_frame,
            text="Polished archive pipeline",
            variable=self.pipeline_var,
            bg=CARD_BG,
            fg=FG,
            selectcolor=CARD_BG,
            activebackground=CARD_BG,
            font=("Segoe UI", 10, "bold"),
            command=self._toggle_pipeline,
        ).pack(side="left")
        self.pipeline_flow = tk.Label(
            self,
            text=(
                "Local NVIDIA Parakeet Word  →  five protected GLM-4.7-Flash review workers  "
                "→  separate GLM Review Word"
            ),
            bg="#eff6ff",
            fg="#1d4ed8",
            font=FONT_SM,
            anchor="w",
            padx=10,
            pady=6,
        )
        self.pipeline_flow.grid(
            row=row + 1, column=0, columnspan=3, sticky="ew", padx=16, pady=(0, 6)
        )
        self.pipeline_note = tk.Label(
            self,
            text=(
                "Completed recordings, stages, and GLM chunks resume after interruption; "
                "a recording stopped during speech-to-text restarts from its beginning. Raw "
                "speech Word files and separate GLM Review copies are kept side by side."
            ),
            bg=CARD_BG,
            fg=FG_DIM,
            font=("Segoe UI", 8),
            anchor="w",
        )
        self.pipeline_note.grid(
            row=row + 2, column=0, columnspan=3, sticky="w", padx=18, pady=(0, 4)
        )

        self.existing_transcripts_frame = tk.Frame(self, bg=CARD_BG)
        self.existing_transcripts_frame.grid(
            row=row + 3, column=0, columnspan=3, sticky="ew", padx=16, pady=(2, 0)
        )
        self.existing_transcripts_var = tk.IntVar(
            value=int(bool(ps.get("existing_transcripts_only", 0)))
        )
        self.existing_transcripts_check = tk.Checkbutton(
            self.existing_transcripts_frame,
            text="Use existing Word transcripts (skip speech-to-text)",
            variable=self.existing_transcripts_var,
            bg=CARD_BG,
            fg=FG,
            selectcolor=CARD_BG,
            activebackground=CARD_BG,
            font=("Segoe UI", 10, "bold"),
            command=self._toggle_pipeline,
        )
        self.existing_transcripts_check.pack(side="left")
        self.existing_transcripts_note = tk.Label(
            self,
            text="",
            bg=CARD_BG,
            fg=FG_DIM,
            font=("Segoe UI", 8),
            anchor="w",
        )
        self.existing_transcripts_note.grid(
            row=row + 4, column=0, columnspan=3, sticky="w", padx=18, pady=(0, 4)
        )
        row += 5

        # Model
        tk.Label(self, text="Model:", bg=CARD_BG, fg=FG,
                 font=("Segoe UI", 10, "bold")).grid(row=row, column=0, sticky="w", padx=16, pady=(4, 6))
        saved_model = ps.get("whisper_model", "nvidia/parakeet-tdt-0.6b-v3")
        if saved_model not in valid_models:
            saved_model = "nvidia/parakeet-tdt-0.6b-v3"  # reset stale/invalid model names
        self.model_var = tk.StringVar(value=saved_model)
        display_map = {k: v for k, v in self.MODEL_OPTIONS}
        self._display_var = tk.StringVar(value=display_map.get(self.model_var.get(), self.MODEL_OPTIONS[0][1]))
        self.model_combo = ttk.Combobox(
            self,
            textvariable=self._display_var,
            values=[v for _, v in self.MODEL_OPTIONS],
            state="readonly",
            width=46,
            font=FONT_SM,
        )
        self.model_combo.grid(
            row=row, column=1, columnspan=2, sticky="w", padx=(8, 16), pady=(4, 6)
        )
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        # ── Row: Recursive + Quality ──
        row += 1
        chk_frame = tk.Frame(self, bg=CARD_BG)
        chk_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=16, pady=(4, 2))

        self.recursive_var = tk.IntVar(value=ps.get("recursive", 1))
        tk.Checkbutton(chk_frame, text="Recursive (include subfolders)", variable=self.recursive_var,
                       bg=CARD_BG, fg=FG, selectcolor=CARD_BG, activebackground=CARD_BG,
                       font=FONT_SM).pack(side="left", padx=(0, 24))

        self.quality_var = tk.IntVar(value=ps.get("quality_mode", 1))
        self.quality_check = tk.Checkbutton(
            chk_frame,
            text="Quality mode (beam search)",
            variable=self.quality_var,
            bg=CARD_BG,
            fg=FG,
            selectcolor=CARD_BG,
            activebackground=CARD_BG,
            font=FONT_SM,
        )
        self.quality_check.pack(side="left")

        # ── Row: Replace outputs ──
        row += 1
        repl_frame = tk.Frame(self, bg=CARD_BG)
        repl_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=16, pady=(4, 2))

        self.replace_var = tk.StringVar(value=ps.get("replace_mode", "skip"))  # skip | all | before
        tk.Label(repl_frame, text="Processing selection:", bg=CARD_BG, fg=FG, font=FONT_SM).pack(side="left", padx=(0, 6))
        for val, label in [
            ("skip", "Skip existing"),
            ("all", "Refresh all"),
            ("before", "Refresh transcripts before…"),
        ]:
            tk.Radiobutton(repl_frame, text=label, variable=self.replace_var, value=val,
                           bg=CARD_BG, fg=FG, selectcolor=CARD_BG, activebackground=CARD_BG,
                           font=FONT_SM, command=self._on_replace_change).pack(side="left", padx=(0, 10))

        # ── Row: Date picker (shown only when "before" selected) ──
        row += 1
        self._date_frame = tk.Frame(self, bg=CARD_BG)
        self._date_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=40, pady=(0, 6))

        tk.Label(self._date_frame, text="Before:", bg=CARD_BG, fg=FG_DIM, font=FONT_SM).pack(side="left")
        self.date_var = tk.StringVar(value=ps.get("replace_before_date", datetime.date.today().isoformat()))
        tk.Entry(self._date_frame, textvariable=self.date_var, width=12,
                 bg=ENTRY_BG, fg=ENTRY_FG, relief="flat", font=FONT_SM).pack(side="left", padx=(4, 8))
        tk.Label(self._date_frame, text="(YYYY-MM-DD)", bg=CARD_BG, fg="#9ca3af", font=("Segoe UI", 8)).pack(side="left")

        self._toggle_date()  # initial visibility

        # Force is intentionally separate from the source-DOCX selection
        # policy.  Ordinarily the polished pipeline resumes valid checkpoints.
        row += 1
        advanced_frame = tk.Frame(self, bg=CARD_BG)
        advanced_frame.grid(
            row=row, column=0, columnspan=3, sticky="w", padx=16, pady=(2, 4)
        )
        self.force_var = tk.IntVar(value=int(bool(ps.get("force_reprocess", 0))))
        self.force_check = tk.Checkbutton(
            advanced_frame,
            text="Reprocess from audio even when verified checkpoints exist (slow)",
            variable=self.force_var,
            bg=CARD_BG,
            fg=FG_DIM,
            selectcolor=CARD_BG,
            activebackground=CARD_BG,
            font=FONT_SM,
        )
        self.force_check.pack(side="left")

        row += 1
        retention_frame = tk.Frame(self, bg=CARD_BG)
        retention_frame.grid(
            row=row, column=0, columnspan=3, sticky="w", padx=16, pady=(0, 4)
        )
        self.retain_troubleshooting_var = tk.IntVar(
            value=int(bool(ps.get("retain_troubleshooting_artifacts", 1)))
        )
        tk.Checkbutton(
            retention_frame,
            text="Retain detailed troubleshooting logging",
            variable=self.retain_troubleshooting_var,
            bg=CARD_BG,
            fg=FG_DIM,
            selectcolor=CARD_BG,
            activebackground=CARD_BG,
            font=FONT_SM,
        ).pack(side="left")
        tk.Label(
            retention_frame,
            text="(compact hashes and resume checkpoints are always kept)",
            bg=CARD_BG,
            fg="#9ca3af",
            font=("Segoe UI", 8),
        ).pack(side="left", padx=(8, 0))

        # Bottom padding
        row += 1
        tk.Frame(self, bg=CARD_BG, height=10).grid(row=row, column=0)
        self._toggle_pipeline()

    # ── helpers ──
    def _on_model_change(self, _evt=None):
        rev = {v: k for k, v in self.MODEL_OPTIONS}
        self.model_var.set(rev.get(self._display_var.get(), self.MODEL_OPTIONS[0][0]))
        self._toggle_pipeline()

    def _toggle_date(self):
        if self.replace_var.get() == "before":
            for w in self._date_frame.winfo_children():
                try:
                    w.configure(state="normal")  # type: ignore[arg-type]
                except Exception:
                    pass
            self._date_frame.grid()
        else:
            self._date_frame.grid_remove()

    def _on_replace_change(self):
        self._toggle_date()
        self._update_existing_transcripts_note()

    def _update_existing_transcripts_note(self):
        conflict = (
            bool(self.pipeline_var.get())
            and bool(self.existing_transcripts_var.get())
            and self.replace_var.get() == "skip"
        )
        if conflict:
            self.existing_transcripts_note.configure(
                text=(
                    "Choose Refresh all or Refresh transcripts before; Skip existing "
                    "selects no source Word transcripts."
                ),
                fg="#b45309",
            )
        else:
            self.existing_transcripts_note.configure(
                text=(
                    "Requires a source-adjacent Whisper DOCX. It remains byte-for-byte "
                    "unchanged; GLM output is written as a separate Review copy."
                ),
                fg=FG_DIM,
            )

    def _toggle_pipeline(self):
        enabled = bool(self.pipeline_var.get())
        existing_only = enabled and bool(self.existing_transcripts_var.get())
        uses_parakeet = self.model_var.get().casefold().startswith("nvidia/parakeet-")
        self.pipeline_flow.configure(
            fg="#1d4ed8" if enabled else FG_DIM,
            text=(
                (
                    "Existing Whisper Word  →  protected GLM-4.7-Flash cleanup  "
                    "→  separate GLM Review Word (no audio inference)"
                    if existing_only
                else (
                    "Local NVIDIA Parakeet Word  →  five protected GLM-4.7-Flash review workers  "
                    "→  separate GLM Review Word"
                    if uses_parakeet
                    else "Local Faster-Whisper Word  →  protected GLM-4.7-Flash cleanup  "
                    "→  separate GLM Review Word"
                )
                )
                if enabled
                else "Local transcription only (legacy mode; no protected cleanup)"
            ),
        )
        self.force_check.configure(state="normal" if enabled else "disabled")
        self.force_check.configure(
            text=(
                "Rerun GLM cleanup and Word rendering even when verified checkpoints exist"
                if existing_only
                else "Reprocess from audio even when verified checkpoints exist (slow)"
            )
        )
        self.existing_transcripts_check.configure(
            state="normal" if enabled else "disabled"
        )
        self.model_combo.configure(state="disabled" if existing_only else "readonly")
        self.quality_check.configure(
            state="disabled" if existing_only or uses_parakeet else "normal"
        )
        if enabled:
            self.pipeline_note.grid()
            self.existing_transcripts_frame.grid()
            self.existing_transcripts_note.grid()
        else:
            self.pipeline_note.grid_remove()
            self.existing_transcripts_frame.grid_remove()
            self.existing_transcripts_note.grid_remove()
        self._update_existing_transcripts_note()

    def apply(self, ps: dict) -> None:
        """Apply a project-settings dict to the panel widgets."""
        valid_models = {k for k, _ in self.MODEL_OPTIONS}
        display_map = {k: v for k, v in self.MODEL_OPTIONS}
        saved_model = ps.get("whisper_model", "nvidia/parakeet-tdt-0.6b-v3")
        if saved_model not in valid_models:
            saved_model = "nvidia/parakeet-tdt-0.6b-v3"  # reset stale/invalid model names
        self.model_var.set(saved_model)
        self._display_var.set(display_map.get(saved_model, self.MODEL_OPTIONS[0][1]))
        self.pipeline_var.set(int(bool(ps.get("polished_pipeline", 1))))
        self.existing_transcripts_var.set(
            int(bool(ps.get("existing_transcripts_only", 0)))
        )
        self.recursive_var.set(ps.get("recursive", 1))
        self.quality_var.set(ps.get("quality_mode", 1))
        self.replace_var.set(ps.get("replace_mode", "skip"))
        self.date_var.set(ps.get("replace_before_date", datetime.date.today().isoformat()))
        self.force_var.set(int(bool(ps.get("force_reprocess", 0))))
        self.retain_troubleshooting_var.set(
            int(bool(ps.get("retain_troubleshooting_artifacts", 1)))
        )
        self._toggle_date()
        self._toggle_pipeline()

    def snapshot(self) -> dict:
        """Return current settings as a dict for persistence."""
        return {
            "polished_pipeline": self.pipeline_var.get(),
            "existing_transcripts_only": self.existing_transcripts_var.get(),
            "whisper_model": self.model_var.get(),
            "recursive": self.recursive_var.get(),
            "quality_mode": self.quality_var.get(),
            "replace_mode": self.replace_var.get(),
            "replace_before_date": self.date_var.get(),
            "force_reprocess": self.force_var.get(),
            "retain_troubleshooting_artifacts": self.retain_troubleshooting_var.get(),
        }


# ─────────────────────────────────────────────────────────────────────
# Log panel with scrolling text
# ─────────────────────────────────────────────────────────────────────
class LogPanel(tk.Frame):
    """Scrolling activity log."""

    def __init__(self, parent):
        super().__init__(parent, bg=CARD_BG, relief="flat", bd=1)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.text = tk.Text(self, wrap=tk.WORD, font=FONT_SM, bg=CARD_BG, fg="#2c3e50",
                            bd=0, highlightthickness=0, insertbackground="#2c3e50")
        self.text.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.text.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.text["yscrollcommand"] = sb.set

        self.text.tag_configure("title", foreground=ACCENT, font=("Segoe UI", 9, "bold"))
        self.text.tag_configure("info", foreground="#60a5fa")
        self.text.tag_configure("error", foreground=RED)

        self.append("Speech-to-Text Transcription Tool\n", "title")
        self.append("Select a file or folder and press Start.\n\n")

    def append(self, msg: str, tag: str | None = None):
        self.text.configure(state="normal")
        self.text.insert("end", msg, tag or ())
        self.text.see("end")
        self.text.configure(state="disabled")

    def clear(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")


# ─────────────────────────────────────────────────────────────────────
# Model preload dialog
# ─────────────────────────────────────────────────────────────────────
# Available models that can be preloaded for offline use.
PRELOAD_MODELS = [
    ("fw-large-v3",       "Faster-Whisper Large-v3",       "large-v3",       "faster-whisper", True),
    ("fw-large-v3-turbo", "Faster-Whisper Large-v3-turbo", "large-v3-turbo", "faster-whisper", True),
    ("native-large-v3",   "Native Whisper Large-v3",       "large-v3",       "native",         False),
]
# Columns: (key, display_name, model_id, backend, default_checked)


class ModelPreloadDialog(tk.Toplevel):
    """Modal dialog to download / cache selected Whisper models."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Download Models")
        self.geometry("520x420")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        tk.Label(self, text="Model Manager", bg=BG, fg="#1a365d",
                 font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=20, pady=(16, 4))
        tk.Label(self, text="Select models to download for offline use.",
                 bg=BG, fg=FG_DIM, font=FONT_SM).pack(anchor="w", padx=20, pady=(0, 12))

        # Model checkboxes with status
        self._vars: dict[str, tk.IntVar] = {}
        self._status_labels: dict[str, tk.Label] = {}
        card = tk.Frame(self, bg=CARD_BG, relief="flat", bd=1)
        card.pack(fill="x", padx=20, pady=(0, 12))

        for key, display, model_id, backend, default in PRELOAD_MODELS:
            row = tk.Frame(card, bg=CARD_BG)
            row.pack(fill="x", padx=12, pady=4)

            var = tk.IntVar(value=1 if default else 0)
            self._vars[key] = var
            tk.Checkbutton(row, text=display, variable=var,
                           bg=CARD_BG, fg=FG, selectcolor=CARD_BG,
                           activebackground=CARD_BG, font=FONT).pack(side="left")

            status = tk.Label(row, text="", bg=CARD_BG, fg=FG_DIM, font=FONT_SM)
            status.pack(side="right", padx=(0, 4))
            self._status_labels[key] = status

            # Check if already cached
            self._check_cached(key, model_id, backend, status)

        # Progress log
        self._log = tk.Text(self, wrap=tk.WORD, height=8, font=FONT_SM,
                            bg=CARD_BG, fg="#2c3e50", bd=0, highlightthickness=0)
        self._log.pack(fill="both", expand=True, padx=20, pady=(0, 8))
        self._log.configure(state="disabled")

        # Buttons
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(fill="x", padx=20, pady=(0, 16))
        self._dl_btn = _styled_btn(btn_frame, "Download Selected", self._start_download,
                                   bg=GREEN, font=("Segoe UI", 11, "bold"))
        self._dl_btn.pack(side="left", padx=(0, 8))
        _styled_btn(btn_frame, "Close", self.destroy, bg="#6b7280",
                    font=("Segoe UI", 11)).pack(side="right")

    # ── helpers ──────────────────────────────────────────────────────
    def _log_msg(self, msg: str):
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    @staticmethod
    def _check_cached(key, model_id, backend, label):
        """Set label text to 'cached' or 'not downloaded'."""
        try:
            if backend == "faster-whisper":
                from huggingface_hub import try_to_load_from_cache
                # Faster-whisper models are stored under Systran/ or mobiuslabsgmbh/
                for prefix in (f"Systran/faster-whisper-{model_id}",
                               f"mobiuslabsgmbh/faster-whisper-{model_id}"):
                    if try_to_load_from_cache(prefix, "model.bin") is not None:
                        label.config(text="cached", fg=GREEN)
                        return
                label.config(text="not downloaded", fg=AMBER)
            else:
                import torch
                cache = os.path.join(torch.hub.get_dir(), "checkpoints")
                if any(model_id in f for f in os.listdir(cache) if f.endswith(".pt")):
                    label.config(text="cached", fg=GREEN)
                    return
                label.config(text="not downloaded", fg=AMBER)
        except Exception:
            label.config(text="unknown", fg=FG_DIM)

    def _start_download(self):
        selected = [(k, m) for k, _d, m, b, _df in PRELOAD_MODELS
                    if self._vars.get(k, tk.IntVar()).get() == 1
                    for m, b in [(m, b)]]
        # Rebuild properly
        selected = []
        for key, display, model_id, backend, _default in PRELOAD_MODELS:
            if self._vars.get(key, tk.IntVar()).get() == 1:
                selected.append((key, display, model_id, backend))

        if not selected:
            self._log_msg("No models selected.")
            return

        self._dl_btn.configure(state="disabled")
        self._log_msg(f"Downloading {len(selected)} model(s)...\n")

        def worker():
            for key, display, model_id, backend in selected:
                self._log_msg(f"Loading {display}...")
                lbl = self._status_labels[key]
                try:
                    if backend == "faster-whisper":
                        os.environ.pop("HF_HUB_OFFLINE", None)
                        from faster_whisper import WhisperModel
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        ctype = "int8"
                        if device == "cuda":
                            try:
                                cap = torch.cuda.get_device_capability(0)
                                if cap[0] >= 7:
                                    ctype = "float16"
                            except Exception:
                                pass
                        m = WhisperModel(model_id, device=device, compute_type=ctype)
                        del m
                    else:
                        import whisper
                        whisper.load_model(model_id)
                    self._log_msg(f"  OK: {display}")
                    self.after(0, lambda l=lbl: l.config(text="cached", fg=GREEN))
                except Exception as e:
                    self._log_msg(f"  FAILED: {e}")
                    self.after(0, lambda l=lbl: l.config(text="failed", fg=RED))

            self._log_msg("\nDone.")
            self.after(0, lambda: self._dl_btn.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()
