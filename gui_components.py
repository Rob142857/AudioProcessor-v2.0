"""Reusable GUI components for the transcription tool."""
import os
import datetime
import tkinter as tk
from tkinter import ttk, filedialog

SUPPORTED_EXTS = (
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma",
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

    def __init__(self, parent, on_folder_selected=None):
        super().__init__(parent, bg=CARD_BG)
        self._on_folder_selected = on_folder_selected
        self.columnconfigure(1, weight=1)

        tk.Label(self, text="Audio / Video source:", bg=CARD_BG, fg=FG,
                 font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 4))

        self.path_var = tk.StringVar()
        tk.Entry(self, textvariable=self.path_var, font=FONT, relief="flat",
                 bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=ENTRY_FG
                 ).grid(row=0, column=1, sticky="ew", padx=(8, 4), pady=(14, 4))

        btn_frame = tk.Frame(self, bg=CARD_BG)
        btn_frame.grid(row=0, column=2, padx=(4, 16), pady=(14, 4))
        _styled_btn(btn_frame, "File…", self._browse_file).pack(side="left", padx=(0, 4))
        _styled_btn(btn_frame, "Folder…", self._browse_folder, bg="#0ea5e9").pack(side="left")

        self.status = tk.Label(self, text="No source selected", bg=CARD_BG, fg=FG_DIM, font=FONT_SM)
        self.status.grid(row=1, column=0, columnspan=3, sticky="w", padx=16, pady=(0, 10))

    # ── helpers ──
    def _browse_file(self):
        types = [
            ("Audio files", "*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.wma"),
            ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
            ("All files", "*.*"),
        ]
        p = filedialog.askopenfilename(title="Select audio / video file", filetypes=types)
        if p:
            self.path_var.set(p)
            self.status.config(text=f"File: {os.path.basename(p)}", fg=GREEN)

    def _browse_folder(self):
        d = filedialog.askdirectory(title="Select folder for batch processing")
        if d:
            self.path_var.set(d)
            n = sum(1 for f in os.listdir(d) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS)
            self.status.config(text=f"Folder: {os.path.basename(d)}  ({n} files)", fg=GREEN)
            if self._on_folder_selected:
                self._on_folder_selected(d)

    def get_path(self) -> str:
        return self.path_var.get().strip()


# ─────────────────────────────────────────────────────────────────────
# Settings panel: model, recursive, replace-outputs, quality
# ─────────────────────────────────────────────────────────────────────
class SettingsPanel(tk.Frame):
    """All transcription settings in a compact card."""

    MODEL_OPTIONS = [
        ("faster-whisper-large-v3", "Faster-Whisper Large-v3  (GPU, recommended)"),
        ("large-v3",                "Native Whisper Large-v3  (GPU / CPU fallback)"),
    ]

    def __init__(self, parent, proj_settings: dict | None = None):
        super().__init__(parent, bg=CARD_BG)
        ps = proj_settings or {}
        self.columnconfigure(1, weight=1)
        row = 0

        # Model
        tk.Label(self, text="Model:", bg=CARD_BG, fg=FG,
                 font=("Segoe UI", 10, "bold")).grid(row=row, column=0, sticky="w", padx=16, pady=(14, 6))
        self.model_var = tk.StringVar(value=ps.get("whisper_model", "faster-whisper-large-v3"))
        display_map = {k: v for k, v in self.MODEL_OPTIONS}
        self._display_var = tk.StringVar(value=display_map.get(self.model_var.get(), self.MODEL_OPTIONS[0][1]))
        combo = ttk.Combobox(self, textvariable=self._display_var,
                             values=[v for _, v in self.MODEL_OPTIONS],
                             state="readonly", width=46, font=FONT_SM)
        combo.grid(row=row, column=1, columnspan=2, sticky="w", padx=(8, 16), pady=(14, 6))
        combo.bind("<<ComboboxSelected>>", self._on_model_change)

        # ── Row: Recursive + Quality ──
        row += 1
        chk_frame = tk.Frame(self, bg=CARD_BG)
        chk_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=16, pady=(4, 2))

        self.recursive_var = tk.IntVar(value=ps.get("recursive", 0))
        tk.Checkbutton(chk_frame, text="Recursive (include subfolders)", variable=self.recursive_var,
                       bg=CARD_BG, fg=FG, selectcolor=CARD_BG, activebackground=CARD_BG,
                       font=FONT_SM).pack(side="left", padx=(0, 24))

        self.quality_var = tk.IntVar(value=ps.get("quality_mode", 1))
        tk.Checkbutton(chk_frame, text="Quality mode (beam search)", variable=self.quality_var,
                       bg=CARD_BG, fg=FG, selectcolor=CARD_BG, activebackground=CARD_BG,
                       font=FONT_SM).pack(side="left")

        # ── Row: Replace outputs ──
        row += 1
        repl_frame = tk.Frame(self, bg=CARD_BG)
        repl_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=16, pady=(4, 2))

        self.replace_var = tk.StringVar(value=ps.get("replace_mode", "skip"))  # skip | all | before
        tk.Label(repl_frame, text="Existing outputs:", bg=CARD_BG, fg=FG, font=FONT_SM).pack(side="left", padx=(0, 6))
        for val, label in [("skip", "Skip"), ("all", "Replace all"), ("before", "Replace if before…")]:
            tk.Radiobutton(repl_frame, text=label, variable=self.replace_var, value=val,
                           bg=CARD_BG, fg=FG, selectcolor=CARD_BG, activebackground=CARD_BG,
                           font=FONT_SM, command=self._toggle_date).pack(side="left", padx=(0, 10))

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

        # Bottom padding
        row += 1
        tk.Frame(self, bg=CARD_BG, height=10).grid(row=row, column=0)

    # ── helpers ──
    def _on_model_change(self, _evt=None):
        rev = {v: k for k, v in self.MODEL_OPTIONS}
        self.model_var.set(rev.get(self._display_var.get(), self.MODEL_OPTIONS[0][0]))

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

    def apply(self, ps: dict) -> None:
        """Apply a project-settings dict to the panel widgets."""
        display_map = {k: v for k, v in self.MODEL_OPTIONS}
        self.model_var.set(ps.get("whisper_model", "faster-whisper-large-v3"))
        self._display_var.set(display_map.get(self.model_var.get(), self.MODEL_OPTIONS[0][1]))
        self.recursive_var.set(ps.get("recursive", 0))
        self.quality_var.set(ps.get("quality_mode", 1))
        self.replace_var.set(ps.get("replace_mode", "skip"))
        self.date_var.set(ps.get("replace_before_date", datetime.date.today().isoformat()))
        self._toggle_date()

    def snapshot(self) -> dict:
        """Return current settings as a dict for persistence."""
        return {
            "whisper_model": self.model_var.get(),
            "recursive": self.recursive_var.get(),
            "quality_mode": self.quality_var.get(),
            "replace_mode": self.replace_var.get(),
            "replace_before_date": self.date_var.get(),
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
