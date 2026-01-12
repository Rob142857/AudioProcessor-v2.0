"""Speech to Text Transcription Tool v1.0.0-beta.1

A clean, professional GUI for converting audio and video files to text using AI.
- Large AI model for professional quality
- Auto device detection (CUDA > DirectML > CPU)
- No VAD segmentation for continuous speech
- Maximum CPU threads based on RAM
- Clean, modern interface
"""
import argparse
import os
import threading
import queue
import sys
import subprocess
import json
from typing import Optional, List
import gc

# GUI toolkit
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
TKINTER_AVAILABLE = True

# Progress window for live transcription display
try:
    from progress_window import init_progress_window, show_progress_window, hide_progress_window, get_progress_window
    PROGRESS_WINDOW_AVAILABLE = True
except ImportError:
    PROGRESS_WINDOW_AVAILABLE = False
    def init_progress_window(root): return None
    def show_progress_window(): return None
    def hide_progress_window(): pass
    def get_progress_window(): return None

# Supported file extensions for batch processing
SUPPORTED_EXTS = (
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma",
    ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"
)

# Default Downloads folder retained for compatibility with CLI --outdir override
DEFAULT_DOWNLOADS = os.path.normpath(os.path.join(os.path.expanduser("~"), "Downloads"))

# Repo root (folder containing this script)
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(REPO_ROOT, ".transcribe_settings.json")

# Global stop flag for transcription cancellation
STOP_FLAG = threading.Event()

def _load_settings() -> dict:
    try:
        with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _save_settings(data: dict) -> None:
    try:
        with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(data or {}, f, indent=2)
    except Exception:
        pass

def _load_project_settings(folder_path: str) -> dict:
    """Load settings specific to a project folder."""
    all_settings = _load_settings()
    return all_settings.get('projects', {}).get(folder_path, {})

def _save_project_settings(folder_path: str, proj_settings: dict) -> None:
    """Save settings specific to a project folder."""
    all_settings = _load_settings()
    if 'projects' not in all_settings:
        all_settings['projects'] = {}
    all_settings['projects'][folder_path] = proj_settings
    _save_settings(all_settings)

def _repo_default_terms_file() -> Optional[str]:
    # Default to special_words.txt in repo root
    default_path = os.path.join(REPO_ROOT, "special_words.txt")
    if os.path.exists(default_path):
        return default_path
    # Fall back to special_words.md if .txt doesn't exist
    for name in ("special_words.md",):
        p = os.path.join(REPO_ROOT, name)
        if os.path.exists(p):
            return p
    return None

def run_transcription(input_file: str, outdir: Optional[str], output_queue: queue.Queue, *, threads_override: Optional[int] = None):
    """Run transcription for a single file with large model and auto device detection.

    Output directory defaults to the source file directory if not provided.
    """
    # Check stop flag before processing
    if STOP_FLAG.is_set():
        output_queue.put("Transcription cancelled by user.\n")
        return
    
    try:
        # Memory cleanup BEFORE processing each file
        print("🧹 Performing memory cleanup before file processing...")
        try:
            import gc
            import psutil
            import torch

            # Force garbage collection (run twice for cyclic references)
            gc.collect()
            collected = gc.collect()
            print(f"   Garbage collected: {collected} objects")

            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations complete
                print("   GPU cache cleared")

            # NOTE: We do NOT delete cached modules here - that's dangerous
            # and can cause hangs or corrupted state. Let Python manage modules.

            # Monitor memory usage
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            print(f"   Memory usage before processing: {memory_mb:.1f} MB")

            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    print(f"   GPU memory before processing: {gpu_memory:.1f} GB")
                except:
                    pass

        except ImportError:
            print("   Memory cleanup skipped (missing dependencies)")
        except Exception as e:
            print(f"   Memory cleanup warning: {e}")
        # Import and run the transcription
        from transcribe_optimised import transcribe_file_simple_auto

        # Determine output dir (default to same folder as source file)
        target_outdir = outdir if outdir else os.path.dirname(input_file)

        # Get model name from environment variable (set by GUI)
        model_name = os.environ.get("TRANSCRIBE_MODEL_NAME", "large-v3")
        
        # Log current settings for debugging batch issues
        punct_setting = os.environ.get("TRANSCRIBE_RESTORE_PUNCTUATION", "not set")
        print(f"📋 Settings for this file:")
        print(f"   Model: {model_name}")
        print(f"   Punctuation restoration: {punct_setting}")
        print(f"   Quality mode: {os.environ.get('TRANSCRIBE_QUALITY_MODE', 'not set')}")

        out_txt = transcribe_file_simple_auto(
            input_file,
            output_dir=target_outdir,
            threads_override=threads_override,
        )

        if out_txt and os.path.exists(out_txt):
            output_queue.put("Transcription complete! DOCX file saved next to source file.\n")
            output_queue.put("  Output: {}\n".format(out_txt))
        else:
            output_queue.put("Transcription failed or no output generated.\n")

    except Exception as e:
        output_queue.put("Transcription failed: {}\n".format(e))
        import traceback
        output_queue.put("Error details: {}\n".format(traceback.format_exc()))

    finally:
        # Memory cleanup AFTER processing each file
        print("🧹 Performing memory cleanup after file processing...")
        try:
            import gc
            import psutil
            import torch

            # Force garbage collection (run twice for cyclic references)
            gc.collect()
            collected = gc.collect()
            print(f"   Post-processing garbage collected: {collected} objects")

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("   GPU cache cleared after processing")

                # Check if GPU memory is still allocated (informational only)
                try:
                    allocated_gb = torch.cuda.memory_allocated() / 1024**3
                    if allocated_gb > 0.1:  # More than 100MB still allocated
                        print(f"   Note: {allocated_gb:.2f} GB GPU memory still allocated (normal for cached model)")
                except:
                    pass

            # NOTE: We do NOT delete cached modules - this is dangerous and can
            # cause hangs, corrupted state, or require expensive re-initialization.
            # Python's garbage collector handles unused objects automatically.

            # Monitor final memory usage
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            print(f"   Memory usage after processing: {memory_mb:.1f} MB")

            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    print(f"   GPU memory after processing: {gpu_memory:.1f} GB")
                except:
                    pass

        except Exception as e:
            print(f"   Post-processing cleanup warning: {e}")


def run_batch_transcription(paths: List[str], outdir_override: Optional[str], output_queue: queue.Queue, *, threads_override: Optional[int] = None):
    """Run batch transcription sequentially over provided file paths.

    Each file is saved next to its source (unless outdir_override is provided).
    Continues processing even if individual files fail.
    """
    total = len(paths)
    output_queue.put("Batch mode: {} eligible files queued.\n".format(total))
    successful = 0
    failed = 0

    # Update progress window for batch
    progress_win = get_progress_window() if PROGRESS_WINDOW_AVAILABLE else None
    if progress_win:
        progress_win.set_file_progress(0, total)

    for idx, p in enumerate(paths, start=1):
        # Check stop flag before each file
        if STOP_FLAG.is_set():
            output_queue.put("\nBatch processing cancelled by user.\n")
            output_queue.put("Successfully processed: {} files\n".format(successful))
            if failed > 0:
                output_queue.put("Failed: {} files\n".format(failed))
            return
        
        output_queue.put("\n[{} / {}] Processing: {}\n".format(idx, total, os.path.basename(p)))
        
        # Update progress window
        if progress_win:
            progress_win.set_file(p)
            progress_win.set_file_progress(idx, total)
            progress_win.set_status("Transcribing...")
            progress_win.set_progress(0)
        
        try:
            run_transcription(p, outdir_override, output_queue, threads_override=threads_override)
            successful += 1
            if progress_win:
                progress_win.set_status("Complete")
                progress_win.set_progress(100)
        except Exception as e:
            failed += 1
            output_queue.put("Failed to process '{}': {}\n".format(os.path.basename(p), str(e)))
            output_queue.put("Continuing with next file...\n")
            if progress_win:
                progress_win.set_status("Failed")

    output_queue.put("\nBatch processing complete!\n")
    output_queue.put("Successfully processed: {} files\n".format(successful))
    if failed > 0:
        output_queue.put("Failed: {} files\n".format(failed))


class QueueWriter:
    """A file-like writer that puts text into a queue."""
    def __init__(self, q: queue.Queue):
        self.q = q
        self.output_queue = q  # Make queue accessible for progress updates

    def write(self, msg):
        if msg:
            # Don't use print() here as it causes infinite recursion with redirected stdout
            self.q.put(str(msg))

    def flush(self):
        pass


def launch_gui(default_outdir: Optional[str] = None, *, default_threads: Optional[int] = None):
    if not TKINTER_AVAILABLE:
        print("Tkinter is not available in this Python build.")
        return

    assert tk is not None and ttk is not None and filedialog is not None and messagebox is not None

    # Memory cleanup summary for the log
    memory_cleanup_info = []
    try:
        import gc, psutil
        collected = gc.collect()
        memory_cleanup_info.append(f"Garbage collector freed {collected} objects")
        mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_cleanup_info.append(f"Current memory usage: {mem_mb:.1f} MB")
        # Clear whisper/transformers modules
        to_clear = [m for m in list(sys.modules.keys()) if m.startswith(('whisper','transformers'))]
        for m in to_clear:
            sys.modules.pop(m, None)
        gc.collect()
        memory_cleanup_info.append("Module caches cleared (whisper/transformers)")
    except Exception as e:
        memory_cleanup_info.append(f"Memory cleanup warning: {e}")

    try:
        root = tk.Tk()
        root.title("Speech to Text Transcription Tool v1.0Beta")
        # Open large enough to show the Start button; also allow maximizing
        root.geometry("1100x800")
        root.minsize(960, 720)
        root.configure(bg='#f8f9fa')
        # Make the root grid resizable so children can expand
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)
        # Maximize by default on Windows to ensure button visibility (ignore if not supported)
        try:
            root.state('zoomed')
        except Exception:
            pass

        # Initialize progress window (must be done from main GUI thread)
        if PROGRESS_WINDOW_AVAILABLE:
            try:
                init_progress_window(root)
            except Exception as e:
                print(f"Note: Progress window init failed: {e}")

        style = ttk.Style()
        style.configure('Clean.TFrame', background='#f8f9fa')
        style.configure('Clean.TLabel', background='#f8f9fa', foreground='#2c3e50', font=('Segoe UI', 11))
        style.configure('Title.TLabel', font=('Segoe UI', 20, 'bold'), foreground='#1a365d', background='#f8f9fa')
        style.configure('Subtitle.TLabel', font=('Segoe UI', 10), foreground='#64748b', background='#f8f9fa')
        style.configure('Section.TLabel', font=('Segoe UI', 13, 'bold'), foreground='#374151', background='#f8f9fa')

        mainframe = ttk.Frame(root, style='Clean.TFrame', padding="30 30 30 30")
        mainframe.grid(column=0, row=0, sticky="nsew")
        # Allow horizontal expansion across all columns
        mainframe.columnconfigure(0, weight=1)
        mainframe.columnconfigure(1, weight=1)
        mainframe.columnconfigure(2, weight=1)
        mainframe.rowconfigure(11, weight=1)
        mainframe.rowconfigure(12, minsize=60)

        # Title
        title_frame = ttk.Frame(mainframe, style='Clean.TFrame')
        title_frame.grid(column=0, row=0, columnspan=3, pady=(0, 30), sticky="ew")
        ttk.Label(title_frame, text="Speech to Text Transcription Tool", style='Title.TLabel').grid(column=0, row=0, sticky="w")
        ttk.Label(title_frame, text="v1.0Beta - Professional AI-powered transcription", style='Subtitle.TLabel').grid(column=0, row=1, sticky="w", pady=(8, 0))

        # Input & Settings (combined)
        ttk.Label(mainframe, text="Input & Settings", style='Section.TLabel').grid(column=0, row=1, columnspan=3, sticky="w", pady=(0, 15))
        combined_frame = tk.Frame(mainframe, bg='white', relief='flat', borderwidth=1)
        combined_frame.grid(column=0, row=2, columnspan=3, sticky="ew", pady=(0, 25))
        # Column 1 expands; buttons stay compact
        combined_frame.columnconfigure(1, weight=1)

        # Row 0: File/Folder selection line
        ttk.Label(combined_frame, text="Audio/Video File or Folder:", background='white', foreground='#374151', font=('Segoe UI', 10, 'bold')).grid(column=0, row=0, sticky="w", padx=20, pady=(18, 6))
        input_var = tk.StringVar()
        tk.Entry(combined_frame, textvariable=input_var, font=('Segoe UI', 10), relief='flat', borderwidth=1, bg='#f9fafb', fg='#111827', insertbackground='#111827').grid(column=1, row=0, sticky="ew", padx=(12, 6), pady=(18, 6))

        # Define handlers before creating buttons
        def browse_input():
            types = [("Audio files", "*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.wma"),("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),("All files", "*.*")]
            p = filedialog.askopenfilename(title="Select audio/video file", filetypes=types)
            if p:
                input_var.set(p)
                status_label.config(text=f"Selected file: {os.path.basename(p)}", foreground='#059669')

        def browse_folder():
            d = filedialog.askdirectory(title="Select folder for batch processing")
            if d:
                input_var.set(d)
                try:
                    files = [f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
                    status_label.config(text=f"Selected folder: {os.path.basename(d)} ({len(files)} eligible files)", foreground='#059669')
                    # Reload project settings for the new folder
                    proj_settings = _load_project_settings(d)
                    domain_var.set(proj_settings.get("domain_terms_file", _repo_default_terms_file() or ""))
                    recursive_var.set(proj_settings.get("recursive", 0))
                    replace_before_enabled_var.set(proj_settings.get("replace_before_enabled", 0))
                    date_var.set(proj_settings.get("replace_before_date", datetime.datetime.now().strftime("%Y-%m-%d")))
                    time_var.set(proj_settings.get("replace_before_time", "00:00"))
                    time_header_var.set(proj_settings.get("time_header", 1))
                    quality_mode_var.set(proj_settings.get("quality_mode", 1))
                    max_repeat_var.set(proj_settings.get("max_repeat_cap", 10))
                except Exception:
                    status_label.config(text=f"Selected folder: {os.path.basename(d)}", foreground='#059669')

        tk.Button(combined_frame, text="Browse File", command=browse_input, font=('Segoe UI', 10, 'bold'), bg='#007acc', fg='white', relief='flat', borderwidth=0, padx=16, pady=6, activebackground='#0056b3', activeforeground='white').grid(column=2, row=0, sticky="e", padx=(6, 6), pady=(18, 6))
        tk.Button(combined_frame, text="Browse Folder", command=browse_folder, font=('Segoe UI', 10, 'bold'), bg='#0ea5e9', fg='white', relief='flat', borderwidth=0, padx=16, pady=6, activebackground='#0284c7', activeforeground='white').grid(column=3, row=0, sticky="e", padx=(6, 20), pady=(18, 6))

        # Row 1: Status line
        status_label = ttk.Label(combined_frame, text="No file or folder selected", font=('Segoe UI', 9), foreground='#6b7280', background='white')
        status_label.grid(column=0, row=1, columnspan=4, sticky="w", padx=20, pady=(0, 8))

        # Thin separator
        ttk.Separator(combined_frame, orient='horizontal').grid(column=0, row=2, columnspan=4, sticky='ew', padx=20, pady=(4, 10))

        # Row 3: Threads override
        tk.Label(combined_frame, text="CPU Threads (optional):", bg='white', fg='#374151', font=('Segoe UI', 10)).grid(column=0, row=3, sticky='w', padx=20, pady=(4, 6))
        threads_var = tk.StringVar(value=str(default_threads) if default_threads and default_threads > 0 else "")
        tk.Entry(combined_frame, textvariable=threads_var, width=10, bg='#f9fafb', fg='#111827', relief='flat').grid(column=1, row=3, sticky='w', padx=(12, 6), pady=(4, 6))
        tk.Label(combined_frame, text="Leave blank for Auto. Or set TRANSCRIBE_THREADS.", bg='white', fg='#6b7280', font=('Segoe UI', 8)).grid(column=2, row=3, columnspan=2, sticky='w', padx=(6, 20), pady=(4, 6))

        # Row 4: Domain terms file (awkward words)
        tk.Label(combined_frame, text="Domain terms file (optional):", bg='white', fg='#374151', font=('Segoe UI', 10)).grid(column=0, row=4, sticky='w', padx=20, pady=(0, 6))
        # Determine default domain file: from project settings or repo-root fallback
        current_folder = os.path.dirname(input_var.get()) if input_var.get() else REPO_ROOT
        proj_settings = _load_project_settings(current_folder)
        _saved_domain = proj_settings.get("domain_terms_file")
        _default_domain = _saved_domain if (_saved_domain and os.path.exists(_saved_domain)) else (_repo_default_terms_file() or "")
        domain_var = tk.StringVar(value=_default_domain)
        tk.Entry(combined_frame, textvariable=domain_var, bg='#f9fafb', fg='#111827', relief='flat').grid(column=1, row=4, sticky='ew', padx=(12, 6), pady=(0, 6))

        # Controls (Browse, Clear, Open sample) grouped together
        controls_frame = tk.Frame(combined_frame, bg='white')
        controls_frame.grid(column=2, row=4, columnspan=2, sticky='w', padx=(6, 20), pady=(0, 6))

        def browse_domain_file():
            p = filedialog.askopenfilename(title="Select domain terms file (one term per line)", filetypes=[("Text/Markdown", "*.txt *.md"), ("All Files", "*.*")])
            if p:
                domain_var.set(p)
                # Persist selection per project
                current_folder = os.path.dirname(input_var.get()) if input_var.get() else REPO_ROOT
                proj_settings = _load_project_settings(current_folder)
                proj_settings["domain_terms_file"] = p
                _save_project_settings(current_folder, proj_settings)
        tk.Button(controls_frame, text="Browse", command=browse_domain_file, font=('Segoe UI', 9, 'bold'), bg='#10b981', fg='white', relief='flat', borderwidth=0, padx=12, pady=4, activebackground='#059669', activeforeground='white').pack(side='left', padx=(0, 6))

        def clear_domain_file():
            domain_var.set("")
            # Persist clear per project
            current_folder = os.path.dirname(input_var.get()) if input_var.get() else REPO_ROOT
            proj_settings = _load_project_settings(current_folder)
            proj_settings["domain_terms_file"] = ""
            _save_project_settings(current_folder, proj_settings)
        tk.Button(controls_frame, text="Clear", command=clear_domain_file, font=('Segoe UI', 9), bg='#e5e7eb', fg='#111827', relief='flat', borderwidth=0, padx=12, pady=4, activebackground='#d1d5db', activeforeground='#111827').pack(side='left', padx=(0, 6))

        def open_sample_terms():
            try:
                sample = _repo_default_terms_file()
                if not sample:
                    # Create default sample if missing
                    sample = os.path.join(REPO_ROOT, 'awkward_words.txt')
                    content = (
                        "# Example domain terms (one per line). Lines starting with # are comments.\n"
                        "# You can also use a Markdown list like:\n"
                        "# - Schrödinger\n# - Noether\n# - Fourier transform\n\n"
                        "Schrödinger\nNoether\nFourier transform\n"
                    )
                    with open(sample, 'w', encoding='utf-8') as f:
                        f.write(content)
                # Launch in Notepad
                try:
                    subprocess.Popen(["notepad.exe", sample])
                except Exception:
                    # Fallback to OS default
                    os.startfile(sample)
            except Exception as e:
                messagebox.showerror("Open sample terms", f"Failed to open sample terms file: {e}")
        tk.Button(controls_frame, text="Open sample", command=open_sample_terms, font=('Segoe UI', 9), bg='#f3f4f6', fg='#111827', relief='flat', borderwidth=0, padx=12, pady=4, activebackground='#e5e7eb', activeforeground='#111827').pack(side='left')

        # Row 5: Max performance mode
        max_perf_var = tk.IntVar(value=1)
        tk.Checkbutton(combined_frame, text="Maximise performance (use all cores, high priority, optimised VRAM)", variable=max_perf_var, bg='white', fg='#374151', selectcolor='white', activebackground='white').grid(column=0, row=5, columnspan=4, sticky='w', padx=20, pady=(0, 8))

        # Row 6: Batch options
        recursive_var = tk.IntVar(value=proj_settings.get("recursive", 0))
        tk.Checkbutton(combined_frame, text="Process subfolders (recursive)", variable=recursive_var, bg='white', fg='#374151', selectcolor='white', activebackground='white').grid(column=0, row=6, columnspan=2, sticky='w', padx=20, pady=(0, 8))
        
        # Replace output files before date/time (for resuming incomplete batches)
        tk.Label(combined_frame, text="Replace outputs before:", bg='white', fg='#374151', font=('Segoe UI', 10)).grid(column=2, row=6, sticky='w', padx=20, pady=(0, 8))
        
        replace_before_enabled_var = tk.IntVar(value=proj_settings.get("replace_before_enabled", 0))
        tk.Checkbutton(combined_frame, text="Enable", variable=replace_before_enabled_var, bg='white', fg='#374151', selectcolor='white', activebackground='white').grid(column=3, row=6, sticky='w', padx=(0, 20), pady=(0, 8))
        
        # Row 6b: Date/time picker
        datetime_frame = tk.Frame(combined_frame, bg='white')
        datetime_frame.grid(column=2, row=7, columnspan=2, sticky='w', padx=20, pady=(0, 8))
        
        import datetime
        default_date = proj_settings.get("replace_before_date", datetime.datetime.now().strftime("%Y-%m-%d"))
        default_time = proj_settings.get("replace_before_time", "00:00")
        
        tk.Label(datetime_frame, text="Date:", bg='white', fg='#6b7280', font=('Segoe UI', 9)).pack(side='left', padx=(0, 4))
        date_var = tk.StringVar(value=default_date)
        tk.Entry(datetime_frame, textvariable=date_var, width=12, bg='#f9fafb', fg='#111827', relief='flat').pack(side='left', padx=(0, 12))
        
        tk.Label(datetime_frame, text="Time:", bg='white', fg='#6b7280', font=('Segoe UI', 9)).pack(side='left', padx=(0, 4))
        time_var = tk.StringVar(value=default_time)
        tk.Entry(datetime_frame, textvariable=time_var, width=8, bg='#f9fafb', fg='#111827', relief='flat').pack(side='left')
        
        tk.Label(datetime_frame, text="(YYYY-MM-DD HH:MM)", bg='white', fg='#9ca3af', font=('Segoe UI', 8)).pack(side='left', padx=(8, 0))

        # Row 8: Output options
        time_header_var = tk.IntVar(value=proj_settings.get("time_header", 1))
        tk.Checkbutton(combined_frame, text="Show 'Transcription time' in DOCX header", variable=time_header_var, bg='white', fg='#374151', selectcolor='white', activebackground='white').grid(column=0, row=8, columnspan=4, sticky='w', padx=20, pady=(0, 8))

        # Thin separator before quality options
        ttk.Separator(combined_frame, orient='horizontal').grid(column=0, row=9, columnspan=4, sticky='ew', padx=20, pady=(4, 10))

        # Row 10: Quality options
        quality_mode_var = tk.IntVar(value=proj_settings.get("quality_mode", 1))  # Default to ON
        tk.Checkbutton(combined_frame, text="Quality mode (beam=10, better punctuation, slower)", variable=quality_mode_var, bg='white', fg='#374151', selectcolor='white', activebackground='white').grid(column=0, row=10, columnspan=2, sticky='w', padx=20, pady=(0, 6))

        # Row 10 (right side): Processing path toggle
        # 0 = Simple path (default, lets Whisper handle segmentation internally)
        # 1 = Dataset path (pre-segments audio, better GPU pipeline but can affect quality)
        use_dataset_var = tk.IntVar(value=proj_settings.get("use_dataset_path", 0))  # Default to Simple
        tk.Checkbutton(combined_frame, text="Use dataset pipeline (experimental, pre-segments audio)", variable=use_dataset_var, bg='white', fg='#374151', selectcolor='white', activebackground='white').grid(column=2, row=10, columnspan=2, sticky='w', padx=20, pady=(0, 6))

        # Row 11: Model selection - expanded with multiple backends
        tk.Label(combined_frame, text="Transcription Model:", bg='white', fg='#374151', font=('Segoe UI', 10, 'bold')).grid(column=0, row=11, sticky='w', padx=20, pady=(8, 6))
        
        # Available models with descriptions
        # All "faster-whisper-" prefixed models use CTranslate2 backend (GPU accelerated)
        MODEL_OPTIONS = [
            ("faster-whisper-large-v3", "Faster-Whisper large-v3 (GPU, best accuracy)"),
            ("faster-whisper-large-v2", "Faster-Whisper large-v2 (GPU, try if v3 punctuation poor)"),
            ("faster-whisper-distil-large-v3", "Faster-Whisper distil-v3 (GPU, 6x faster, English)"),
            ("faster-whisper-distil-large-v2", "Faster-Whisper distil-v2 (GPU, fast, English)"),
            ("faster-whisper-large-v3-turbo", "Faster-Whisper turbo (GPU, fastest)"),
            ("large-v3", "Native Whisper large-v3 (may use CPU)"),
            ("large-v2", "Native Whisper large-v2 (may use CPU)"),
        ]
        
        # Default to faster-whisper-large-v3 for best balance of speed and accuracy
        model_choice_var = tk.StringVar(value=proj_settings.get("whisper_model", "faster-whisper-large-v3"))
        
        model_frame = tk.Frame(combined_frame, bg='white')
        model_frame.grid(column=1, row=11, columnspan=3, sticky='w', padx=(12, 20), pady=(8, 6))
        
        # Create dropdown/combobox for model selection
        model_values = [opt[1] for opt in MODEL_OPTIONS]
        model_keys = [opt[0] for opt in MODEL_OPTIONS]
        
        def get_model_display(key):
            for k, v in MODEL_OPTIONS:
                if k == key:
                    return v
            return MODEL_OPTIONS[0][1]
        
        def get_model_key(display):
            for k, v in MODEL_OPTIONS:
                if v == display:
                    return k
            return MODEL_OPTIONS[0][0]
        
        model_display_var = tk.StringVar(value=get_model_display(model_choice_var.get()))
        
        model_combo = ttk.Combobox(
            model_frame,
            textvariable=model_display_var,
            values=model_values,
            state='readonly',
            width=50,
            font=('Segoe UI', 9)
        )
        model_combo.pack(side='left', padx=(0, 10))
        
        def on_model_change(event=None):
            model_choice_var.set(get_model_key(model_display_var.get()))
        
        model_combo.bind('<<ComboboxSelected>>', on_model_change)

        # Row 13: Initial prompt for context (helps with specialized vocabulary)
        tk.Label(combined_frame, text="Context hint (optional):", bg='white', fg='#374151', font=('Segoe UI', 10)).grid(column=0, row=13, sticky='w', padx=20, pady=(8, 6))
        initial_prompt_var = tk.StringVar(value=proj_settings.get("initial_prompt", ""))
        initial_prompt_entry = tk.Entry(combined_frame, textvariable=initial_prompt_var, font=('Segoe UI', 9), width=70, bg='#f9fafb', fg='#111827', relief='flat')
        initial_prompt_entry.grid(column=1, row=13, columnspan=3, sticky='w', padx=(12, 20), pady=(8, 6))
        
        # Tooltip/hint for initial prompt
        prompt_hint = tk.Label(combined_frame, text="💡 E.g. 'Lecture on esoteric philosophy, Gurdjieff Work, Fourth Way teachings'", bg='white', fg='#6b7280', font=('Segoe UI', 8))
        prompt_hint.grid(column=1, row=14, columnspan=3, sticky='w', padx=(12, 20), pady=(0, 6))

        # Row 15: Compact description
        desc = (
            "Auto device (CUDA/DirectML/CPU) • Faster-Whisper for speed\n"
            "Outputs saved next to source file(s)."
        )
        ttk.Label(combined_frame, text=desc, background='white', foreground='#374151', font=('Segoe UI', 9), wraplength=920, justify='left').grid(column=0, row=15, columnspan=4, sticky='w', padx=20, pady=(8, 16))

        # Handlers attached to the buttons above

        # Log
        ttk.Label(mainframe, text="Activity Log", style='Section.TLabel').grid(column=0, row=10, columnspan=3, sticky="w", pady=(0, 15))
        log_frame = tk.Frame(mainframe, bg='white', relief='flat', borderwidth=1)
        log_frame.grid(column=0, row=11, columnspan=3, sticky="nsew", pady=(0, 25))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        log_text = tk.Text(log_frame, wrap=tk.WORD, height=25, font=('Segoe UI', 9), bg='white', fg='#2c3e50', borderwidth=0, highlightthickness=0, insertbackground='#2c3e50')
        log_text.grid(column=0, row=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=log_text.yview)
        log_scroll.grid(column=1, row=0, sticky="ns")
        log_text['yscrollcommand'] = log_scroll.set
        log_text.configure(state='normal')
        log_text.insert('end', "Welcome to Speech to Text Transcription Tool v1.0Beta!\n", "title")
        log_text.insert('end', "Select an audio or video file and click 'Start Transcription' to begin.\n\n")
        if memory_cleanup_info:
            log_text.insert('end', "System Memory Cleanup Results:\n", "info")
            for info in memory_cleanup_info:
                log_text.insert('end', f"  {info}\n")
            log_text.insert('end', "\n")
        log_text.configure(state='disabled')
        log_text.tag_configure("title", foreground="#007acc", font=('Segoe UI', 9, 'bold'))
        log_text.tag_configure("info", foreground="#60a5fa")

        q = queue.Queue()
        last_progress_line = [""]  # Track last progress line for in-place updates
        
        # Track audio duration for progress estimation
        # Based on benchmarks: large-v3-turbo processes ~24x faster than real-time on GTX 1070 Ti
        audio_duration_seconds = [0]  # Mutable container for nested function access
        TRANSCRIPTION_SPEED_RATIO = 24  # Audio seconds processed per wall-clock second

        def format_log_message(msg: str) -> str:
            """Format log message, handling progress bar updates."""
            return msg

        def is_progress_line(msg: str) -> bool:
            """Check if message is a progress bar update (contains progress indicators)."""
            # Detect tqdm-style progress bars and our custom progress bars
            progress_indicators = ['%|', '█', '░', '/s]', 'iB/s', 'frames/s', 'it/s']
            return any(ind in msg for ind in progress_indicators)
        
        def parse_duration_to_seconds(duration_str: str) -> float:
            """Parse duration string like '01:54:26.64' or '06:45.19' to seconds."""
            try:
                import re
                # Match HH:MM:SS.ms or MM:SS.ms
                match = re.search(r'(\d+):(\d+):(\d+(?:\.\d+)?)', duration_str)
                if match:
                    hours, mins, secs = match.groups()
                    return int(hours) * 3600 + int(mins) * 60 + float(secs)
                match = re.search(r'(\d+):(\d+(?:\.\d+)?)', duration_str)
                if match:
                    mins, secs = match.groups()
                    return int(mins) * 60 + float(secs)
            except:
                pass
            return 0
        
        def parse_elapsed_to_seconds(elapsed_str: str) -> float:
            """Parse elapsed time like '2:40' or '30s' to seconds."""
            try:
                import re
                # Match M:SS format
                match = re.search(r'(\d+):(\d+)', elapsed_str)
                if match:
                    mins, secs = match.groups()
                    return int(mins) * 60 + int(secs)
                # Match XXs format
                match = re.search(r'(\d+)s', elapsed_str)
                if match:
                    return int(match.group(1))
            except:
                pass
            return 0

        def poll_queue():
            try:
                while True:
                    msg = q.get_nowait()
                    log_text.configure(state='normal')
                    
                    # Handle carriage return (in-place update) messages
                    if msg.startswith('\r'):
                        msg = msg.lstrip('\r')
                        # If it's a progress update, replace the last line
                        if is_progress_line(msg) or is_progress_line(last_progress_line[0]):
                            # Delete the last line if it was a progress line
                            if last_progress_line[0]:
                                try:
                                    log_text.delete("end-2l", "end-1l")
                                except:
                                    pass
                    
                    # Update progress window with transcribed text
                    # Transcribed text lines start with spaces OR contain transcribed content
                    if PROGRESS_WINDOW_AVAILABLE:
                        pw = get_progress_window()
                        if pw:
                            # Capture audio duration from log (e.g., "⏱️  Duration: 01:54:26.64")
                            if '⏱️' in msg and 'Duration:' in msg:
                                dur_secs = parse_duration_to_seconds(msg)
                                if dur_secs > 0:
                                    audio_duration_seconds[0] = dur_secs
                            
                            # Estimate progress from elapsed time (e.g., "⏳ Transcribing: 2:40 | GPU:")
                            if '⏳ Transcribing:' in msg and audio_duration_seconds[0] > 0:
                                try:
                                    import re
                                    # Extract elapsed time
                                    match = re.search(r'Transcribing:\s*([0-9:]+s?)', msg)
                                    if match:
                                        elapsed_secs = parse_elapsed_to_seconds(match.group(1))
                                        if elapsed_secs > 0:
                                            # Estimate: elapsed * speed_ratio = audio processed
                                            estimated_audio_processed = elapsed_secs * TRANSCRIPTION_SPEED_RATIO
                                            percent = min(95, (estimated_audio_processed / audio_duration_seconds[0]) * 100)
                                            pw.set_progress(percent)
                                            pw.set_status(f"Transcribing... {int(percent)}%")
                                except:
                                    pass
                            
                            # Detect transcribed text (starts with spaces, or is plain text without emoji)
                            stripped = msg.strip()
                            if (msg.startswith('   ') and len(stripped) > 5) or \
                               (len(stripped) > 20 and not any(c in msg for c in '📊✅❌⚠️🔄🧹💾📄🎵⏱⏳')) :
                                try:
                                    pw.set_text(stripped[:80])
                                except:
                                    pass
                            
                            # Update progress from segment messages (multiple patterns)
                            if 'segments' in msg.lower():
                                try:
                                    import re
                                    # Pattern 1: "Completed X/Y segments" or "kept X/Y segments"
                                    match = re.search(r'(\d+)/(\d+)\s*segments', msg)
                                    if match:
                                        completed = int(match.group(1))
                                        total = int(match.group(2))
                                        percent = (completed / total * 100) if total > 0 else 0
                                        pw.set_progress(percent)
                                except:
                                    pass
                            
                            # Reset progress for new file (detect "[X / Y] Processing:")
                            if '] Processing:' in msg:
                                audio_duration_seconds[0] = 0
                                pw.set_progress(0)
                            
                            # Set 100% on completion
                            if 'TRANSCRIPTION COMPLETE' in msg or '🎉' in msg:
                                pw.set_progress(100)
                                pw.set_status("Complete!")
                    
                    # Skip noisy tqdm intermediate updates (keep only significant ones)
                    if is_progress_line(msg):
                        # Only show progress at 0%, 25%, 50%, 75%, 100% or if it's our clean format
                        if '█' in msg and '░' in msg:
                            # Our clean progress bar - always show
                            pass
                        elif any(p in msg for p in ['0%|', '25%|', '50%|', '75%|', '100%|', '100%']):
                            # Milestone progress - show
                            pass
                        elif msg.strip().endswith(('Complete', 'completed', '✓', 'done')):
                            # Completion message - show
                            pass
                        else:
                            # Skip intermediate tqdm updates
                            last_progress_line[0] = msg
                            log_text.configure(state='disabled')
                            continue
                    
                    # Filter out pure whitespace and redundant newlines
                    if msg.strip() or msg == '\n':
                        log_text.insert('end', format_log_message(msg))
                        last_progress_line[0] = msg if is_progress_line(msg) else ""
                    
                    log_text.see('end')
                    log_text.configure(state='disabled')
            except queue.Empty:
                pass
            root.after(100, poll_queue)  # Faster polling for smoother progress

        def start_transcription_thread():
            inp = input_var.get().strip()
            if not inp or not os.path.exists(inp):
                messagebox.showerror("Input Required", f"Please select a valid file or folder.\nPath: '{inp}'\nExists: {os.path.exists(inp) if inp else False}")
                return

            log_text.configure(state='normal')
            log_text.delete(1.0, 'end')
            log_text.insert('end', "Starting transcription...\n")
            log_text.configure(state='disabled')
            run_btn.configure(state='disabled')
            stop_btn.configure(state='normal')
            STOP_FLAG.clear()  # Reset stop flag

            def worker():
                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout = QueueWriter(q)
                sys.stderr = QueueWriter(q)
                try:
                    def _parse_int_opt(s: str) -> Optional[int]:
                        try:
                            v = int(s.strip())
                            return v if v > 0 else None
                        except Exception:
                            return None

                    thr = _parse_int_opt(threads_var.get())

                    # Apply max performance env for the worker
                    try:
                        if max_perf_var.get() == 1:
                            os.environ["TRANSCRIBE_MAX_PERF"] = "1"
                        else:
                            os.environ.pop("TRANSCRIBE_MAX_PERF", None)
                    except Exception:
                        pass

                    # Apply domain terms file (if provided)
                    try:
                        dom_path = domain_var.get().strip()
                        if dom_path:
                            os.environ["TRANSCRIBE_AWKWARD_FILE"] = dom_path
                            q.put(f"Using domain terms file: {os.path.basename(dom_path)}\n")
                        else:
                            os.environ.pop("TRANSCRIBE_AWKWARD_FILE", None)
                        # Persist current selection per project
                        current_folder = os.path.dirname(inp) if os.path.isfile(inp) else inp
                        proj_settings = _load_project_settings(current_folder)
                        proj_settings["domain_terms_file"] = dom_path
                        proj_settings["recursive"] = recursive_var.get()
                        proj_settings["replace_before_enabled"] = replace_before_enabled_var.get()
                        proj_settings["replace_before_date"] = date_var.get()
                        proj_settings["replace_before_time"] = time_var.get()
                        proj_settings["time_header"] = time_header_var.get()
                        proj_settings["quality_mode"] = quality_mode_var.get()
                        proj_settings["use_dataset_path"] = use_dataset_var.get()
                        proj_settings["whisper_model"] = model_choice_var.get()
                        proj_settings["initial_prompt"] = initial_prompt_var.get()
                        _save_project_settings(current_folder, proj_settings)
                        
                    except Exception as e:
                        q.put(f"Warning: could not apply domain terms file: {e}\n")

                    # Apply time header preference
                    try:
                        if time_header_var.get() == 1:
                            os.environ.pop("TRANSCRIBE_HIDE_TIME", None)
                        else:
                            os.environ["TRANSCRIBE_HIDE_TIME"] = "1"
                    except Exception:
                        pass

                    # Apply quality mode
                    try:
                        if quality_mode_var.get() == 1:
                            os.environ["TRANSCRIBE_QUALITY_MODE"] = "1"
                        else:
                            os.environ.pop("TRANSCRIBE_QUALITY_MODE", None)
                    except Exception:
                        pass

                    # Apply model selection
                    try:
                        selected_model = model_choice_var.get()
                        os.environ["TRANSCRIBE_MODEL_NAME"] = selected_model

                        # If user picked a native Whisper model explicitly, prefer native backend.
                        # This disables the low-VRAM auto-switch-to-faster-whisper heuristic.
                        is_native_selection = not (
                            selected_model.startswith("faster-whisper-")

                            or selected_model == "insanely-fast-whisper"
                        )
                        if is_native_selection:
                            os.environ["TRANSCRIBE_FORCE_NATIVE_WHISPER"] = "1"
                        else:
                            os.environ.pop("TRANSCRIBE_FORCE_NATIVE_WHISPER", None)
                    except Exception:
                        pass

                    # Apply processing path selection
                    try:
                        if use_dataset_var.get() == 1:
                            os.environ["TRANSCRIBE_USE_DATASET"] = "1"
                        else:
                            os.environ.pop("TRANSCRIBE_USE_DATASET", None)
                    except Exception:
                        pass

                    # Apply initial prompt for context hints
                    # ALWAYS enable special_words.txt usage + optional GUI prompt overlay
                    try:
                        # Enable special_words.txt domain terms by default
                        os.environ["TRANSCRIBE_ALLOW_PROMPT"] = "1"
                        
                        # Add optional GUI context hint if provided
                        prompt = initial_prompt_var.get().strip()
                        if prompt:
                            os.environ["TRANSCRIBE_INITIAL_PROMPT"] = prompt
                        else:
                            os.environ.pop("TRANSCRIBE_INITIAL_PROMPT", None)
                    except Exception:
                        pass

                    # Show progress window
                    progress_win = None
                    if PROGRESS_WINDOW_AVAILABLE:
                        try:
                            progress_win = show_progress_window()
                        except Exception as pw_err:
                            q.put(f"Note: Progress window unavailable: {pw_err}\n")

                    if os.path.isdir(inp):
                        def _is_supported(filename: str) -> bool:
                            return os.path.splitext(filename)[1].lower() in SUPPORTED_EXTS

                        def _should_skip_file(path: str) -> bool:
                            """Check if file should be skipped based on output file modification time."""
                            if replace_before_enabled_var.get() == 0:
                                return False  # Feature disabled, never skip
                            
                            base = os.path.splitext(os.path.basename(path))[0]
                            folder = os.path.dirname(path)
                            docx_path = os.path.join(folder, base + ".docx")
                            
                            # Check if DOCX output exists
                            if not os.path.exists(docx_path):
                                return False  # No output, don't skip
                            
                            # Parse cutoff datetime
                            try:
                                import datetime
                                cutoff_str = f"{date_var.get()} {time_var.get()}"
                                cutoff_dt = datetime.datetime.strptime(cutoff_str, "%Y-%m-%d %H:%M")
                                
                                # Get DOCX modification time
                                docx_mtime = os.path.getmtime(docx_path)
                                docx_dt = datetime.datetime.fromtimestamp(docx_mtime)
                                
                                # Skip if output is NEWER than cutoff (was created after the cutoff)
                                return docx_dt >= cutoff_dt
                                
                            except Exception as e:
                                q.put(f"Warning: Could not parse date/time: {e}\n")
                                return False  # On error, don't skip

                        files = []
                        skipped = 0
                        try:
                            if recursive_var.get() == 1:
                                for dirpath, _dirs, names in os.walk(inp):
                                    for name in sorted(names):
                                        if not _is_supported(name):
                                            continue
                                        full = os.path.join(dirpath, name)
                                        if _should_skip_file(full):
                                            rel = os.path.relpath(full, inp)
                                            q.put(f"Skipping (output newer than cutoff): {rel}\n")
                                            skipped += 1
                                            continue
                                        files.append(full)
                            else:
                                for name in sorted(os.listdir(inp)):
                                    full = os.path.join(inp, name)
                                    if os.path.isfile(full) and _is_supported(name):
                                        if _should_skip_file(full):
                                            q.put(f"Skipping (output newer than cutoff): {name}\n")
                                            skipped += 1
                                            continue
                                        files.append(full)
                        except Exception as e:
                            q.put(f"Failed to list folder '{inp}': {e}\n")

                        if not files:
                            if skipped > 0:
                                q.put("All files skipped due to existing outputs.\n")
                            else:
                                q.put("No supported media files found in the selected folder.\n")
                        else:
                            if skipped > 0:
                                q.put(f"{skipped} file(s) skipped due to existing outputs.\n")
                            run_batch_transcription(files, outdir_override=None, output_queue=q, threads_override=thr)
                    else:
                        # Single file - update progress window
                        if progress_win:
                            progress_win.set_file(inp)
                            progress_win.set_file_progress(1, 1)
                            progress_win.set_status("Transcribing...")
                        run_transcription(inp, outdir=None, output_queue=q, threads_override=thr)
                        if progress_win:
                            progress_win.set_status("Complete")
                            progress_win.set_progress(100)
                    q.put("Transcription process finished!\n")
                except Exception as e:
                    import traceback
                    q.put(f"Worker error: {e}\n{traceback.format_exc()}")
                finally:
                    sys.stdout = old_out
                    sys.stderr = old_err
                    # Hide progress window
                    if PROGRESS_WINDOW_AVAILABLE:
                        try:
                            hide_progress_window()
                        except:
                            pass
                    root.after(0, lambda: run_btn.configure(state='normal'))
                    root.after(0, lambda: stop_btn.configure(state='disabled'))

            threading.Thread(target=worker, daemon=True).start()

        # Button frame for run, stop, and clear cache buttons
        button_frame = tk.Frame(mainframe, bg='white')
        button_frame.grid(column=0, row=12, columnspan=3, pady=(10, 0))
        
        run_btn = tk.Button(button_frame, text="Start Transcription", command=start_transcription_thread, font=('Segoe UI', 12, 'bold'), bg='#007acc', fg='white', relief='flat', borderwidth=0, padx=30, pady=12, activebackground='#0056b3', activeforeground='white', cursor='hand2')
        run_btn.pack(side='left', padx=(0, 10))
        
        def stop_transcription():
            """Stop the running transcription process."""
            STOP_FLAG.set()
            stop_btn.configure(state='disabled')
            log_text.configure(state='normal')
            log_text.insert('end', "\n🛑 Stopping transcription...\n")
            log_text.configure(state='disabled')
            log_text.see('end')
        
        stop_btn = tk.Button(button_frame, text="Stop", command=stop_transcription, font=('Segoe UI', 12, 'bold'), bg='#dc2626', fg='white', relief='flat', borderwidth=0, padx=30, pady=12, activebackground='#b91c1c', activeforeground='white', cursor='hand2', state='disabled')
        stop_btn.pack(side='left', padx=(0, 10))
        
        def clear_cache():
            """Clear GPU, RAM, and model cache."""
            log_text.configure(state='normal')
            log_text.insert('end', "\n🧹 Clearing cache...\n")
            log_text.configure(state='disabled')
            log_text.see('end')
            
            def clear_worker():
                try:
                    # Force garbage collection
                    collected = gc.collect()
                    log_text.configure(state='normal')
                    log_text.insert('end', f"   Garbage collected: {collected} objects\n")
                    log_text.configure(state='disabled')
                    
                    # Clear GPU cache if available
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            log_text.configure(state='normal')
                            log_text.insert('end', "   GPU cache cleared\n")
                            log_text.configure(state='disabled')
                            
                            # Show GPU memory
                            try:
                                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                                log_text.configure(state='normal')
                                log_text.insert('end', f"   GPU memory: {gpu_memory:.2f} GB\n")
                                log_text.configure(state='disabled')
                            except:
                                pass
                    except ImportError:
                        log_text.configure(state='normal')
                        log_text.insert('end', "   PyTorch not available, GPU cache not cleared\n")
                        log_text.configure(state='disabled')
                    
                    # Show RAM usage
                    try:
                        import psutil
                        process = psutil.Process(os.getpid())
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        log_text.configure(state='normal')
                        log_text.insert('end', f"   RAM usage: {memory_mb:.1f} MB\n")
                        log_text.configure(state='disabled')
                    except:
                        pass
                    
                    log_text.configure(state='normal')
                    log_text.insert('end', "✅ Cache cleared successfully\n")
                    log_text.configure(state='disabled')
                    log_text.see('end')
                    
                except Exception as e:
                    log_text.configure(state='normal')
                    log_text.insert('end', f"❌ Error clearing cache: {e}\n")
                    log_text.configure(state='disabled')
                    log_text.see('end')
            
            threading.Thread(target=clear_worker, daemon=True).start()
        
        cache_btn = tk.Button(button_frame, text="Clear Cache", command=clear_cache, font=('Segoe UI', 12, 'bold'), bg='#f59e0b', fg='white', relief='flat', borderwidth=0, padx=30, pady=12, activebackground='#d97706', activeforeground='white', cursor='hand2')
        cache_btn.pack(side='left')
        
        poll_queue()
        root.mainloop()
    except Exception as e:
        import traceback
        error_msg = f"GUI Error: {e}\n{traceback.format_exc()}"
        print(error_msg)
        try:
            if tk and tk.Tk:
                r = tk.Tk(); r.withdraw(); messagebox.showerror("GUI Error", error_msg); r.destroy()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Speech to Text Transcription Tool v1.0Beta")
    parser.add_argument("--input", help="input audio/video file OR folder (if provided, runs headless)")
    parser.add_argument("--outdir", help="optional output folder override; defaults to saving next to source file(s)")
    parser.add_argument("--gui", action="store_true", help="launch GUI mode")
    parser.add_argument("--threads", type=int, help="Override CPU threads for PyTorch/OMP/MKL")
    parser.add_argument("--awkward-file", dest="awkward_file", help="Path to domain terms file (one term per line). If omitted, defaults to repo-root awkward_words.* when present.")
    args = parser.parse_args()

    outdir = args.outdir or None  # default to same-as-source when not specified

    if args.gui or "--gui" in sys.argv:
        # Launch GUI in this process
        launch_gui(default_outdir=outdir, default_threads=args.threads)
        return

    if args.input:
        # Run headless
        q = queue.Queue()

        def runner():
            # Apply domain terms file for headless mode
            try:
                dom = args.awkward_file.strip() if args.awkward_file else None
                if not dom:
                    dom = _repo_default_terms_file()
                if dom and os.path.exists(dom):
                    os.environ["TRANSCRIBE_AWKWARD_FILE"] = dom
                    q.put(f"Using domain terms file: {os.path.basename(dom)}\n")
            except Exception as e:
                q.put(f"Warning: could not apply domain terms file: {e}\n")

            p = args.input
            if os.path.isdir(p):
                # Batch over folder (non-recursive)
                files = [
                    os.path.join(p, name)
                    for name in sorted(os.listdir(p))
                    if os.path.isfile(os.path.join(p, name)) and os.path.splitext(name)[1].lower() in SUPPORTED_EXTS
                ]
                if not files:
                    q.put("No supported media files found in the provided folder.\n")
                else:
                    run_batch_transcription(files, outdir_override=outdir, output_queue=q, threads_override=args.threads)
            else:
                run_transcription(p, outdir, q, threads_override=args.threads)

        t = threading.Thread(target=runner)
        t.start()

        # Print live output
        while t.is_alive() or not q.empty():
            try:
                msg = q.get(timeout=0.2)
                print(msg, end="")
            except queue.Empty:
                pass
        return

    # Launch GUI in detached process
    try:
        subprocess.run(["cmd", "/c", "start", "python", __file__, "--gui"], 
                      creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
        print("GUI launched in background. You can now use the terminal for other commands.")
    except Exception as e:
        print(f"Failed to launch GUI in background: {e}")
        print("Launching GUI in foreground instead...")
        launch_gui(default_outdir=outdir)


if __name__ == "__main__":
    main()
