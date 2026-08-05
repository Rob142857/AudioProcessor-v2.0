# AudioProcessor v2.0

Audio and video transcription for Windows x64 using Faster-Whisper (CTranslate2), with an optional protected GLM cleanup stage for the Dr Philip Groves archive.

For archive-wide work, use the new resume-safe pipeline documented in [docs/UNIFIED_PIPELINE.md](docs/UNIFIED_PIPELINE.md). It preserves raw text and timestamps, checkpoints each stage, prevents same-stem overwrites, calls the protected GLM-4.7-Flash cleanup service, and writes final DOCX files into a separate output tree.

## What It Does

Converts audio/video files into professionally formatted DOCX transcripts. It detects available NVIDIA acceleration and retains a CPU fallback.

### Key Features

- **Faster-Whisper Large-v3** — fidelity-oriented local default
- **Faster-Whisper Large-v3-turbo** — faster comparison/overflow option
- **Native Whisper Large-v3** — OpenAI fallback (GPU or CPU)
- **Vintage tape preprocessing** — noise reduction, loudness normalisation, dynamic range compression optimised for 1980s–90s recordings
- **Single file or recursive batch** — process one file or an entire folder tree
- **Stateful archive resume** — source/model/config hashes and stage manifests replace existence-only skipping in `archive_pipeline.py`
- **Per-project settings** — model, recursive mode, and replace policy saved per folder
- **Domain terms** — feed a `special_words.txt` to improve recognition of specialist vocabulary
- **Clean componentised GUI** — `gui_transcribe.py` (main) + `gui_components.py` (panels)

## Quick Start

### Safe local setup

Use Python 3.11 or 3.12. Review installation scripts locally before running them; the project no longer recommends piping a changing remote PowerShell script directly into `iex`.

```powershell
git clone https://github.com/Rob142857/AudioProcessor-v2.0.git
cd AudioProcessor-v2.0
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Install the PyTorch wheel appropriate for your GPU from pytorch.org.
# Faster-Whisper also requires compatible CUDA 12 cuBLAS/cuDNN runtime DLLs.

# Cache the models (~4.7 GB total)
python preload_models.py

# Verify, then launch the legacy GUI
python pipeline_doctor.py
python gui_transcribe.py
```

Or use the one-click bootstrap:
```powershell
.\run.bat
```

For recursive transcription plus protected GLM cleanup:

```powershell
.\.venv\Scripts\python.exe configure_cleanup_credentials.py
.\run_full_pipeline.bat "C:\path\to\recordings" --limit 1
```

The first command validates the service token and stores it in Windows Credential Manager; it does not write the token to repository settings. Environment variables remain available for unattended runs. Review the one-file canary before removing `--limit 1`.

## Requirements

- Windows 10/11 x64
- Python 3.11 or 3.12 recommended/tested target
- 8 GB+ RAM (16 GB recommended)
- NVIDIA GPU with 8 GB VRAM (GTX 1070 Ti or newer recommended)
- bundled `ffmpeg.exe` or a compatible FFmpeg on PATH
