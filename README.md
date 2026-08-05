# AudioProcessor v2.0

Audio and video transcription for Windows x64 using Faster-Whisper (CTranslate2), with an optional protected GLM cleanup stage for the Dr Philip Groves archive.

For archive-wide work, use the new resume-safe pipeline documented in [docs/UNIFIED_PIPELINE.md](docs/UNIFIED_PIPELINE.md). It preserves raw text and timestamps, checkpoints each stage, prevents same-stem overwrites, calls the protected GLM-4.7-Flash cleanup service, and writes final DOCX files into a separate output tree. The normal folder-batch launcher then publishes verified DOCX files beside their source recordings, with originals retained in a separate timestamped backup tree.

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

The supported GTX 1070 Ti lane is exactly Python 3.12 x64, PyTorch 2.6.0+cu124, Faster-Whisper 1.2.1, and CTranslate2 4.8.1. Review and run the local installer; do not pipe a changing remote script into PowerShell.

```powershell
git clone https://github.com/Rob142857/AudioProcessor-v2.0.git
cd AudioProcessor-v2.0
.\install_geforce.ps1
```

The installer tries registered `py -3.12` first, then `%LOCALAPPDATA%\Programs\Python\Python312\python.exe`. It installs the exact cu124 lane and checks CUDA 12.4, compute capability 6.1/`sm_61`, and CTranslate2 INT8 support, but deliberately does not download every model. Launch the GUI after setup with:

```powershell
.\run.bat
```

The GUI defaults to **Polished archive pipeline**: local Faster-Whisper,
protected GLM-4.7-Flash cleanup, verification, and polished Word publication.
It runs the full environment and protected-access preflight before opening any
audio. Existing-output selection remains available, including strict
`Replace transcripts before…`; normal reruns reuse verified checkpoints, while
the separate advanced reprocess checkbox deliberately starts again from audio.
Switch polished mode off only when a local, uncleaned legacy transcript is
specifically wanted. Cleanup credentials remain in Windows Credential Manager
and are never stored in the GUI project settings.

For recursive transcription plus protected GLM cleanup:

```powershell
.\.venv\Scripts\python.exe configure_cleanup_credentials.py
.\run_full_pipeline.bat "C:\path\to\recordings" --limit 1 --no-publish-source-docx
```

The first command validates the service token and stores it in Windows Credential Manager; it does not write the token to repository settings. Environment variables remain available for unattended runs. Review the one-file canary before removing `--limit 1`.

For a complete unrestricted archive-folder run, `run_full_pipeline.bat` publishes each verified final DOCX beside its source audio by default. Before replacing any existing DOCX, the publisher creates a new sibling backup tree named `<archive-name> - Legacy DOCX Backup - <UTC run-id>`, verifies the whole batch, and retains the backups. Publication is blocked if any job failed, needs review, was cancelled, or if `--dry-run` or `--limit` is active. Use `--no-publish-source-docx` to keep results only in the separate polished output tree.

Single-file publication is supported for a reviewed canary. Always give it a new, dedicated `--output` directory which contains only that run's manifest; its guarded publication scope is the input file's parent directory. Folder batches remain the normal archive workflow.

```powershell
.\run_full_pipeline.bat "C:\path\to\recordings\one-tape.wav" --output "C:\path\to\fresh-canary-output"
```

## Requirements

- Windows 10/11 x64
- Python 3.12 x64
- PyTorch 2.6.0+cu124, Faster-Whisper 1.2.1, CTranslate2 4.8.1
- 8 GB+ RAM (16 GB recommended)
- NVIDIA GPU with 8 GB VRAM (GTX 1070 Ti or newer recommended)
- bundled `ffmpeg.exe` or a compatible FFmpeg on PATH
