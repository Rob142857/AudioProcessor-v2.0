# AudioProcessor v2.0

Audio and video transcription for Windows x64 using local NVIDIA Parakeet or Faster-Whisper, with protected GLM cleanup for the Dr Philip Groves archive.

For archive-wide work, use the resume-safe pipeline documented in [docs/UNIFIED_PIPELINE.md](docs/UNIFIED_PIPELINE.md). It writes durable raw text before review, checkpoints each stage, uses the complete pinned glossary for GLM cleanup, and writes working artifacts into a separate output tree. With the Parakeet default, one local GPU worker continues to the next recording as soon as its raw transcript is durable, while two independent GLM review workers consume that queue. Fresh STT publishes `<name>.docx` plus `<name> - GLM Review.docx`; existing-transcript mode leaves the source `<name>.docx` byte-for-byte unchanged and publishes only the review sibling. Every GLM Review copy carries a removable `Needs human review.` notice beneath its provenance statement.

## What It Does

Converts audio/video files into professionally formatted DOCX transcripts. It detects available NVIDIA acceleration and retains a CPU fallback.

### Key Features

- **NVIDIA Parakeet TDT 0.6B v3** — fast local archive default, loaded once per batch
- **Ten GLM review workers** — begin as each durable Parakeet transcript is ready, without holding the GPU lane
- **Faster-Whisper Large-v3** — retained comparison/fallback option
- **Faster-Whisper Large-v3-turbo** — faster comparison/overflow option
- **Native Whisper Large-v3** — OpenAI fallback (GPU or CPU)
- **Vintage tape preprocessing** — noise reduction, loudness normalisation, dynamic range compression optimised for 1980s–90s recordings
- **Single file or recursive batch** — process one file or an entire folder tree
- **Stateful archive resume** — source/model/config hashes and stage manifests replace existence-only skipping in `archive_pipeline.py`
- **Skip-Whisper archive refresh** — import an existing source-adjacent legacy DOCX, leave it untouched, and publish protected GLM output as a separate ` - GLM Review.docx`
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

The GUI defaults to **Polished archive pipeline**: local NVIDIA Parakeet,
ten protected GLM-4.7-Flash cleanup workers, and separate human-review Word
publication. Its two live panes show the GPU speech-to-text lane and the GLM
review queue independently.
It runs the full environment and protected-access preflight before opening any
audio. Existing-output selection remains available, including strict
`Refresh transcripts before…`; normal reruns reuse verified checkpoints, while
the separate advanced reprocess checkbox deliberately starts again from audio.
Switch polished mode off only when a local, uncleaned legacy transcript is
specifically wanted. Cleanup credentials remain in Windows Credential Manager
and are never stored in the GUI project settings.

When a collection already has legacy Faster-Whisper Word transcripts, select
**Use existing Word transcripts (skip Whisper)** and **Refresh all** (or the
strict before-date policy). The route becomes existing Word → protected
GLM-4.7-Flash → separate GLM Review Word. It does not load a speech model, decode audio,
probe audio duration, or create timestamp artifacts. Force means rerun GLM and
Word from the preserved imported `raw.txt`; it never re-imports a document that
may already have been polished.

For recursive transcription plus protected GLM cleanup:

```powershell
.\.venv\Scripts\python.exe configure_cleanup_credentials.py
.\run_full_pipeline.bat "C:\path\to\recordings" --limit 1 --no-publish-source-docx
```

The first command validates the service token and stores it in Windows Credential Manager; it does not write the token to repository settings. Environment variables remain available for unattended runs. Review the one-file canary before removing `--limit 1`.

The equivalent headless skip-Whisper route is:

```powershell
.\run_full_pipeline.bat "C:\path\to\recordings" --existing-transcripts-only --existing-docx-mode all
```

### Local Parakeet comparison

Parakeet is installed in an isolated environment and is now the production GUI
default. The comparison runner remains useful for one-recording checks; it
writes only `<name> - Parakeet v3.txt` and a small provenance JSON file beside it.

```powershell
.\.parakeet-venv\Scripts\python.exe .\parakeet_compare.py "C:\path\to\one-recording-folder"
```

Do not run this while the GUI is using the GTX 1070 Ti for Parakeet.

For a complete unrestricted archive-folder run, `run_full_pipeline.bat` publishes each completed reviewed job immediately. Fresh STT creates or refreshes the raw speech `<name>.docx` according to the selected policy and writes GLM output to `<name> - GLM Review.docx`. Existing-transcript mode never changes the source DOCX. An older tool-generated review copy is atomically replaced only when its hash is proven by a prior journal; a manually changed review fails closed. Exact backups live under the separate polished workspace in `publication-backups/<run-id>/`, never as archive sidecars. A `needs_review` result is still published for human checking and always records `approval_state: pending_human_review`; failed or incomplete jobs are not published. Use `--no-publish-source-docx` to keep results only in the separate polished output tree.

The GUI checkbox **Retain detailed troubleshooting logging** defaults on. Turning it off prevents optional `run.jsonl` event logs and stores compact terminology digests instead of duplicating the full selected/dropped glossary lists. Operational manifests, hashes and checkpoints remain because safe resume and provenance depend on them.

## Source DOCX cleanup after review

When the archive run is complete and its GLM Review documents are accepted, use
the separate planner below. It is **read-only by default**. It recursively
keeps only source-side ` - GLM Review.docx` files proven by completed local
Parakeet manifests in the polished workspace, and lists every other source
DOCX for review:

```powershell
python prepare_docx_cleanup.py "C:\\path\\to\\archive"
```

After reviewing the exact count, a deliberate `--apply --expected-count N`
does not permanently erase files: it moves the approved non-final DOCX files
to `docx-cleanup-quarantine` in the polished workspace, preserving the folder
structure for recovery. Do not run the apply mode until the batch is complete
and you have approved the dry-run list.

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
