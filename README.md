# AudioProcessor v2.0

High-performance audio and video transcription for Windows x64 using Faster-Whisper (CTranslate2) with automatic NVIDIA GPU acceleration.

## What It Does

Converts audio/video files into professionally formatted DOCX transcripts. Automatically detects and uses NVIDIA GPUs for up to 8x real-time processing, with CPU fallback.

### Key Features

- **Faster-Whisper Large-v3** — best accuracy, ~4x real-time on GPU (int8)
- **Faster-Whisper Large-v3-turbo** — 2x faster, half the VRAM, excellent quality
- **Native Whisper Large-v3** — OpenAI fallback (GPU or CPU)
- **Vintage tape preprocessing** — noise reduction, loudness normalisation, dynamic range compression optimised for 1980s–90s recordings
- **Single file or recursive batch** — process one file or an entire folder tree
- **Skip / replace / replace-before-date** — resume interrupted batches without re-transcribing
- **Per-project settings** — model, recursive mode, and replace policy saved per folder
- **Domain terms** — feed a `special_words.txt` to improve recognition of specialist vocabulary
- **Clean componentised GUI** — `gui_transcribe.py` (main) + `gui_components.py` (panels)

## Quick Start

```powershell
# Clone and set up
git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git
cd AudioProcessorAlphaVersion
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Install PyTorch for your GPU — example for CUDA 12.4:
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Cache the models (~4.7 GB total)
python preload_models.py

# Launch
python gui_transcribe.py
```

Or use the one-click bootstrap:
```powershell
.\run.bat
```

## Requirements

- Windows 10/11 x64
- Python 3.11+
- 8 GB+ RAM (16 GB recommended)
- NVIDIA GPU with 8 GB VRAM (GTX 1070 Ti or newer recommended)
- `ffmpeg` on PATH
