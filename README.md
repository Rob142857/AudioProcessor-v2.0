# AudioProcessor v1.02beta

High-performance audio and video transcription for Windows x64 using OpenAI Whisper with automatic GPU acceleration.

## What It Does

Converts audio/video files into professionally formatted text transcripts with advanced AI-powered punctuation restoration. Automatically detects and utilizes NVIDIA GPUs for 3-5x faster processing, with CPU fallback.

### ✨ New: Advanced Punctuation Restoration

Now includes specialized BERT-based punctuation refinement (`oliverguhr/fullstop-punctuation-multilang-large`) that significantly improves sentence and paragraph boundary detection in spoken language transcripts. See [PUNCTUATION_QUICKSTART.md](PUNCTUATION_QUICKSTART.md) for details.

## Installation

Run this single command:
```powershell
irm https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install_geforce.ps1 | iex
```

This installs Python, creates a virtual environment, installs dependencies, downloads models, and launches the GUI.


## Requirements

- Windows 10/11 x64
- 8GB+ RAM (16GB recommended)
- Optional: NVIDIA GPU for acceleration
