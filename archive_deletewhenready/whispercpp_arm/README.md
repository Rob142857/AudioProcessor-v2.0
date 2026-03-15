# Whisper.cpp ARM64 Transcription App with NPU Acceleration

This is a high-performance transcription tool for ARM64 Windows devices, using Whisper.cpp (C++) with OpenVINO backend for Neural Processing Unit (NPU) acceleration, providing up to 50x realtime transcription performance.

## ✨ Key Features

- **🚀 NPU Acceleration**: Automatic Neural Processing Unit acceleration via OpenVINO backend
- **🎯 High Performance**: 6-15x realtime transcription for 30-minute audio files
- **📊 Professional Quality**: Uses Turbo v3 model for excellent accuracy with smaller footprint
- **🎨 Modern GUI**: Dedicated ARM64 interface with real-time progress tracking
- **📁 Batch Processing**: Transcribe individual files or entire folders
- **📄 Multiple Outputs**: Generates .txt and .docx files with intelligent paragraph formatting
- **🔧 Easy Installation**: Automatic setup via universal installer

## 🏗️ Architecture

**Core Components:**
- **Whisper.cpp**: C++ implementation with ARM64 optimizations
- **OpenVINO Backend**: Provides NPU, GPU, and CPU device support
- **Turbo v3 Model**: Optimized model for speed and quality balance
- **Python Orchestration**: Tkinter GUI with threading for responsive interface

**NPU Acceleration:**
- Automatic device detection (`-oved NPU` flag)
- OpenVINO runtime handles hardware acceleration
- Fallback to CPU if NPU unavailable
- Real-time performance monitoring

## 📋 Requirements

- **Hardware**: Windows ARM64 device with NPU support
- **OS**: Windows 11 ARM64
- **Python**: 3.11+ (automatically installed by universal installer)
- **Dependencies**: ONNX Runtime, Transformers, NumPy, python-docx
- **Binaries**: Pre-built ARM64 whisper.cpp binaries included

## 🚀 Quick Start

1. **Run Universal Installer** (from project root):
   ```cmd
   install_universal.bat
   ```
   This automatically detects ARM64 and sets up the NPU-accelerated environment.

2. **Launch GUI**:
   ```cmd
   cd whispercpp_arm
   python gui_transcribe.py
   ```

3. **Transcribe Files**:
   - Click "Select File(s)" or "Select Folder"
   - Choose your audio/video files
   - Click "Transcribe" and watch NPU-accelerated progress

## 📊 Performance

**Typical Performance on ARM64 with NPU:**
- **30-minute audio file**: 2-5 minutes processing time
- **Speed**: 6-15x realtime
- **Quality**: Professional-grade with Turbo v3 model
- **Memory Usage**: Optimized for ARM64 devices

**Hardware Utilization:**
- **NPU**: Primary acceleration device
- **CPU**: Fallback and orchestration
- **GPU**: Optional secondary acceleration (if available)

## 📁 File Structure

```
whispercpp_arm/
├── gui_transcribe.py          # Main ARM64 GUI (NPU accelerated)
├── transcribe.py              # Core transcription logic
├── requirements.txt           # ARM64-specific dependencies
├── models/                    # Turbo v3 model files
├── output/                    # Transcription outputs
├── whisper.cpp/               # Pre-built ARM64 binaries
│   └── build/bin/
│       └── whisper-cli.exe    # NPU-capable executable
└── .venv_arm/                 # Isolated ARM64 environment
```

## 🔧 Configuration

**Automatic Configuration:**
- NPU detection and utilization
- Model path resolution
- Output directory management
- Hardware-specific optimizations

**Command Line (Advanced):**
```cmd
whisper-cli.exe -m models/ggml-large-v3-turbo.bin -f input.wav -otxt -of output -oved NPU
```

## 🛠️ Troubleshooting

**NPU Issues:**
- **Not Detected**: Ensure OpenVINO runtime is installed
- **Slow Performance**: Check for background processes using NPU
- **Errors**: Verify ARM64 architecture and driver versions

**Installation Issues:**
- **Missing Files**: Ensure using latest repository version
- **Python Errors**: Check .venv_arm environment activation
- **Binary Issues**: Verify whisper-cli.exe is ARM64 build

**Performance Optimization:**
- Close other applications using NPU/GPU
- Ensure latest Windows ARM64 drivers
- Monitor system resources during transcription

## 🔄 Integration with Main Project

This ARM64 implementation is fully integrated with the main AudioProcessorAlphaVersion project:

- **Universal Installer**: `install_universal.bat` automatically detects ARM64
- **Shared Documentation**: Main README includes ARM64 NPU information
- **Consistent Interface**: Similar GUI experience across architectures
- **Unified Support**: Single repository for all platform variants

## 📈 Future Enhancements

- [ ] ONNX punctuation restoration integration
- [ ] Enhanced NPU utilization monitoring
- [ ] Additional OpenVINO device optimizations
- [ ] ARM64-specific model fine-tuning
- [ ] Advanced AI guardrails (copyright mute, language filtering)

---

**Ready to experience lightning-fast ARM64 transcription?** Run `install_universal.bat` and start transcribing with NPU acceleration!
