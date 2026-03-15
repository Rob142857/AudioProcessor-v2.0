@echo off
REM Download and build Whisper.cpp for Windows ARM64
REM Requires: git, cmake, Visual Studio 2022 (ARM64 tools)

REM Clone the repo
if not exist whisper.cpp (
    git clone https://github.com/ggerganov/whisper.cpp.git
)
cd whisper.cpp

REM Download the large-v3-turbo model (optional, already handled above)
REM call models\download-ggml-model.cmd large-v3-turbo

REM Create build directory
if exist build rmdir /s /q build
mkdir build
cd build

REM Configure for ARM64 with clang-cl using Ninja
"C:\Program Files\CMake\bin\cmake.exe" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="C:\Program Files\LLVM\bin\clang-cl.exe" -DCMAKE_CXX_COMPILER="C:\Program Files\LLVM\bin\clang-cl.exe" -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DCMAKE_RC_COMPILER="C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/arm64/rc.exe" -DCMAKE_MAKE_PROGRAM="C:\Users\RobertEvans\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe\ninja.exe" ..

REM Build the CLI binary
"C:\Program Files\CMake\bin\cmake.exe" --build . --config Release --target whisper-cli

REM Copy the binary to parent folder for use
cd bin
copy whisper-cli.exe ..\..\whisper.cpp.exe
cd ..\..

@echo off
if exist whisper.cpp.exe (
    echo Build and copy successful: whisper.cpp.exe
) else (
    echo Build failed or binary not found.
)
