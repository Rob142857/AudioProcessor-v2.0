@echo off
REM Download Whisper.cpp large-v3-turbo model using curl or PowerShell
set MODEL_URL=https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
set MODEL_PATH=models\ggml-large-v3-turbo.bin

if not exist models mkdir models

REM Try curl first
curl -L %MODEL_URL% -o %MODEL_PATH%
if exist %MODEL_PATH% (
    echo Model downloaded successfully: %MODEL_PATH%
    exit /b 0
)

REM Fallback to PowerShell if curl fails
powershell -Command "Invoke-WebRequest -Uri '%MODEL_URL%' -OutFile '%MODEL_PATH%'"
if exist %MODEL_PATH% (
    echo Model downloaded successfully: %MODEL_PATH%
    exit /b 0
)

echo Failed to download model. Please check your internet connection or download manually.
exit /b 1
