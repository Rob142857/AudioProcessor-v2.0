@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

set "AUDIOPROCESSOR_PYTHON=.venv\Scripts\python.exe"
if not exist "%AUDIOPROCESSOR_PYTHON%" (
  echo AudioProcessor's pinned local environment was not found.
  echo.
  echo Run the reviewed installer first:
  echo   .\install_geforce.ps1
  echo.
  pause
  exit /b 1
)

"%AUDIOPROCESSOR_PYTHON%" -c "import faster_whisper, keyring, psutil, torch, whisper; from docx import Document" 2>nul
if errorlevel 1 (
  echo AudioProcessor's pinned environment is incomplete.
  echo.
  echo Run:
  echo   .\install_geforce.ps1
  echo.
  pause
  exit /b 1
)

"%AUDIOPROCESSOR_PYTHON%" gui_transcribe.py --gui

if errorlevel 1 (
  echo.
  echo The GUI failed to start. Check the error messages above.
  echo.
)
pause
endlocal
