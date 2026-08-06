@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

set "AUDIOPROCESSOR_PYTHON=.venv\Scripts\python.exe"
rem Prefer the established Faster-Whisper environment when it is healthy, but
rem keep the new Parakeet-default GUI usable when an old .venv points to a
removed base Python installation. The selected pipeline still runs a strict
rem model-specific preflight before touching an archive.
"%AUDIOPROCESSOR_PYTHON%" -c "import keyring, psutil, tkinter; from docx import Document" 2>nul
if errorlevel 1 (
  set "AUDIOPROCESSOR_PYTHON=.parakeet-venv\Scripts\python.exe"
  "%AUDIOPROCESSOR_PYTHON%" -c "import keyring, psutil, tkinter; from docx import Document" 2>nul
  if errorlevel 1 (
    echo AudioProcessor's local environments are incomplete.
    echo.
    echo Run the reviewed installer first:
    echo   .\install_geforce.ps1 -RecreateVenv
    echo.
    pause
    exit /b 1
  )
  echo Using the isolated NVIDIA Parakeet environment for the GUI.
)

"%AUDIOPROCESSOR_PYTHON%" gui_transcribe.py --gui

if errorlevel 1 (
  echo.
  echo The GUI failed to start. Check the error messages above.
  echo.
)
pause
endlocal
