@echo off
setlocal
cd /d "%~dp0"

set "PIPELINE_PYTHON=.venv\Scripts\python.exe"
if not exist "%PIPELINE_PYTHON%" (
  echo AudioProcessor's local environment was not found.
  echo.
  echo Create a tested Python 3.11 or 3.12 environment first:
  echo   py -3.12 -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo.
  echo PyTorch/CUDA must be installed for this computer before local transcription.
  pause
  exit /b 1
)

set "DOCTOR_CLEANUP_ARG="
for %%A in (%*) do (
  if /I "%%~A"=="--no-cleanup" set "DOCTOR_CLEANUP_ARG=--no-cleanup"
  if /I "%%~A"=="--render-only" set "DOCTOR_CLEANUP_ARG=--no-cleanup"
)

"%PIPELINE_PYTHON%" pipeline_doctor.py %DOCTOR_CLEANUP_ARG%
if errorlevel 1 (
  echo.
  echo Preflight failed. No archive files were changed.
  pause
  exit /b 1
)

"%PIPELINE_PYTHON%" archive_pipeline.py %*
set "PIPELINE_EXIT=%ERRORLEVEL%"
echo.
if "%PIPELINE_EXIT%"=="0" echo Pipeline completed with all generated artifacts verified.
if "%PIPELINE_EXIT%"=="1" echo Pipeline stopped or one or more jobs failed. Run it again to resume.
if "%PIPELINE_EXIT%"=="3" echo Pipeline completed, with one or more transcripts marked needs_review.
pause
exit /b %PIPELINE_EXIT%
