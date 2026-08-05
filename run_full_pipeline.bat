@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

set "PIPELINE_PYTHON=.venv\Scripts\python.exe"
if not exist "%PIPELINE_PYTHON%" (
  echo AudioProcessor's pinned local environment was not found.
  echo.
  echo Run the reviewed local installer first:
  echo   .\install_geforce.ps1
  echo.
  echo Supported lane: Python 3.12 x64, torch 2.6.0+cu124,
  echo Faster-Whisper 1.2.1, and CTranslate2 4.8.1.
  pause
  exit /b 1
)

rem Parse launcher policy flags while preserving all pipeline arguments.
rem --no-publish-source-docx belongs to this launcher and is never forwarded.
set "PIPELINE_ARGS="
set "FLAG_NO_CLEANUP=0"
set "FLAG_CLEANUP_ONLY=0"
set "FLAG_RENDER_ONLY=0"
set "FLAG_DRY_RUN=0"
set "FLAG_EXISTING_TRANSCRIPTS_ONLY=0"
set "PUBLISH_SOURCE_DOCX=--publish-source-docx"

:parse_arguments
if "%~1"=="" goto arguments_parsed
if /I "%~1"=="--no-publish-source-docx" goto disable_source_publish
if /I "%~1"=="--publish-source-docx" goto enable_source_publish
if /I "%~1"=="--no-cleanup" set "FLAG_NO_CLEANUP=1"
if /I "%~1"=="--cleanup-only" set "FLAG_CLEANUP_ONLY=1"
if /I "%~1"=="--render-only" set "FLAG_RENDER_ONLY=1"
if /I "%~1"=="--dry-run" set "FLAG_DRY_RUN=1"
if /I "%~1"=="--existing-transcripts-only" set "FLAG_EXISTING_TRANSCRIPTS_ONLY=1"
if /I "%~1"=="--use-existing-docx" set "FLAG_EXISTING_TRANSCRIPTS_ONLY=1"
if /I "%~1"=="--skip-stt" set "FLAG_EXISTING_TRANSCRIPTS_ONLY=1"
set "PIPELINE_ARGS=%PIPELINE_ARGS% %1"
shift
goto parse_arguments

:disable_source_publish
set "PUBLISH_SOURCE_DOCX="
shift
goto parse_arguments

:enable_source_publish
set "PUBLISH_SOURCE_DOCX=--publish-source-docx"
shift
goto parse_arguments

:arguments_parsed
if "%FLAG_CLEANUP_ONLY%"=="1" if "%FLAG_RENDER_ONLY%"=="1" (
  echo --cleanup-only and --render-only are mutually exclusive.
  pause
  exit /b 2
)

set "DOCTOR_MODE=full"
set "DOCTOR_GPU_ARG=--require-gpu"
set "DOCTOR_CLEANUP_ARG="

if "%FLAG_DRY_RUN%"=="1" goto doctor_inventory
if "%FLAG_EXISTING_TRANSCRIPTS_ONLY%"=="1" goto doctor_existing_transcripts
if "%FLAG_RENDER_ONLY%"=="1" goto doctor_render
if "%FLAG_CLEANUP_ONLY%"=="1" goto doctor_cleanup
if "%FLAG_NO_CLEANUP%"=="1" goto doctor_transcribe_only
goto doctor_ready

:doctor_inventory
set "DOCTOR_MODE=inventory"
set "DOCTOR_GPU_ARG="
set "DOCTOR_CLEANUP_ARG=--no-cleanup"
set "PUBLISH_SOURCE_DOCX="
goto doctor_ready

:doctor_existing_transcripts
set "DOCTOR_MODE=cleanup-only"
set "DOCTOR_GPU_ARG="
goto doctor_ready

:doctor_render
set "DOCTOR_MODE=render-only"
set "DOCTOR_GPU_ARG="
set "DOCTOR_CLEANUP_ARG=--no-cleanup"
goto doctor_ready

:doctor_cleanup
set "DOCTOR_MODE=cleanup-only"
set "DOCTOR_GPU_ARG="
if "%FLAG_NO_CLEANUP%"=="1" set "DOCTOR_CLEANUP_ARG=--no-cleanup"
goto doctor_ready

:doctor_transcribe_only
set "DOCTOR_MODE=transcribe"
set "DOCTOR_GPU_ARG=--require-gpu"
set "DOCTOR_CLEANUP_ARG=--no-cleanup"

:doctor_ready
"%PIPELINE_PYTHON%" pipeline_doctor.py --mode "%DOCTOR_MODE%" %DOCTOR_GPU_ARG% %DOCTOR_CLEANUP_ARG%
if errorlevel 1 (
  echo.
  echo Preflight failed. No archive files were changed.
  pause
  exit /b 1
)

"%PIPELINE_PYTHON%" archive_pipeline.py %PIPELINE_ARGS% %PUBLISH_SOURCE_DOCX%
set "PIPELINE_EXIT=%ERRORLEVEL%"
echo.
if "%PIPELINE_EXIT%"=="0" echo Pipeline completed with all generated artifacts verified.
if "%PIPELINE_EXIT%"=="1" echo Pipeline stopped or one or more jobs failed. Run it again to resume.
if "%PIPELINE_EXIT%"=="3" echo Pipeline completed, with one or more transcripts marked needs_review.
pause
exit /b %PIPELINE_EXIT%
