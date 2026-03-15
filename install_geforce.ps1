<#
.SYNOPSIS
    AudioProcessor v2.0 — One-liner bootstrap for Windows x64 + NVIDIA GeForce.

.DESCRIPTION
    Downloads and sets up everything needed to run AudioProcessor on a GeForce GPU:
      1. Checks / installs Python 3.11+
      2. Checks / installs Git
      3. Checks / installs FFmpeg
      4. Clones the repository
      5. Creates a virtual environment and installs dependencies
      6. Installs PyTorch with CUDA 12.4 support
      7. Pre-downloads the recommended Whisper models
      8. Launches the GUI

    Usage (one-liner from any PowerShell):
      irm https://raw.githubusercontent.com/Rob142857/AudioProcessor-v2.0/main/install_geforce.ps1 | iex

.NOTES
    Requires: Windows 10/11 x64, NVIDIA GeForce GPU with recent drivers, internet.
    The script will NOT modify system Python — it uses a virtual environment.
#>

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoURL  = "https://github.com/Rob142857/AudioProcessor-v2.0.git"
$RepoName = "AudioProcessor-v2.0"
$InstallDir = Join-Path $env:USERPROFILE $RepoName

# ── Helpers ──────────────────────────────────────────────────────────
function Write-Step  { param([string]$Msg) Write-Host "`n==> $Msg" -ForegroundColor Cyan }
function Write-Ok    { param([string]$Msg) Write-Host "    OK: $Msg" -ForegroundColor Green }
function Write-Warn  { param([string]$Msg) Write-Host "    WARN: $Msg" -ForegroundColor Yellow }
function Write-Fail  { param([string]$Msg) Write-Host "    FAIL: $Msg" -ForegroundColor Red }

function Test-Command { param([string]$Name) $null -ne (Get-Command $Name -ErrorAction SilentlyContinue) }

# ── 1. Python ────────────────────────────────────────────────────────
Write-Step "Checking Python..."
$py = $null
foreach ($candidate in @("python", "python3", "py")) {
    if (Test-Command $candidate) {
        $ver = & $candidate --version 2>&1
        if ($ver -match "(\d+)\.(\d+)") {
            $major = [int]$Matches[1]; $minor = [int]$Matches[2]
            if ($major -eq 3 -and $minor -ge 10) {
                $py = $candidate
                Write-Ok "$ver"
                break
            }
        }
    }
}
if (-not $py) {
    Write-Warn "Python 3.10+ not found. Attempting install via winget..."
    if (Test-Command "winget") {
        winget install --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements
        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                     [System.Environment]::GetEnvironmentVariable("Path", "User")
        if (Test-Command "python") { $py = "python"; Write-Ok "Python installed." }
    }
    if (-not $py) {
        Write-Fail "Could not install Python automatically."
        Write-Host "Please install Python 3.11 from https://www.python.org/downloads/ and re-run." -ForegroundColor Yellow
        exit 1
    }
}

# ── 2. Git ───────────────────────────────────────────────────────────
Write-Step "Checking Git..."
if (Test-Command "git") {
    Write-Ok (git --version)
} else {
    Write-Warn "Git not found. Attempting install via winget..."
    if (Test-Command "winget") {
        winget install --id Git.Git --accept-source-agreements --accept-package-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                     [System.Environment]::GetEnvironmentVariable("Path", "User")
    }
    if (-not (Test-Command "git")) {
        Write-Fail "Could not install Git. Please install from https://git-scm.com/ and re-run."
        exit 1
    }
    Write-Ok "Git installed."
}

# ── 3. FFmpeg ────────────────────────────────────────────────────────
Write-Step "Checking FFmpeg..."
if (Test-Command "ffmpeg") {
    Write-Ok "ffmpeg found on PATH."
} else {
    Write-Warn "FFmpeg not found. Attempting install via winget..."
    if (Test-Command "winget") {
        winget install --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                     [System.Environment]::GetEnvironmentVariable("Path", "User")
    }
    if (Test-Command "ffmpeg") {
        Write-Ok "FFmpeg installed."
    } else {
        Write-Warn "FFmpeg not on PATH. Transcription may fail without it."
        Write-Host "    Install from https://ffmpeg.org/download.html or: winget install Gyan.FFmpeg" -ForegroundColor Yellow
    }
}

# ── 4. Clone repo ───────────────────────────────────────────────────
Write-Step "Cloning AudioProcessor..."
if (Test-Path (Join-Path $InstallDir ".git")) {
    Write-Ok "Repository already exists at $InstallDir — pulling latest."
    Push-Location $InstallDir
    git pull --ff-only 2>&1 | Out-Null
    Pop-Location
} else {
    if (Test-Path $InstallDir) {
        Write-Warn "$InstallDir exists but is not a git repo. Backing up and re-cloning."
        Rename-Item $InstallDir "$InstallDir.bak_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    }
    git clone $RepoURL $InstallDir
    Write-Ok "Cloned to $InstallDir"
}
Set-Location $InstallDir

# ── 5. Virtual environment ──────────────────────────────────────────
Write-Step "Creating virtual environment..."
$venvDir = Join-Path $InstallDir ".venv"
if (-not (Test-Path (Join-Path $venvDir "Scripts" "python.exe"))) {
    & $py -m venv $venvDir
    Write-Ok "venv created."
} else {
    Write-Ok "venv already exists."
}

$venvPython = Join-Path $venvDir "Scripts" "python.exe"
$venvPip    = Join-Path $venvDir "Scripts" "pip.exe"

Write-Step "Upgrading pip..."
& $venvPython -m pip install --upgrade pip --quiet 2>&1 | Out-Null
Write-Ok "pip up to date."

# ── 6. Install dependencies ─────────────────────────────────────────
Write-Step "Installing requirements..."
& $venvPip install -r (Join-Path $InstallDir "requirements.txt") --quiet 2>&1 | Out-Null
Write-Ok "Requirements installed."

Write-Step "Installing PyTorch with CUDA 12.4 support (GeForce)..."
& $venvPip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet 2>&1 | Out-Null
Write-Ok "PyTorch (CUDA 12.4) installed."

# ── 7. GPU check ────────────────────────────────────────────────────
Write-Step "Verifying GPU..."
$gpuCheck = & $venvPython -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f'{name} ({mem:.1f} GB VRAM)')
else:
    print('NO_GPU')
" 2>&1
if ($gpuCheck -match "NO_GPU") {
    Write-Warn "CUDA GPU not detected. PyTorch will fall back to CPU."
    Write-Host "    Ensure NVIDIA drivers are installed: https://www.nvidia.com/Download/index.aspx" -ForegroundColor Yellow
} else {
    Write-Ok "GPU: $gpuCheck"
}

# ── 8. Preload models ───────────────────────────────────────────────
Write-Step "Pre-downloading Whisper models (this may take a few minutes)..."
& $venvPython (Join-Path $InstallDir "preload_models.py") 2>&1 | ForEach-Object { Write-Host "    $_" }
Write-Ok "Models cached."

# ── 9. Launch ────────────────────────────────────────────────────────
Write-Step "Setup complete! Launching AudioProcessor..."
Write-Host ""
Write-Host "  Install location: $InstallDir" -ForegroundColor White
Write-Host "  To run again:     cd '$InstallDir'; .\.venv\Scripts\Activate.ps1; python gui_transcribe.py" -ForegroundColor White
Write-Host ""

& $venvPython (Join-Path $InstallDir "gui_transcribe.py")
