#requires -version 5.1
<#
.SYNOPSIS
  One-click installer for AudioProcessor on Windows x64 with NVIDIA GeForce GPUs.

.DESCRIPTION
  Streamlined installer optimized for NVIDIA GeForce users:
  - Auto-detects GeForce GPU and selects optimal CUDA version
  - Installs only essential dependencies (no C++ Build Tools needed)
  - Sets up Python 3.11, creates venv, installs PyTorch with CUDA
  - Installs requirements and preloads Whisper models
  - Launches GUI when complete

.PARAMETER SkipModelPreload
  Skip Whisper model preloading (saves time, models download on first use).

.PARAMETER SkipGUI
  Skip launching the GUI after installation.

.PARAMETER Quiet
  Reduce output verbosity.

.EXAMPLE
  # Quick one-liner (from GitHub):
  irm https://raw.githubusercontent.com/Rob142857/AudioProcessorAlphaVersion/main/install_geforce.ps1 | iex

.EXAMPLE
  # Local install with options:
  powershell -ExecutionPolicy Bypass -File .\install_geforce.ps1 -SkipModelPreload

.NOTES
  Optimized for Windows 10/11 x64 with NVIDIA GeForce GPUs.
  No CUDA Toolkit installation required - PyTorch includes CUDA runtime.
  No C++ Build Tools required - all packages have pre-compiled wheels.
#>

[CmdletBinding()]
param(
  [switch]$SkipModelPreload,
  [switch]$SkipGUI,
  [switch]$Quiet
)

$ErrorActionPreference = 'Stop'

function Write-Info($msg){ if(-not $Quiet){ Write-Host $msg -ForegroundColor Cyan } }
function Write-Ok($msg){ Write-Host $msg -ForegroundColor Green }
function Write-Warn($msg){ Write-Warning $msg }
function Write-Err($msg){ Write-Host $msg -ForegroundColor Red }

# Ensure x64 Windows
if(-not [Environment]::Is64BitOperatingSystem){
  Write-Err "This installer requires Windows x64. Detected: $([Environment]::OSArchitecture)"
  exit 1
}

Write-Host "`n=== AudioProcessor v1.02beta - GeForce Edition ===" -ForegroundColor Green
Write-Host "Optimized installer for NVIDIA GeForce GPUs`n" -ForegroundColor Gray

# Prompt for installation folder (ensures clean install)
$defaultInstallPath = Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'AudioProcessor'
Write-Host "Where would you like to install AudioProcessor?" -ForegroundColor Yellow
Write-Host "Press Enter for default: $defaultInstallPath" -ForegroundColor Gray
$userInput = Read-Host "Install path"

if([string]::IsNullOrWhiteSpace($userInput)){
  $RepoRoot = $defaultInstallPath
} else {
  $RepoRoot = $userInput
}

# Expand any environment variables and resolve relative paths
$RepoRoot = [System.Environment]::ExpandEnvironmentVariables($RepoRoot)
$RepoRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($RepoRoot)

# Check if directory already exists
if(Test-Path $RepoRoot){
  Write-Warn "Directory already exists: $RepoRoot"
  $overwrite = Read-Host "Continue and use this directory? (y/N)"
  if($overwrite -notmatch '^y(es)?$'){
    Write-Info "Installation cancelled."
    exit 0
  }
} else {
  Write-Info "Creating installation directory..."
  New-Item -ItemType Directory -Path $RepoRoot -Force | Out-Null
}

Set-Location $RepoRoot
Write-Ok "Installation directory: $RepoRoot"

# Helper: test if command exists
function Test-Command($name){ $null -ne (Get-Command $name -ErrorAction SilentlyContinue) }

# Download/clone repository if not already present
$guiScript = Join-Path $RepoRoot 'gui_transcribe.py'
if(-not (Test-Path $guiScript)){
  Write-Info "`nDownloading AudioProcessor repository..."
  
  # Try git clone first, fallback to zip download
  if(Test-Command 'git'){
    try{
      Write-Info "Cloning repository with git..."
      & git clone https://github.com/Rob142857/AudioProcessorAlphaVersion.git $RepoRoot 2>&1 | Out-Null
      if(-not (Test-Path $guiScript)){
        # Git might have created a subdirectory
        $subDir = Join-Path $RepoRoot 'AudioProcessorAlphaVersion'
        if(Test-Path $subDir){
          Get-ChildItem $subDir | Move-Item -Destination $RepoRoot -Force
          Remove-Item $subDir -Force
        }
      }
    } catch {
      Write-Warn "Git clone failed: $_"
    }
  }
  
  # Fallback to zip download if git failed or not available
  if(-not (Test-Path $guiScript)){
    Write-Info "Downloading repository as zip..."
    $zipPath = Join-Path $RepoRoot 'repo.zip'
    try{
      Invoke-WebRequest -Uri 'https://github.com/Rob142857/AudioProcessorAlphaVersion/archive/refs/heads/main.zip' -OutFile $zipPath
      Expand-Archive -Path $zipPath -DestinationPath $RepoRoot -Force
      
      # Move files from subdirectory to root
      $extractedDir = Join-Path $RepoRoot 'AudioProcessorAlphaVersion-main'
      if(Test-Path $extractedDir){
        Get-ChildItem $extractedDir | Move-Item -Destination $RepoRoot -Force
        Remove-Item $extractedDir -Force
      }
      Remove-Item $zipPath -Force
    } catch {
      Write-Err "Failed to download repository: $_"
      exit 1
    }
  }
  
  if(Test-Path $guiScript){
    Write-Ok "Repository downloaded successfully"
  } else {
    Write-Err "Failed to download repository files"
    exit 1
  }
} else {
  Write-Info "Repository files already present"
}

# Helper: run process and throw on error
function Invoke-Checked([string]$Cmd, [string[]]$ArgList){
  if(-not $Quiet){ Write-Host "> $Cmd $($ArgList -join ' ')" -ForegroundColor DarkGray }
  $p = Start-Process -FilePath $Cmd -ArgumentList $ArgList -NoNewWindow -PassThru -Wait
  if($p.ExitCode -ne 0){ throw "Command failed: $Cmd (exit code $($p.ExitCode))" }
}

# Check for winget
$HasWinget = Test-Command 'winget'
if(-not $HasWinget){
  Write-Warn "winget not found. Please install App Installer from Microsoft Store or install Python/ffmpeg manually."
}

# 1) Ensure Python 3.11+
Write-Info "`n[1/7] Checking Python 3.11..."
$PythonCmd = $null
foreach($cmd in @('python','py')){
  if(Test-Command $cmd){
    try{
      $verStr = & $cmd --version 2>&1 | Select-String '3\.(1[1-9]|[2-9]\d)'
      if($verStr){ $PythonCmd = $cmd; break }
    } catch {}
  }
}

if(-not $PythonCmd -and $HasWinget){
  Write-Info "Installing Python 3.11 via winget..."
  try{
    Invoke-Checked 'winget' @('install','-e','--id','Python.Python.3.11','--silent','--accept-package-agreements','--accept-source-agreements')
    $env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')
    if(Test-Command 'python'){ $PythonCmd = 'python' }
  } catch { Write-Warn "Python install failed: $_" }
}

if(-not $PythonCmd){ $PythonCmd = 'python' } # Try anyway

try{
  $pyVer = & $PythonCmd --version 2>&1
  Write-Ok "Python: $pyVer"
} catch {
  Write-Err "Python not found or not responding. Please install Python 3.11+ and try again."
  exit 1
}

# 2) Create virtual environment
Write-Info "`n[2/7] Setting up virtual environment..."
$VenvDir = Join-Path $RepoRoot '.venv'
if(-not (Test-Path $VenvDir)){
  & $PythonCmd -m venv $VenvDir
}

$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
if(-not (Test-Path $VenvPython)){
  Write-Err "Failed to create virtual environment."
  exit 1
}

# Upgrade pip
Invoke-Checked $VenvPython @('-m','pip','install','--upgrade','pip','setuptools','wheel','--quiet')

# 3) Detect GeForce GPU and determine optimal CUDA version
Write-Info "`n[3/7] Detecting NVIDIA GeForce GPU..."

function Get-GeForceInfo(){
  try{
    # Try nvidia-smi first (most reliable)
    if(Test-Command 'nvidia-smi'){
      $output = & nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader 2>$null
      if($output){
        $parts = $output -split ','
        return @{
          Name = $parts[0].Trim()
          Driver = $parts[1].Trim()
          CudaVersion = $parts[2].Trim()
          Found = $true
        }
      }
    }
    
    # Fallback to WMI
    $gpu = Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object { $_.Name -match 'NVIDIA|GeForce' } | Select-Object -First 1
    if($gpu){
      return @{
        Name = $gpu.Name
        Driver = $gpu.DriverVersion
        CudaVersion = 'Unknown'
        Found = $true
      }
    }
  } catch {}
  
  return @{ Found = $false }
}

$gpuInfo = Get-GeForceInfo

if($gpuInfo.Found){
  Write-Ok "Detected: $($gpuInfo.Name)"
  if($gpuInfo.Driver){ Write-Host "  Driver: $($gpuInfo.Driver)" -ForegroundColor Gray }
  if($gpuInfo.CudaVersion -ne 'Unknown'){ Write-Host "  CUDA: $($gpuInfo.CudaVersion)" -ForegroundColor Gray }
  
  # Determine optimal PyTorch CUDA version
  # CUDA 12.1 is recommended for most modern GeForce GPUs (RTX 20xx/30xx/40xx, GTX 16xx)
  # CUDA 11.8 for older cards (GTX 10xx series)
  $cudaIndex = 'cu121' # Default to CUDA 12.1
  
  # Check if older GPU (GTX 10xx series) - use CUDA 11.8
  if($gpuInfo.Name -match 'GTX 10\d{2}'){
    $cudaIndex = 'cu118'
    Write-Info "Detected GTX 10-series GPU, using CUDA 11.8 build"
  } else {
    Write-Info "Using CUDA 12.1 build (recommended for RTX/modern GPUs)"
  }
} else {
  Write-Warn "No NVIDIA GPU detected. Falling back to CPU-only PyTorch."
  $cudaIndex = 'cpu'
}

# 4) Install PyTorch with detected CUDA version
Write-Info "`n[4/7] Installing PyTorch ($cudaIndex)..."
if($cudaIndex -eq 'cpu'){
  Invoke-Checked $VenvPython @('-m','pip','install','torch','torchvision','torchaudio','--index-url','https://download.pytorch.org/whl/cpu')
} else {
  try{
    Invoke-Checked $VenvPython @('-m','pip','install','torch','torchvision','torchaudio','--index-url',"https://download.pytorch.org/whl/$cudaIndex")
  } catch {
    Write-Warn "CUDA PyTorch install failed, falling back to CPU..."
    Invoke-Checked $VenvPython @('-m','pip','install','torch','torchvision','torchaudio','--index-url','https://download.pytorch.org/whl/cpu')
  }
}

# 5) Install project requirements
Write-Info "`n[5/7] Installing dependencies..."
$reqFile = Join-Path $RepoRoot 'requirements.txt'
if(Test-Path $reqFile){
  Invoke-Checked $VenvPython @('-m','pip','install','-r',$reqFile,'--quiet')
} else {
  Write-Warn "requirements.txt not found, installing core packages..."
  Invoke-Checked $VenvPython @('-m','pip','install','openai-whisper','python-docx','deepmultilingualpunctuation','psutil','tqdm','--quiet')
}

# 6) Ensure ffmpeg
Write-Info "`n[6/7] Checking ffmpeg..."
if(Test-Command 'ffmpeg'){
  Write-Ok "ffmpeg found on PATH"
} else {
  $bundled = Join-Path $RepoRoot 'ffmpeg.exe'
  if(Test-Path $bundled){
    Write-Info "Using bundled ffmpeg.exe"
  } elseif($HasWinget) {
    Write-Info "Installing ffmpeg via winget..."
    try{
      Invoke-Checked 'winget' @('install','-e','--id','Gyan.FFmpeg','--silent','--accept-package-agreements','--accept-source-agreements')
      $env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')
    } catch {
      Write-Warn "ffmpeg install failed. Please install manually: https://ffmpeg.org/download.html"
    }
  } else {
    Write-Warn "ffmpeg not found. Please install manually: https://ffmpeg.org/download.html"
  }
}

# 7) Preload Whisper models (optional)
if(-not $SkipModelPreload){
  Write-Info "`n[7/7] Preloading Whisper models (large-v3 and large-v3-turbo)..."
  Write-Info "This may take several minutes depending on your connection..."
  $preloadScript = Join-Path $RepoRoot 'preload_models.py'
  if(Test-Path $preloadScript){
    try{
      Invoke-Checked $VenvPython @($preloadScript)
    } catch {
      Write-Warn "Model preload failed (will download on first use): $_"
    }
  } else {
    Write-Warn "preload_models.py not found, skipping model download"
  }
} else {
  Write-Info "`n[7/7] Skipping model preload (models will download on first use)"
}

# Installation complete!
Write-Host "`n" -NoNewline
Write-Ok "================================================================"
Write-Ok "   Installation Complete! AudioProcessor v1.02beta"
Write-Ok "================================================================"

# Show system info
try{
  $pyVersion = & $VenvPython -c "import sys;print(f'Python {sys.version.split()[0]}')" 2>$null
  $torchInfo = & $VenvPython -c "import torch;print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')" 2>$null
  Write-Host "`n  $pyVersion" -ForegroundColor Gray
  Write-Host "  $torchInfo" -ForegroundColor Gray
  if($gpuInfo.Found){
    Write-Host "  GPU: $($gpuInfo.Name)" -ForegroundColor Gray
  }
} catch {}

Write-Host "`n  Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Activate venv:  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  2. Launch GUI:     python gui_transcribe.py" -ForegroundColor White
Write-Host "  3. Or headless:    python gui_transcribe.py --input `"file.mp4`"`n" -ForegroundColor White

# Launch GUI automatically
if(-not $SkipGUI){
  Write-Info "Launching GUI in 3 seconds... (Press Ctrl+C to cancel)"
  Start-Sleep -Seconds 3
  try{
    & $VenvPython (Join-Path $RepoRoot 'gui_transcribe.py')
  } catch {
    Write-Warn "Failed to launch GUI: $_"
  }
}

exit 0
