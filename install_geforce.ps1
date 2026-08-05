<#
.SYNOPSIS
    Pinned local setup for AudioProcessor on Windows x64 + GTX 1070 Ti.

.DESCRIPTION
    Run this reviewed script from a local AudioProcessor checkout. It creates a
    Python 3.12 x64 virtual environment, installs the proven Pascal-compatible
    torch 2.6.0+cu124 lane before the pinned application requirements, runs
    pip's dependency check, and executes the offline pipeline doctor.

    The script does not clone or update Git repositories, install system
    software, download Whisper models, change credentials, or start the GUI
    unless -Launch is supplied.

.PARAMETER RecreateVenv
    Remove and recreate only this checkout's .venv when it is incompatible.

.PARAMETER Launch
    Launch gui_transcribe.py after every setup check passes.

.EXAMPLE
    .\install_geforce.ps1

.EXAMPLE
    .\install_geforce.ps1 -RecreateVenv -Launch

.NOTES
    Prerequisites: Windows 10/11 x64, Python 3.12 x64, an NVIDIA driver,
    internet access for Python packages, and the bundled ffmpeg.exe in this
    checkout. Python may be registered with py.exe or installed in the normal
    per-user Python312 directory.
#>

[CmdletBinding()]
param(
    [switch]$RecreateVenv,
    [switch]$Launch
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$PinnedPython = "3.12"
$PinnedTorch = "2.6.0+cu124"
$TorchIndex = "https://download.pytorch.org/whl/cu124"
$PinnedPip = "26.2.1"
$PinnedSetuptools = "83.0.0"
$PinnedWheel = "0.47.0"

$ProjectRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    throw "Run install_geforce.ps1 from a local AudioProcessor checkout."
}
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"
$DoctorPath = Join-Path $ProjectRoot "pipeline_doctor.py"
$GuiPath = Join-Path $ProjectRoot "gui_transcribe.py"
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

function Write-Step {
    param([Parameter(Mandatory)][string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([Parameter(Mandatory)][string]$Message)
    Write-Host "    OK: $Message" -ForegroundColor Green
}

function Write-Warn {
    param([Parameter(Mandatory)][string]$Message)
    Write-Host "    WARN: $Message" -ForegroundColor Yellow
}

function Invoke-Native {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$ArgumentList = @()
    )
    & $FilePath @ArgumentList
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        $rendered = ($ArgumentList -join " ")
        throw "Native command failed ($exitCode): $FilePath $rendered"
    }
}

function Get-PythonProbe {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$PrefixArguments = @()
    )
    $probe = @(
        "-c",
        "import platform,struct; print(f'{platform.python_version()}|{struct.calcsize(`"P`") * 8}')"
    )
    return (Invoke-Native -FilePath $FilePath -ArgumentList ($PrefixArguments + $probe) | Select-Object -Last 1)
}

function Remove-LocalVenv {
    $expectedVenv = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot ".venv"))
    $resolvedVenv = [System.IO.Path]::GetFullPath($VenvDir)
    if ($resolvedVenv -ne $expectedVenv -or (Split-Path -Parent $resolvedVenv) -ne $ProjectRoot) {
        throw "Refusing to remove unexpected virtual environment path: $resolvedVenv"
    }
    $venvEntry = Get-Item -LiteralPath $VenvDir -Force
    if (($venvEntry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing to recursively remove a .venv reparse point: $resolvedVenv"
    }
    Write-Warn "Removing incompatible or incomplete $resolvedVenv"
    Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
}

if ([System.Environment]::OSVersion.Platform -ne [System.PlatformID]::Win32NT) {
    throw "This installer is for Windows only."
}
if (-not [System.Environment]::Is64BitOperatingSystem) {
    throw "AudioProcessor requires 64-bit Windows."
}
foreach ($requiredFile in @($RequirementsPath, $DoctorPath, $GuiPath)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required checkout file is missing: $requiredFile"
    }
}

Write-Step "Locating Python $PinnedPython x64"
$launcher = Get-Command "py.exe" -CommandType Application -ErrorAction SilentlyContinue
$BasePythonFilePath = $null
$BasePythonPrefixArguments = @()
$baseProbe = $null

if ($null -ne $launcher) {
    try {
        $candidateProbe = Get-PythonProbe -FilePath $launcher.Source -PrefixArguments @("-$PinnedPython")
        if ($candidateProbe -match '^3\.12\.\d+\|64$') {
            $BasePythonFilePath = $launcher.Source
            $BasePythonPrefixArguments = @("-$PinnedPython")
            $baseProbe = $candidateProbe
            Write-Ok "Found Python through py -$PinnedPython"
        } else {
            Write-Warn "py -$PinnedPython resolved to '$candidateProbe'; trying the per-user installation"
        }
    } catch {
        Write-Warn "py -$PinnedPython is unavailable; trying the per-user installation"
    }
}

$perUserPython = $null
if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
    $perUserPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
}
if ($null -eq $BasePythonFilePath -and $null -ne $perUserPython -and (Test-Path -LiteralPath $perUserPython -PathType Leaf)) {
    try {
        $candidateProbe = Get-PythonProbe -FilePath $perUserPython
        if ($candidateProbe -match '^3\.12\.\d+\|64$') {
            $BasePythonFilePath = $perUserPython
            $BasePythonPrefixArguments = @()
            $baseProbe = $candidateProbe
            Write-Ok "Found per-user Python at $perUserPython"
        }
    } catch {
        Write-Warn "Per-user Python exists but could not be executed: $perUserPython"
    }
}

if ($null -eq $BasePythonFilePath) {
    throw "Python 3.12 x64 was not found through py -3.12 or $perUserPython. Install Python 3.12 x64 from python.org, then rerun this local script."
}
Write-Ok "Python $($baseProbe -replace '\|', ' / ')-bit"

if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
    Write-Step "Validating existing virtual environment"
    $venvCompatible = $false
    try {
        $venvProbe = Get-PythonProbe -FilePath $VenvPython
        $venvCompatible = $venvProbe -match '^3\.12\.\d+\|64$'
    } catch {
        $venvProbe = "unusable"
    }
    if (-not $venvCompatible) {
        if (-not $RecreateVenv) {
            throw ".venv is '$venvProbe'. Rerun with -RecreateVenv to replace only this checkout's environment."
        }
        Remove-LocalVenv
    } else {
        Write-Ok "Existing .venv is Python $($venvProbe -replace '\|', ' / ')-bit"
    }
} elseif ((Test-Path -LiteralPath $VenvDir) -and -not $RecreateVenv) {
    throw ".venv exists but has no usable Python. Rerun with -RecreateVenv."
} elseif (Test-Path -LiteralPath $VenvDir) {
    Remove-LocalVenv
}

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    Write-Step "Creating Python $PinnedPython x64 virtual environment"
    Invoke-Native -FilePath $BasePythonFilePath -ArgumentList ($BasePythonPrefixArguments + @("-m", "venv", $VenvDir))
    Write-Ok "Created $VenvDir"
}

Write-Step "Installing pinned packaging tools"
Invoke-Native -FilePath $VenvPython -ArgumentList @(
    "-m", "pip", "install", "--disable-pip-version-check", "--upgrade",
    "pip==$PinnedPip", "setuptools==$PinnedSetuptools", "wheel==$PinnedWheel"
)
Write-Ok "pip $PinnedPip, setuptools $PinnedSetuptools, wheel $PinnedWheel"

Write-Step "Installing Pascal-compatible PyTorch before Whisper"
Invoke-Native -FilePath $VenvPython -ArgumentList @(
    "-m", "pip", "install", "--disable-pip-version-check", "--upgrade",
    "torch==$PinnedTorch", "--index-url", $TorchIndex
)
Write-Ok "PyTorch $PinnedTorch from the official cu124 index"

Write-Step "Installing pinned AudioProcessor requirements"
Invoke-Native -FilePath $VenvPython -ArgumentList @(
    "-m", "pip", "install", "--disable-pip-version-check", "--upgrade",
    "--requirement", $RequirementsPath
)
Write-Ok "Pinned application requirements installed"

Write-Step "Checking dependency consistency"
Invoke-Native -FilePath $VenvPython -ArgumentList @("-m", "pip", "check")
Write-Ok "No broken Python requirements"

Write-Step "Running offline transcription environment doctor"
Invoke-Native -FilePath $VenvPython -ArgumentList @(
    $DoctorPath, "--mode", "transcribe", "--require-gpu", "--no-cleanup"
)
Write-Ok "Python, package, PyTorch CUDA, and CTranslate2 CUDA checks passed"

Write-Warn "Setup does not download models or prove transcription quality. Run and review a real tape CUDA/int8 canary before archive-wide processing."

if ($Launch) {
    Write-Step "Launching AudioProcessor"
    Invoke-Native -FilePath $VenvPython -ArgumentList @($GuiPath, "--gui")
} else {
    Write-Host ""
    Write-Host "Environment setup complete. Launch with:" -ForegroundColor White
    Write-Host "  & '$VenvPython' '$GuiPath' --gui" -ForegroundColor White
}
