<#
.SYNOPSIS
    HomeIntel setup script (Windows PowerShell).
.DESCRIPTION
    Creates a Python virtualenv at .venv, installs PyTorch (CPU by default,
    CUDA cu128 with -Gpu), installs backend Python deps, installs frontend
    npm deps, and creates .env from .env.example if missing.

    Safe to re-run: an existing .venv / .env are left alone, npm install is
    naturally idempotent.
.PARAMETER Gpu
    Install the CUDA (cu128) build of PyTorch instead of the CPU build.
    Only meaningful with an NVIDIA GPU + recent drivers. macOS has no CUDA
    support at all — use setup.sh there (always installs the CPU/MPS build).
.EXAMPLE
    .\setup.ps1
.EXAMPLE
    .\setup.ps1 -Gpu
#>
[CmdletBinding()]
param(
    [switch]$Gpu
)

$ErrorActionPreference = 'Stop'

$ScriptDir = $PSScriptRoot
Set-Location $ScriptDir

function Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Assert-LastExitCode {
    param([string]$What)
    if ($LASTEXITCODE -ne 0) {
        throw "$What failed with exit code $LASTEXITCODE"
    }
}

# ── 1. Locate Python 3.11+ ────────────────────────────────────────────────────
Step "Checking for Python 3.11+"

$PythonExe = $null
$PythonArgs = @()

# Prefer the Windows "py" launcher pinned to 3.11, then fall back to whatever
# "python" / "python3" resolves to on PATH.
$candidates = @(
    @{ Cmd = 'py'; Args = @('-3.11') },
    @{ Cmd = 'python'; Args = @() },
    @{ Cmd = 'python3'; Args = @() }
)

foreach ($c in $candidates) {
    $cmd = Get-Command $c.Cmd -ErrorAction SilentlyContinue
    if (-not $cmd) { continue }

    $verOutput = & $c.Cmd @($c.Args) -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $verOutput) { continue }

    $parts = $verOutput.Trim().Split('.')
    if ($parts.Count -lt 2) { continue }
    $major = [int]$parts[0]
    $minor = [int]$parts[1]
    if ($major -eq 3 -and $minor -ge 11) {
        $PythonExe = $c.Cmd
        $PythonArgs = $c.Args
        break
    }
}

if (-not $PythonExe) {
    throw "Python 3.11+ not found on PATH (tried 'py -3.11', 'python', 'python3'). Install it from https://www.python.org/downloads/"
}

$resolvedVer = & $PythonExe @PythonArgs -c "import sys; print(sys.version.split()[0])"
Write-Host "Using Python $resolvedVer ($PythonExe $($PythonArgs -join ' '))"

# ── 2. Create virtualenv ──────────────────────────────────────────────────────
$VenvDir = Join-Path $ScriptDir ".venv"
if (Test-Path $VenvDir) {
    Step "Virtualenv already exists at .venv (skipping creation)"
} else {
    Step "Creating virtualenv at .venv"
    & $PythonExe @PythonArgs -m venv $VenvDir
    Assert-LastExitCode "venv creation"
}

$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    throw "Could not find venv activate script at $ActivateScript"
}
. $ActivateScript

# ── 3. Upgrade pip ────────────────────────────────────────────────────────────
Step "Upgrading pip"
python -m pip install --upgrade pip
Assert-LastExitCode "pip upgrade"

# ── 4. Install PyTorch (CPU by default, CUDA cu128 with -Gpu) ─────────────────
if ($Gpu) {
    Step "Installing PyTorch (CUDA cu128 build - NVIDIA GPU)"
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    Assert-LastExitCode "PyTorch (CUDA) install"
} else {
    Step "Installing PyTorch (CPU build)"
    Write-Host "NOTE: on Windows with an NVIDIA GPU, re-run as '.\setup.ps1 -Gpu' for the CUDA build."
    Write-Host "NOTE: macOS has no CUDA support - always uses the CPU/MPS build (use setup.sh there)."
    python -m pip install torch torchvision
    Assert-LastExitCode "PyTorch (CPU) install"
}

# ── 5. Backend dependencies ───────────────────────────────────────────────────
Step "Installing backend dependencies (backend/requirements.txt)"
python -m pip install -r (Join-Path $ScriptDir "backend\requirements.txt")
Assert-LastExitCode "backend requirements install"

$devReq = Join-Path $ScriptDir "backend\requirements-dev.txt"
if (Test-Path $devReq) {
    Write-Host "NOTE: backend/requirements-dev.txt is available for contributors (pytest, lint, etc.)."
    Write-Host "      Install it with: pip install -r backend/requirements-dev.txt"
}

# ── 6. Frontend dependencies ──────────────────────────────────────────────────
Step "Checking for Node.js / npm"
$node = Get-Command node -ErrorAction SilentlyContinue
$npm = Get-Command npm -ErrorAction SilentlyContinue
if (-not $node -or -not $npm) {
    throw "node and/or npm not found on PATH. Install Node.js (includes npm): https://nodejs.org/"
}
Write-Host "Using node $(node --version), npm $(npm --version)"

Step "Installing frontend dependencies (npm install)"
Push-Location (Join-Path $ScriptDir "frontend")
try {
    npm install
    Assert-LastExitCode "npm install"
} finally {
    Pop-Location
}

# ── 7. .env file ──────────────────────────────────────────────────────────────
Step "Checking for .env"
$envFile = Join-Path $ScriptDir ".env"
$envExample = Join-Path $ScriptDir ".env.example"
if (Test-Path $envFile) {
    Write-Host ".env already exists - leaving it untouched."
} else {
    Copy-Item $envExample $envFile
    Write-Host "Created .env from .env.example - review and edit it before running the app."
}

Step "Setup complete"
Write-Host "Next steps:"
Write-Host "  1. Review/edit .env (NAS_WATCH_PATH, QDRANT_URL, OLLAMA_* as needed)."
Write-Host "  2. Make sure Ollama and Qdrant are reachable."
Write-Host "  3. Start the app: .\run.ps1"
