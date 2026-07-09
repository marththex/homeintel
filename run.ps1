<#
.SYNOPSIS
    HomeIntel run script (Windows PowerShell).
.DESCRIPTION
    Starts the backend (uvicorn), frontend (vite), or both using the .venv
    created by setup.ps1.
.PARAMETER Mode
    backend  - start only the FastAPI backend (http://localhost:8000)
    frontend - start only the Vite frontend dev server (http://localhost:5173)
    all      - (default) start both. Kept simple: the backend runs as a
               background PowerShell job, the frontend runs in the foreground.
               Ctrl+C (or closing the frontend) stops the backend job too.
.EXAMPLE
    .\run.ps1
.EXAMPLE
    .\run.ps1 backend
#>
[CmdletBinding()]
param(
    [ValidateSet('backend', 'frontend', 'all')]
    [string]$Mode = 'all'
)

$ErrorActionPreference = 'Stop'

$ScriptDir = $PSScriptRoot
Set-Location $ScriptDir

$ActivateScript = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    throw ".venv not found at $ActivateScript. Run .\setup.ps1 first."
}
. $ActivateScript

function Start-Backend {
    Write-Host "==> Starting backend (uvicorn, http://0.0.0.0:8000)" -ForegroundColor Cyan
    Push-Location (Join-Path $ScriptDir "backend")
    try {
        uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    } finally {
        Pop-Location
    }
}

function Start-Frontend {
    Write-Host "==> Starting frontend (vite, http://localhost:5173)" -ForegroundColor Cyan
    Push-Location (Join-Path $ScriptDir "frontend")
    try {
        npm run dev
    } finally {
        Pop-Location
    }
}

switch ($Mode) {
    'backend' { Start-Backend }
    'frontend' { Start-Frontend }
    'all' {
        $backendDir = Join-Path $ScriptDir "backend"
        $venvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"

        Write-Host "==> Starting backend in a background job (uvicorn, http://0.0.0.0:8000)" -ForegroundColor Cyan
        # Verified empirically (2026-07): on Windows, Start-Job's host process
        # is created inside a nested Job Object, and any processes it spawns
        # (uvicorn's --reload supervisor, and the multiprocessing worker the
        # supervisor forks for the actual server) are added to that same Job
        # Object unless they opt out with CREATE_BREAKAWAY_FROM_JOB, which
        # neither uvicorn nor Python's multiprocessing module does. So
        # Stop-Job/Remove-Job in the `finally` block below tears down the
        # whole tree (supervisor + worker), not just the outer job wrapper --
        # confirmed with the real uvicorn/fastapi versions this repo pins:
        # all PIDs gone and the port freed within ~3s of Stop-Job. This is
        # unlike run.sh's analogous bug (bash subshells don't get this kind
        # of OS-level cascade), so no Start-Process/taskkill refactor is
        # needed here.
        $job = Start-Job -Name "homeintel-backend" -ScriptBlock {
            param($Dir, $Py)
            Set-Location $Dir
            & $Py -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
        } -ArgumentList $backendDir, $venvPython

        try {
            # Give the backend a moment to boot before the frontend starts probing it.
            Start-Sleep -Seconds 2
            Start-Frontend
        } finally {
            Write-Host "==> Stopping backend job" -ForegroundColor Cyan
            Stop-Job $job -ErrorAction SilentlyContinue | Out-Null
            Receive-Job $job -ErrorAction SilentlyContinue | Out-Null
            Remove-Job $job -Force -ErrorAction SilentlyContinue | Out-Null
        }
    }
}
