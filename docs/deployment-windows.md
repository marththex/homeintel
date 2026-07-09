# Advanced: Auto-Start on Windows Login + NAS Share Mount (optional)

> This is one optional deployment recipe, not the primary way to run HomeIntel.
> Most users should just use `setup.ps1` once and `run.ps1` (or `run.sh`) each
> time they want the app running — see the main [README](../README.md) Quick
> Start. Read this page only if you want HomeIntel to run unattended as a
> background service on a Windows machine, watching a folder that lives on a
> NAS/SMB share rather than local disk.

This guide covers two independent things you can combine or use separately:

1. Mounting a network (SMB) share as a drive letter so HomeIntel can index it.
2. Registering the backend and frontend as Windows scheduled tasks that start
   automatically at login (and restart if they crash), instead of running
   `run.ps1` manually in a terminal.

Placeholders used below — replace with your own values:

| Placeholder | Meaning | Example |
|---|---|---|
| `<REPO_PATH>` | Absolute path where you cloned this repo | `C:\Users\you\homeintel` |
| `<NAS_IP>` | IP or hostname of your NAS / file server | `192.168.1.50` |
| `<NAS_SHARE>` | Name of the SMB share to mount | `MyShare` |
| `<DRIVE_LETTER>` | Drive letter to map the share to | `Z:` |
| `<VENV_OR_CONDA_ACTIVATE>` | Command that activates your Python environment | `<REPO_PATH>\.venv\Scripts\activate.bat` (venv) or `C:\Users\you\miniconda3\Scripts\activate.bat homeintel` (conda) |

---

## 1. Mount the NAS share as a drive letter

If the folder you want to index lives on a NAS or another machine on your LAN,
mount it over SMB before pointing `NAS_WATCH_PATH` at it:

```powershell
net use <DRIVE_LETTER> \\<NAS_IP>\<NAS_SHARE> /persistent:yes
```

Confirm it mounted:

```powershell
dir <DRIVE_LETTER>\
```

Then set in `.env`:

```dotenv
NAS_WATCH_PATH=<DRIVE_LETTER>/
```

**Gotcha:** a share mounted this way from your interactive login session is
**not visible** to tasks that run in Windows session 0 (which is where Task
Scheduler runs things when "Run whether user is logged on or not" is used).
If you plan to auto-start via Task Scheduler (section 2), the `net use`
command needs to run *as part of the scheduled task's own startup script*,
not just once in your terminal — see the sample script below.

---

## 2. Auto-start via Windows Task Scheduler

The idea: two scheduled tasks, `HomeIntel-API` and `HomeIntel-UI`, each
running a small wrapper script at user login. The backend wrapper remaps the
NAS drive (cheap and safe to repeat) before starting uvicorn, so it works
whether or not the task runs in session 0.

### 2a. Backend startup script

Save this as e.g. `<REPO_PATH>\scripts\start_backend.ps1` (not part of the
repo — this is a local file you create for your own deployment):

```powershell
$LogFile = "<REPO_PATH>\logs\api.log"
$NasIp = "<NAS_IP>"
$NasShare = "<NAS_SHARE>"
$DriveLetter = "<DRIVE_LETTER>"

"[$(Get-Date)] Backend starting" | Out-File -Append $LogFile

# Map the drive (safe to run even if already mapped).
net use $DriveLetter "\\$NasIp\$NasShare" /persistent:no *>> $LogFile

# Wait for the mount to actually be available.
while (-not (Test-Path "$DriveLetter\")) {
    "[$(Get-Date)] Waiting for $DriveLetter\" | Out-File -Append $LogFile
    Start-Sleep -Seconds 5
}

"[$(Get-Date)] $DriveLetter\ ready, starting uvicorn" | Out-File -Append $LogFile

# Activate the environment, then run the backend.
# venv:
& "<REPO_PATH>\.venv\Scripts\Activate.ps1"
# conda (alternative):
# & <VENV_OR_CONDA_ACTIVATE>

Set-Location "<REPO_PATH>\backend"
uvicorn main:app --host 0.0.0.0 --port 8000 *>> $LogFile
```

### 2b. Frontend startup script

Save as `<REPO_PATH>\scripts\start_frontend.ps1`:

```powershell
Set-Location "<REPO_PATH>\frontend"
npm run dev
```

(In a production-style deployment you'd more likely run `npm run build` once
and serve the static `dist/` output, or let the backend serve it — see
`docs/ARCHITECTURE.md`. `npm run dev` is the simplest option to mirror the
original setup.)

### 2c. Register the scheduled tasks

Run once, from an elevated PowerShell prompt, to register both tasks to run
at login as the current user:

```powershell
$action1 = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"<REPO_PATH>\scripts\start_backend.ps1`""
$trigger1 = New-ScheduledTaskTrigger -AtLogOn
Register-ScheduledTask -TaskName "HomeIntel-API" -Action $action1 -Trigger $trigger1 -RunLevel Highest

$action2 = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"<REPO_PATH>\scripts\start_frontend.ps1`""
$trigger2 = New-ScheduledTaskTrigger -AtLogOn
Register-ScheduledTask -TaskName "HomeIntel-UI" -Action $action2 -Trigger $trigger2 -RunLevel Highest
```

### 2d. Manage the tasks

```powershell
# Start / stop manually
Start-ScheduledTask "HomeIntel-API"
Stop-ScheduledTask "HomeIntel-API"
Start-ScheduledTask "HomeIntel-UI"
Stop-ScheduledTask "HomeIntel-UI"

# View backend logs
Get-Content "<REPO_PATH>\logs\api.log" -Wait

# Unregister if you want to go back to running things manually
Unregister-ScheduledTask -TaskName "HomeIntel-API" -Confirm:$false
Unregister-ScheduledTask -TaskName "HomeIntel-UI" -Confirm:$false
```

---

## Notes

- This whole setup is optional. If you're just trying HomeIntel out, run
  `setup.ps1` once and `run.ps1` whenever you want it up — no scheduled tasks,
  no drive mapping required, and it can index any local folder.
- The `net use` re-map inside the backend wrapper script is intentional and
  cheap: it's the workaround for the session-0 visibility gotcha above, and
  it's harmless to run against an already-mounted drive.
- If your GPU is busy or you want to free VRAM, stop the tasks/services and
  Ollama independently — they're not tied together by this setup.
