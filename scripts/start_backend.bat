@echo off
set LOGFILE=C:\Users\admin\Documents\homeintel\logs\api.log
set NAS_HOST=172.16.0.100
set NAS_SHARE=NFSdocker
echo [%DATE% %TIME%] Backend starting >> %LOGFILE%

:: Map Z: drive (safe to run even if already mapped)
net use Z: \\%NAS_HOST%\%NAS_SHARE% /persistent:no >> %LOGFILE% 2>&1

:: Wait for mount to be available
:waitnas
if not exist "Z:\" (
    echo [%DATE% %TIME%] Waiting for Z:\ >> %LOGFILE%
    timeout /t 5 /nobreak >nul
    goto waitnas
)

echo [%DATE% %TIME%] Z:\ ready, starting uvicorn >> %LOGFILE%
call C:\Users\admin\miniconda3\Scripts\activate.bat homeintel >> %LOGFILE% 2>&1
cd /d C:\Users\admin\Documents\homeintel\backend
uvicorn main:app --host 0.0.0.0 --port 8000 >> %LOGFILE% 2>&1
