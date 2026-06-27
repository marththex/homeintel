@echo off
call C:\Users\admin\miniconda3\Scripts\activate.bat homeintel
cd /d C:\Users\admin\Documents\homeintel\backend
uvicorn main:app --host 0.0.0.0 --port 8000
