@echo off
setlocal
cd /d "%~dp0"
python verify_bundle.py --integrity-only
if errorlevel 1 exit /b 1
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --workers 1
if errorlevel 1 exit /b 1
endlocal
