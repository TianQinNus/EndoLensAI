@echo off
setlocal
cd /d "%~dp0.."
set "PYTHONPATH=%CD%;%PYTHONPATH%"
python e11_background_windows\run_e11.py summarize
if errorlevel 1 exit /b 1
endlocal
