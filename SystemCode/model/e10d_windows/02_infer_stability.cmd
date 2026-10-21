@echo off
setlocal
cd /d "%~dp0.."
set "PYTHONPATH=%CD%;%PYTHONPATH%"
python e10d_windows\run_e10d.py infer
if errorlevel 1 exit /b 1
endlocal
