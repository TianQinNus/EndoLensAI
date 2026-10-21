@echo off
setlocal
cd /d "%~dp0.."
set "PYTHONPATH=%CD%;%PYTHONPATH%"
python e10d_windows\test_e10d.py
if errorlevel 1 exit /b 1
python e10d_windows\run_e10d.py freeze
if errorlevel 1 exit /b 1
endlocal
