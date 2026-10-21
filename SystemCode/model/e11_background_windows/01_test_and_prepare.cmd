@echo off
setlocal
cd /d "%~dp0.."
set "PYTHONPATH=%CD%;%PYTHONPATH%"
python e11_background_windows\test_e11.py
if errorlevel 1 exit /b 1
python e11_background_windows\run_e11.py prepare
if errorlevel 1 exit /b 1
endlocal
