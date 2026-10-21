@echo off
setlocal
cd /d "%~dp0.."
python e7a2_windows\run_e7a2.py select
if errorlevel 1 exit /b 1
endlocal
