@echo off
setlocal
cd /d "%~dp0.."
python e7a2_windows\run_e7a2.py apply-test
if errorlevel 1 exit /b 1
endlocal
