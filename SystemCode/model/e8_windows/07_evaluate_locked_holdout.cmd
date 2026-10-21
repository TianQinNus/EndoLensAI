@echo off
setlocal
cd /d "%~dp0.."
python e8_windows\run_e8.py evaluate-holdout
if errorlevel 1 exit /b 1
endlocal
