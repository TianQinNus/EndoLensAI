@echo off
setlocal
cd /d "%~dp0.."
python e8_windows\run_e8.py lock-holdout --seed 42
if errorlevel 1 exit /b 1
endlocal
