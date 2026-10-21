@echo off
setlocal
cd /d "%~dp0.."
python e8_windows\run_e8.py train --arm A_baseline
if errorlevel 1 exit /b 1
endlocal
