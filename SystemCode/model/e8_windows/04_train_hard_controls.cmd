@echo off
setlocal
cd /d "%~dp0.."
python e8_windows\run_e8.py train --arm C_hard
if errorlevel 1 exit /b 1
python e8_windows\run_e8.py train --arm D_expanded_hard
if errorlevel 1 exit /b 1
endlocal
