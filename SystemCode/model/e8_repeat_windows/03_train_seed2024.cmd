@echo off
setlocal
cd /d "%~dp0.."
python e8_repeat_windows\run_e8_repeat.py train-seed --seed 2024
if errorlevel 1 exit /b 1
endlocal
