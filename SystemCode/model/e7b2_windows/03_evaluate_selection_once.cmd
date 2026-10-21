@echo off
setlocal
cd /d "%~dp0.."
python e7b2_windows\run_e7b2.py evaluate
if errorlevel 1 exit /b 1
endlocal
