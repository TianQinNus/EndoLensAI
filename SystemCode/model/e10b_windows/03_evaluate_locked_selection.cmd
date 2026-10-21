@echo off
setlocal
cd /d "%~dp0.."
python e10b_windows\run_e10b.py evaluate
if errorlevel 1 exit /b 1
endlocal
