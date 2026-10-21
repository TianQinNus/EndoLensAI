@echo off
setlocal
cd /d "%~dp0.."
python e10c_windows\run_e10c.py evaluate
if errorlevel 1 exit /b 1
endlocal
