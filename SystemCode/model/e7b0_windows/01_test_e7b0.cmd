@echo off
setlocal
cd /d "%~dp0.."
python e7b0_windows\test_e7b0.py
if errorlevel 1 exit /b 1
endlocal
