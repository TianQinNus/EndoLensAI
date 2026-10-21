@echo off
setlocal
cd /d "%~dp0.."
python e7b1_windows\test_e7b1.py
if errorlevel 1 exit /b 1
endlocal
