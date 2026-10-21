@echo off
setlocal
cd /d "%~dp0.."
python e7a2_windows\test_e7a2.py
if errorlevel 1 exit /b 1
endlocal
