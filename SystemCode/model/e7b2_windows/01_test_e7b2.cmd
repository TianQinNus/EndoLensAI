@echo off
setlocal
cd /d "%~dp0.."
python e7b2_windows\test_e7b2.py
if errorlevel 1 exit /b 1
endlocal
