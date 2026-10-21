@echo off
setlocal
cd /d "%~dp0.."
python e7a_windows\test_e7a.py
if errorlevel 1 exit /b 1
endlocal
