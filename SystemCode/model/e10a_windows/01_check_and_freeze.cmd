@echo off
setlocal
cd /d "%~dp0.."
python e10a_windows\test_e10a.py
if errorlevel 1 exit /b 1
python e10a_windows\run_e10a.py freeze
if errorlevel 1 exit /b 1
endlocal
