@echo off
setlocal
cd /d "%~dp0.."
python e10c_windows\test_e10c.py
if errorlevel 1 exit /b 1
python e10c_windows\run_e10c.py freeze
if errorlevel 1 exit /b 1
endlocal
