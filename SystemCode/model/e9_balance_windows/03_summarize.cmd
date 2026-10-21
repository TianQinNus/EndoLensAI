@echo off
setlocal
cd /d "%~dp0.."
python e9_balance_windows\run_e9_balance.py summarize
if errorlevel 1 exit /b 1
python e9_balance_windows\run_e9_balance.py verify
if errorlevel 1 exit /b 1
endlocal
