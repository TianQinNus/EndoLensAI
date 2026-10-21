@echo off
setlocal
cd /d "%~dp0.."
if exist "runs\e8_repeat_summary\audit.json" goto verify
python e8_repeat_windows\run_e8_repeat.py summarize
if errorlevel 1 exit /b 1
:verify
python e8_repeat_windows\run_e8_repeat.py verify
if errorlevel 1 exit /b 1
endlocal
