@echo off
setlocal
cd /d "%~dp0.."
python e8_repeat_windows\test_e8_repeat.py
if errorlevel 1 exit /b 1
if exist "runs\e8_repeat_plan.json" goto check
python e8_repeat_windows\run_e8_repeat.py prepare
if errorlevel 1 exit /b 1
:check
python e8_repeat_windows\run_e8_repeat.py status
if errorlevel 1 exit /b 1
echo E8-R READY. NEW FORMAL TRAINING WAS NOT STARTED BY THIS CHECK.
endlocal
