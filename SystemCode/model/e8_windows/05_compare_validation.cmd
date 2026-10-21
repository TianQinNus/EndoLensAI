@echo off
setlocal
cd /d "%~dp0.."
python e8_windows\run_e8.py compare --seed 42
if errorlevel 1 exit /b 1
echo VALIDATION COMPARISON COMPLETE. C6 HOLDOUT IS STILL UNEVALUATED.
endlocal
