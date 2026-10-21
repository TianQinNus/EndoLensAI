@echo off
setlocal
cd /d "%~dp0.."
python e10a_windows\run_e10a.py infer
if errorlevel 1 exit /b 1
endlocal
