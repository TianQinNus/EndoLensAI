@echo off
setlocal
cd /d "%~dp0.."
python e5_windows\test_e5.py
if errorlevel 1 exit /b 1
python e5_windows\run_e5.py --audit-only --output "runs\e5_audit_seed42"
if errorlevel 1 exit /b 1
python e5_windows\run_e5.py --smoke --output "runs\e5_smoke_seed42"
if errorlevel 1 exit /b 1
endlocal
