@echo off
setlocal
cd /d "%~dp0.."
python e5_windows\run_e5.py --output "runs\e5_fusion_equal_seed42"
if errorlevel 1 exit /b 1
endlocal
