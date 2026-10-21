@echo off
setlocal
cd /d "%~dp0.."
python e6_windows\run_e6.py apply-test --config "runs\e6_selection_seed42\selection_config.json" --output "runs\e6_final_seed42"
if errorlevel 1 exit /b 1
endlocal
