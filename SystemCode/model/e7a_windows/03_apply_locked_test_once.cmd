@echo off
setlocal
cd /d "%~dp0.."
python e7a_windows\run_e7a.py apply-test ^
  --config "runs\e7a_selection_seed42\selection_config.json" ^
  --output "runs\e7a_final_seed42"
if errorlevel 1 exit /b 1
endlocal
