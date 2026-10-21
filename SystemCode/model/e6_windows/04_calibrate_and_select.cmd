@echo off
setlocal
cd /d "%~dp0.."
python e6_windows\run_e6.py select --manifest-run "runs\e6_development_seed42" --inference-run "runs\e6_development_inference_seed42" --output "runs\e6_selection_seed42"
if errorlevel 1 exit /b 1
endlocal
