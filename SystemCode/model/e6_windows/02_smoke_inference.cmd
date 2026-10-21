@echo off
setlocal
cd /d "%~dp0.."
python e6_windows\run_e6.py infer --manifest-run "runs\e6_development_seed42" --output "runs\e6_development_smoke_seed42" --device cuda --smoke
if errorlevel 1 exit /b 1
endlocal
