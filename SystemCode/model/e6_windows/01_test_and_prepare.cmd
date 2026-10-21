@echo off
setlocal
cd /d "%~dp0.."
python e6_windows\test_e6.py
if errorlevel 1 exit /b 1
python e6_windows\check_models.py --device cuda
if errorlevel 1 exit /b 1
python e6_windows\run_e6.py prepare --dataset "dataset\E6Development" --metadata "dataset\E6Development\metadata.csv" --output "runs\e6_development_seed42"
if errorlevel 1 exit /b 1
endlocal
