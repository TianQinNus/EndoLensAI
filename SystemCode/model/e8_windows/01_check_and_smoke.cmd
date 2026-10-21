@echo off
setlocal
cd /d "%~dp0.."
python e8_windows\test_e8.py
if errorlevel 1 exit /b 1
python e8_windows\run_e8.py preflight
if errorlevel 1 exit /b 1
python e8_windows\run_e8.py check-gpu
if errorlevel 1 exit /b 1
for %%A in (A_baseline B_expanded C_hard D_expanded_hard) do (
    python e8_windows\run_e8.py smoke --arm %%A
    if errorlevel 1 exit /b 1
)
echo E8 CHECKS PASSED. NO FORMAL TRAINING OR HOLDOUT EVALUATION WAS STARTED.
endlocal
