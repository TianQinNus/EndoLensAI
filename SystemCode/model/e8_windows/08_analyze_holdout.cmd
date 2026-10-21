@echo off
setlocal
cd /d "%~dp0.."
python e8_windows\test_holdout_analysis.py
if errorlevel 1 exit /b 1
if exist "runs\e8_holdout_analysis_seed42\verification.json" goto verify
python e8_windows\analyze_holdout.py analyze
if errorlevel 1 exit /b 1
:verify
python e8_windows\analyze_holdout.py verify
if errorlevel 1 exit /b 1
echo CACHED HOLDOUT ANALYSIS VERIFIED. NO INFERENCE OR FUSION SEARCH WAS PERFORMED.
endlocal
