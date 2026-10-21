@echo off
setlocal
cd /d "%~dp0.."
if exist "runs\e8_data_seed42\protocol.json" goto verify
python e8_windows\download_polypgen.py
if errorlevel 1 exit /b 1
if exist "runs\e8_data_seed42\bundle.json" goto lock
python e8_windows\prepare_data.py
if errorlevel 1 exit /b 1
:lock
python e8_windows\run_e8.py lock-protocol
if errorlevel 1 exit /b 1
:verify
python e8_windows\run_e8.py preflight
if errorlevel 1 exit /b 1
endlocal
