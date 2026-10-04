@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv not found. Run start.bat first.
    pause
    exit /b 1
)

echo Download missing Qwen-Image 2.1 model files into the model folder.
"%~dp0venv\Scripts\python.exe" backend\download_models.py %*
pause
