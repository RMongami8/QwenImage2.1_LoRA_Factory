@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"

:: Change this number if the port is already in use.
set APP_PORT=8002

echo [Qwen-Image 2.1 LoRA Factory] Initializing...

:: Create the virtual environment (Python 3.12) if it does not exist.
if not exist "venv\Scripts\python.exe" (
    echo [INFO] Creating virtual environment venv ...
    where uv >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        uv venv venv --python 3.12
    ) else (
        py -3.12 -m venv venv
    )
    if not exist "venv\Scripts\python.exe" (
        echo.
        echo [ERROR] Could not create the virtual environment.
        echo         Install Python 3.12 ^(https://www.python.org/^) or uv ^(https://docs.astral.sh/uv/^).
        pause
        exit /b 1
    )
)

set VENV_PYTHON=%~dp0venv\Scripts\python.exe

"%VENV_PYTHON%" backend\setup_check.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Setup check failed. Please read the messages above.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [Qwen-Image 2.1 LoRA Factory] Starting server...
echo The browser opens automatically. If port %APP_PORT% is busy, the next free port is used.
echo.

cd backend
"%VENV_PYTHON%" main.py

pause
