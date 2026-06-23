@echo off
REM ============================================================
REM STARTUP SCRIPT FOR FASTAPI SERVER
REM ============================================================

cd /d "%~dp0"

echo ============================================================
echo FastAPI UNet++ Boundary Prediction Server
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check if dependencies are installed
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo ✅ Starting server...
echo    Server URL: http://localhost:8000
echo    API Docs:   http://localhost:8000/docs
echo    Test Client: http://localhost:8000/redoc
echo.
echo Press CTRL+C to stop the server
echo.

python main.py

pause
