@echo off
REM ============================================================================
REM 3D Generation System - Windows Quick Start
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ==================== 3D Generation System ====================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.9+
    exit /b 1
)

REM Check if running from correct directory
if not exist "backend\requirements.txt" (
    echo ERROR: Run this script from the project root directory
    exit /b 1
)

echo [1/4] Installing/updating dependencies...
cd backend
pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)
cd ..
echo ✓ Dependencies installed

echo.
echo [2/4] Checking GPU availability...
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo.

echo [3/4] Starting Backend Server...
start "Giramille Backend" cmd /k "cd backend && python app.py"
echo ✓ Backend started on http://localhost:5000
echo.

REM Wait for backend to start
timeout /t 3 /nobreak

echo [4/4] Starting Frontend...
start "Giramille Frontend" cmd /k "cd frontend && npm run dev"
echo ✓ Frontend will start on http://localhost:3000
echo.

echo ==================== System Started ====================
echo.
echo 📍 Backend API:    http://localhost:5000
echo 📍 Frontend UI:    http://localhost:3000
echo 📍 3D Generation:  http://localhost:3000/generate-3d
echo.
echo Press Ctrl+C in either window to stop
echo.
pause
