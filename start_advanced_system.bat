<<<<<<< HEAD
@echo off
echo ========================================
echo    Giramille AI Advanced System
echo    Freepik-Level AI Image Generation
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 16+ and try again
    pause
    exit /b 1
)

echo Setting up advanced training environment...
python scripts/setup_advanced_training.py

echo.
echo Starting advanced frontend...
cd frontend
start "Giramille AI Frontend" cmd /k "npm run dev"

echo.
echo Starting advanced backend...
cd ..
python -m venv venv_advanced
call venv_advanced\Scripts\activate.bat
pip install -r requirements.txt
start "Giramille AI Backend" cmd /k "python backend/app.py"

echo.
echo ========================================
echo    Advanced System Started!
echo ========================================
echo.
echo Frontend: http://localhost:3000/advanced
echo Backend:  http://localhost:5000
echo.
echo Features Available:
echo - AI Image Generation (Gemini-style)
echo - Modular Scene Editing (Freepik-level)
echo - Multi-view Generation (Adobe Illustrator-style)
echo - Professional Vector Export
echo - Real-time Manipulation
echo - Custom Asset Integration
echo.
echo Press any key to exit...
pause >nul
=======
@echo off
echo ========================================
echo    Giramille AI Advanced System
echo    Freepik-Level AI Image Generation
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 16+ and try again
    pause
    exit /b 1
)

echo Setting up advanced training environment...
python scripts/setup_advanced_training.py

echo.
echo Starting advanced frontend...
cd frontend
start "Giramille AI Frontend" cmd /k "npm run dev"

echo.
echo Starting advanced backend...
cd ..
python -m venv venv_advanced
call venv_advanced\Scripts\activate.bat
pip install -r requirements.txt
start "Giramille AI Backend" cmd /k "python backend/app.py"

echo.
echo ========================================
echo    Advanced System Started!
echo ========================================
echo.
echo Frontend: http://localhost:3000/advanced
echo Backend:  http://localhost:5000
echo.
echo Features Available:
echo - AI Image Generation (Gemini-style)
echo - Modular Scene Editing (Freepik-level)
echo - Multi-view Generation (Adobe Illustrator-style)
echo - Professional Vector Export
echo - Real-time Manipulation
echo - Custom Asset Integration
echo.
echo Press any key to exit...
pause >nul
>>>>>>> 93065687c720c01a1e099ca0338e62bd0fa3ae90
