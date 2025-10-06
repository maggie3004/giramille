<<<<<<< HEAD
@echo off
echo Starting AI Image Generation Advanced Editor...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo Node.js not found. Please install Node.js 16+ and try again.
    pause
    exit /b 1
)

echo Installing backend dependencies...
cd backend
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install backend dependencies.
    pause
    exit /b 1
)

echo.
echo Installing frontend dependencies...
cd ..\frontend
npm install
if errorlevel 1 (
    echo Failed to install frontend dependencies.
    pause
    exit /b 1
)

echo.
echo Starting backend server...
start "Backend Server" cmd /k "cd ..\backend && python run.py"

echo Waiting for backend to start...
timeout /t 3 /nobreak >nul

echo.
echo Starting frontend development server...
start "Frontend Server" cmd /k "cd frontend && npm run dev"

echo.
echo Both servers are starting...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo Advanced Editor: http://localhost:3000/editor
echo.
echo Press any key to stop both servers...
pause >nul

echo Stopping servers...
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im node.exe >nul 2>&1
echo Servers stopped.
=======
@echo off
echo Starting AI Image Generation Advanced Editor...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo Node.js not found. Please install Node.js 16+ and try again.
    pause
    exit /b 1
)

echo Installing backend dependencies...
cd backend
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install backend dependencies.
    pause
    exit /b 1
)

echo.
echo Installing frontend dependencies...
cd ..\frontend
npm install
if errorlevel 1 (
    echo Failed to install frontend dependencies.
    pause
    exit /b 1
)

echo.
echo Starting backend server...
start "Backend Server" cmd /k "cd ..\backend && python run.py"

echo Waiting for backend to start...
timeout /t 3 /nobreak >nul

echo.
echo Starting frontend development server...
start "Frontend Server" cmd /k "cd frontend && npm run dev"

echo.
echo Both servers are starting...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo Advanced Editor: http://localhost:3000/editor
echo.
echo Press any key to stop both servers...
pause >nul

echo Stopping servers...
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im node.exe >nul 2>&1
echo Servers stopped.
>>>>>>> 93065687c720c01a1e099ca0338e62bd0fa3ae90
