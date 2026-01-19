#!/bin/bash
# ============================================================================
# 3D Generation System - Linux/Mac Quick Start
# ============================================================================

echo ""
echo "==================== 3D Generation System ===================="
echo ""

# Check if running from correct directory
if [ ! -f "backend/requirements.txt" ]; then
    echo "ERROR: Run this script from the project root directory"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.9+"
    exit 1
fi

echo "[1/4] Installing/updating dependencies..."
cd backend
pip3 install -q -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi
cd ..
echo "✓ Dependencies installed"
echo ""

echo "[2/4] Checking GPU availability..."
python3 -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ""

echo "[3/4] Starting Backend Server..."
if command -v gnome-terminal &> /dev/null; then
    gnome-terminal -- bash -c "cd backend && python3 app.py; exec bash"
elif command -v xterm &> /dev/null; then
    xterm -e "cd backend && python3 app.py" &
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open -a Terminal --args "cd \"$(pwd)/backend\" && python3 app.py"
else
    # Fallback: start in background
    (cd backend && python3 app.py) &
fi
echo "✓ Backend started on http://localhost:5000"
echo ""

# Wait for backend to start
sleep 3

echo "[4/4] Starting Frontend..."
if command -v gnome-terminal &> /dev/null; then
    gnome-terminal -- bash -c "cd frontend && npm run dev; exec bash"
elif command -v xterm &> /dev/null; then
    xterm -e "cd frontend && npm run dev" &
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open -a Terminal --args "cd \"$(pwd)/frontend\" && npm run dev"
else
    # Fallback: start in background
    (cd frontend && npm run dev) &
fi
echo "✓ Frontend will start on http://localhost:3000"
echo ""

echo "==================== System Started ===================="
echo ""
echo "📍 Backend API:    http://localhost:5000"
echo "📍 Frontend UI:    http://localhost:3000"
echo "📍 3D Generation:  http://localhost:3000/generate-3d"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Keep script running
wait
