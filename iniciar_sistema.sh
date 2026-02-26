#!/bin/bash

echo ""
echo "========================================"
echo "🚀 DOG BREED CLASSIFIER - STARTUP"
echo "========================================"
echo ""

cd "c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG"

echo "📍 Working directory: $(pwd)"
echo ""

echo "🔧 Checking required files..."

if [ ! -f "testing_api_119_classes.py" ]; then
    echo "❌ ERROR: testing_api_119_classes.py not found"
    read -p "Press Enter to exit..."
    exit 1
fi

if [ ! -f "start_frontend.py" ]; then
    echo "❌ ERROR: start_frontend.py not found"
    read -p "Press Enter to exit..."
    exit 1
fi

if [ ! -f "simple_frontend_119.html" ]; then
    echo "❌ ERROR: simple_frontend_119.html not found"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "✅ All required files found"
echo ""

echo "🤖 Starting ResNet50 model API..."
echo "📊 Model loading may take a few seconds..."
echo ""

# Start API server in background
python testing_api_119_classes.py &
API_PID=$!

echo "⏳ Waiting for API to initialize..."
sleep 10

echo ""
echo "🌐 Starting frontend server..."
python start_frontend.py &
FRONTEND_PID=$!

echo "⏳ Waiting for frontend to initialize..."
sleep 5

echo ""
echo "✅ System started successfully!"
echo ""
echo "📋 Service URLs:"
echo "   🤖 API Backend: http://localhost:8000"
echo "   🌐 Frontend:    http://localhost:3000/simple_frontend_119.html"
echo ""
echo "💡 The browser should open automatically"
echo "⚠️  To stop the system: Ctrl+C"
echo ""

# Cleanup handler: terminate background processes on exit
cleanup() {
    echo ""
    echo "🛑 Stopping system..."
    kill $API_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ System stopped"
    exit 0
}

# Trap SIGINT (Ctrl+C) to trigger cleanup
trap cleanup INT

# Keep script alive while background processes run
wait