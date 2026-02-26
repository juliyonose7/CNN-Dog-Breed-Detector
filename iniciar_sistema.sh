#!/bin/bash

echo ""
echo "========================================"
echo "🚀 DOG BREED CLASSIFIER - STARTUP"
echo "========================================"
echo ""

cd "c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG"

echo "📍 Directorio actual: $(pwd)"
echo ""

echo "🔧 Verificando archivos necesarios..."

if [ ! -f "testing_api_119_classes.py" ]; then
    echo "❌ ERROR: testing_api_119_classes.py no encontrado"
    read -p "Presiona Enter para salir..."
    exit 1
fi

if [ ! -f "start_frontend.py" ]; then
    echo "❌ ERROR: start_frontend.py no encontrado"
    read -p "Presiona Enter para salir..."
    exit 1
fi

if [ ! -f "simple_frontend_119.html" ]; then
    echo "❌ ERROR: simple_frontend_119.html no encontrado"
    read -p "Presiona Enter para salir..."
    exit 1
fi

echo "✅ Todos los archivos encontrados"
echo ""

echo "🤖 Iniciando API del modelo ResNet50..."
echo "📊 Esto puede tomar unos segundos para cargar el modelo..."
echo ""

# Start API in background
python testing_api_119_classes.py &
API_PID=$!

echo "⏳ Esperando que la API se inicie..."
sleep 10

echo ""
echo "🌐 Iniciando servidor frontend..."
python start_frontend.py &
FRONTEND_PID=$!

echo "⏳ Esperando que el frontend se inicie..."
sleep 5

echo ""
echo "✅ Sistema iniciado exitosamente!"
echo ""
echo "📋 URLs importantes:"
echo "   🤖 API Backend: http://localhost:8000"
echo "   🌐 Frontend:    http://localhost:3000/simple_frontend_119.html"
echo ""
echo "💡 El navegador debería abrirse automáticamente"
echo "⚠️  Para detener el sistema: Ctrl+C"
echo ""

# Function to clean up processes on exit
cleanup() {
    echo ""
    echo "🛑 Deteniendo sistema..."
    kill $API_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Sistema detenido"
    exit 0
}

# Capture Ctrl+C
trap cleanup INT

# Keep script running
wait