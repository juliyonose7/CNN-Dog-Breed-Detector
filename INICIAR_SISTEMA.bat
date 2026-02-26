@echo off
echo.
echo ========================================
echo 🚀 DOG BREED CLASSIFIER - STARTUP
echo ========================================
echo.

cd /d "c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG"

echo 📍 Directorio actual: %cd%
echo.

echo 🔧 Verificando archivos necesarios...
if not exist "testing_api_119_classes.py" (
    echo ❌ ERROR: testing_api_119_classes.py no encontrado
    pause
    exit /b 1
)

if not exist "start_frontend.py" (
    echo ❌ ERROR: start_frontend.py no encontrado
    pause
    exit /b 1
)

if not exist "simple_frontend_119.html" (
    echo ❌ ERROR: simple_frontend_119.html no encontrado
    pause
    exit /b 1
)

echo ✅ Todos los archivos encontrados
echo.

echo 🤖 Iniciando API del modelo ResNet50...
echo 📊 Esto puede tomar unos segundos para cargar el modelo...
echo.

start "API Backend" /min cmd /k "python testing_api_119_classes.py"

echo ⏳ Esperando que la API se inicie...
timeout /t 10 /nobreak >nul

echo.
echo 🌐 Iniciando servidor frontend...
start "Frontend Server" /min cmd /k "python start_frontend.py"

echo ⏳ Esperando que el frontend se inicie...
timeout /t 5 /nobreak >nul

echo.
echo ✅ Sistema iniciado exitosamente!
echo.
echo 📋 URLs importantes:
echo    🤖 API Backend: http://localhost:8000
echo    🌐 Frontend:    http://localhost:3000/simple_frontend_119.html
echo.
echo 💡 El navegador debería abrirse automáticamente
echo ⚠️  NO CIERRES esta ventana para mantener el sistema funcionando
echo.
echo 🛑 Para detener el sistema: Ctrl+C en las ventanas de API y Frontend
echo.

pause