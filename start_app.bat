@echo off
echo 🚀 Iniciando Dog Detector AI - Sistema Completo
echo.

echo 📡 Verificando servidor API...
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Servidor API no está ejecutándose. Iniciando...
    start "API Server" cmd /k "cd /d C:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG && C:/Python313/python.exe quick_api.py"
    timeout /t 3 >nul
) else (
    echo ✅ Servidor API ya está ejecutándose en puerto 8000
)

echo.
echo 🌐 Iniciando servidor frontend...
start "Frontend Server" cmd /k "cd /d C:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG && C:/Python313/python.exe frontend_server.py"

timeout /t 2 >nul

echo.
echo 🎯 Abriendo navegador...
start http://localhost:3000/standalone.html

echo.
echo ✅ Sistema iniciado!
echo 📱 Frontend: http://localhost:3000/standalone.html
echo 🔧 API: http://localhost:8000
echo 📚 Docs API: http://localhost:8000/docs
echo.
echo 💡 Para detener: Cierra las ventanas de consola que se abrieron
pause