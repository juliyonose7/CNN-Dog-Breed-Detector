@echo off
echo 🚀 Iniciando frontend React para testing del modelo de 119 clases...
echo =============================================================

cd /d "C:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\dog-detector-frontend"
echo 📁 Directorio actual: %CD%

echo 📦 Verificando dependencias...
if not exist node_modules (
    echo ⬇️ Instalando dependencias...
    npm install
) else (
    echo ✅ Dependencias ya instaladas
)

echo 🔄 Iniciando servidor de desarrollo React...
echo 🌐 El frontend estará disponible en: http://localhost:3000
echo 🔗 API Backend disponible en: http://localhost:8000
echo.
echo 📝 Para probar el modelo:
echo   1. Abre http://localhost:3000 en tu navegador
echo   2. Sube una imagen de un perro
echo   3. El modelo clasificará entre 119 razas diferentes
echo.
echo ⚠️ Asegúrate de que el API backend esté ejecutándose en puerto 8000
echo.

npm start

pause