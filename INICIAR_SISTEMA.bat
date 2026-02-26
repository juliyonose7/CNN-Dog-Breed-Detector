@echo off
echo.
echo ========================================
echo 🚀 DOG BREED CLASSIFIER - STARTUP
echo ========================================
echo.

cd /d "c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG"

echo Working directory: %cd%
echo.

echo Checking required files...
if not exist "testing_api_119_classes.py" (
    echo ERROR: testing_api_119_classes.py not found
    pause
    exit /b 1
)

if not exist "start_frontend.py" (
    echo ERROR: start_frontend.py not found
    pause
    exit /b 1
)

if not exist "simple_frontend_119.html" (
    echo ERROR: simple_frontend_119.html not found
    pause
    exit /b 1
)

echo All required files found
echo.

echo Starting ResNet50 model API...
echo Model loading may take a few seconds...
echo.

start "API Backend" /min cmd /k "python testing_api_119_classes.py"

echo Waiting for API to initialize...
timeout /t 10 /nobreak >nul

echo.
echo Starting frontend server...
start "Frontend Server" /min cmd /k "python start_frontend.py"

echo Waiting for frontend to initialize...
timeout /t 5 /nobreak >nul

echo.
echo System started successfully!
echo.
echo Service URLs:
echo    API Backend: http://localhost:8000
echo    Frontend:    http://localhost:3000/simple_frontend_119.html
echo.
echo The browser should open automatically
echo DO NOT close this window - it keeps the system running
echo.
echo To stop the system: Ctrl+C in the API and Frontend windows
echo.

pause