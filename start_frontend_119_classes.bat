@echo off
echo Starting React frontend for 119-class model testing...
echo =============================================================

cd /d "C:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\dog-detector-frontend"
echo Working directory: %CD%

echo Checking dependencies...
if not exist node_modules (
    echo Installing dependencies...
    npm install
) else (
    echo Dependencies already installed
)

echo Starting React development server...
echo Frontend will be available at: http://localhost:3000
echo API Backend available at: http://localhost:8000
echo.
echo To test the model:
echo   1. Open http://localhost:3000 in your browser
echo   2. Upload a dog image
echo   3. The model will classify across 119 different breeds
echo.
echo Make sure the API backend is running on port 8000
echo.

npm start

pause