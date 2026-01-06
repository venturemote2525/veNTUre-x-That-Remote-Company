@echo off
REM Build and run both frontend and backend Docker containers

echo =====================================
echo Building and starting Backend...
echo =====================================
cd backend
docker compose up --build -d
if %ERRORLEVEL% neq 0 (
    echo Backend build failed!
    exit /b %ERRORLEVEL%
)

echo.
echo =====================================
echo Building and starting Frontend...
echo =====================================
cd ..\frontend
docker compose up --build -d
if %ERRORLEVEL% neq 0 (
    echo Frontend build failed!
    exit /b %ERRORLEVEL%
)

cd ..

echo.
echo =====================================
echo All services started!
echo =====================================
echo Backend:  http://localhost:8080
echo Frontend: Scan QR code from 'docker compose logs -f' in frontend folder
echo.
echo To view logs:
echo   Backend:  docker compose -f backend/docker-compose.yml logs -f
echo   Frontend: docker compose -f frontend/docker-compose.yml logs -f
echo.
echo To stop all:
echo   stop-all.bat
