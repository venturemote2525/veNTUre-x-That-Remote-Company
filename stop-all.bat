@echo off
REM Stop all Docker containers

echo Stopping Backend...
cd backend
docker compose down

echo Stopping Frontend...
cd ..\frontend
docker compose down

cd ..
echo All services stopped.
