#!/bin/bash
# Build and run both frontend and backend Docker containers

set -e

echo "====================================="
echo "Building and starting Backend..."
echo "====================================="
cd backend
docker compose up --build -d

echo ""
echo "====================================="
echo "Building and starting Frontend..."
echo "====================================="
cd ../frontend
docker compose up --build -d

echo ""
echo "====================================="
echo "All services started!"
echo "====================================="
echo "Backend:  http://localhost:8080"
echo "Frontend: Scan QR code from 'docker compose logs -f frontend'"
echo ""
echo "To view logs:"
echo "  Backend:  docker compose -f backend/docker-compose.yml logs -f"
echo "  Frontend: docker compose -f frontend/docker-compose.yml logs -f"
echo ""
echo "To stop all:"
echo "  ./stop-all.sh"
