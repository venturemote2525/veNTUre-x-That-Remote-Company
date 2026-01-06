#!/bin/bash
# Stop all Docker containers

echo "Stopping Backend..."
cd backend
docker compose down

echo "Stopping Frontend..."
cd ../frontend
docker compose down

echo "All services stopped."
