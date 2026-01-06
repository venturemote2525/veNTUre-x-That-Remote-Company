#!/bin/bash
# Build and run the backend Docker container

set -e

echo "Building and starting backend container..."
docker compose up --build
