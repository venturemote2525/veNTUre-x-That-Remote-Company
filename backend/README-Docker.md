# Backend Docker Setup

## Prerequisites

- Docker installed
- Docker Compose (or `docker compose` plugin)
- Backend `.env` file present in this folder (used by docker-compose)

## Build and Run with Docker Compose (recommended)

From the `backend` folder:

```bash
docker compose up --build
```

This will:
- Build the `venture-backend` image from the Dockerfile
- Start the backend on port `8080`

The API will be available at:

```
http://localhost:8080
```

To stop the containers:

```bash
docker compose down
```

## Build and Run with Docker Only (optional)

Build the image:

```bash
docker build -t venture-backend .
```

Run the container:

```bash
docker run --name venture-backend \
	--env-file .env \
	-p 8080:8080 \
	venture-backend
```

Stop and remove the container:

```bash
docker stop venture-backend && docker rm venture-backend
```

