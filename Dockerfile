# Food Nutrition Analysis System - Docker Container
# Multi-stage build for optimized production image

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

# Set build environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Production image
FROM python:3.11-slim as production

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_HOME=/app/.cache/torch
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV MODEL_CACHE_DIR=/app/.cache/models

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create necessary directories with proper permissions
RUN mkdir -p \
    /app/.cache/torch \
    /app/.cache/huggingface \
    /app/.cache/models \
    /app/outputs \
    /app/temp \
    /app/logs \
    /app/src/training/checkpoints \
    && chown -R appuser:appuser /app

# Copy project files (excluding what's in .dockerignore)
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Pre-create Python cache to speed up startup
RUN python -c "import sys; print('Python cache initialized')"

# Expose ports
EXPOSE 8501

# Add labels for better container management
LABEL maintainer="Food Analysis Team" \
      version="1.0.0" \
      description="Food Nutrition Analysis System with Segmentation and Depth Estimation" \
      architecture="multi-stage" \
      components="streamlit,pytorch,opencv,depth-anything,mask-rcnn"

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command - run nutrition analysis app
CMD ["streamlit", "run", "src/pipeline/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
