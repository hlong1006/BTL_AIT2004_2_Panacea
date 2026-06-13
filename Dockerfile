# Multi-stage build for efficiency
FROM python:3.10-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-docker.txt .

# Install Python dependencies (CPU-only PyTorch, no CUDA)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefer-binary --retries 5 --default-timeout=1000 -r requirements-docker.txt

# Copy application code
COPY . .

# Create volumes for data and outputs
RUN mkdir -p /app/data /app/outputs /app/models

# Expose port for API (future use)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command (will be overridden by docker-compose)
CMD ["python", "web_app.py"]
