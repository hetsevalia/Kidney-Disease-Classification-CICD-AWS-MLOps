# Use Python 3.11 slim base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for PyTorch, image processing, and utilities
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p \
    artifacts/data_ingestion \
    artifacts/prepare_base_model \
    artifacts/training \
    logs \
    templates \
    config

# Verify cnnClassifier package is installed
RUN python -c "from cnnClassifier import logger; print('Package installed successfully')" || \
    (echo "Installing cnnClassifier package..." && pip install -e .)

# Expose Flask port
EXPOSE 8080

# Set environment variables for Flask
ENV FLASK_APP=app.py \
    FLASK_ENV=production

# Health check to ensure container is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/model-status || exit 1

# Run the Flask application
CMD ["python", "app.py"]
