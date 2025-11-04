# Docker Guide - Kidney Disease Classification

## 🐳 Docker Setup and Usage

This guide explains how to build and run the Kidney Disease Classification application using Docker.

---

## 📋 Prerequisites

- Docker installed on your system ([Install Docker](https://docs.docker.com/get-docker/))
- At least 4GB of free disk space
- Docker daemon running

---

## 🚀 Quick Start

### Build Docker Image

```bash
# Navigate to project directory
cd /path/to/Kidney-Disease-Classification-Deep-Learning-Project

# Build the Docker image
docker build -t kidney-classifier:latest .
```

### Run Docker Container

```bash
# Run the container
docker run -d \
  --name kidney-classifier \
  -p 8080:8080 \
  kidney-classifier:latest

# View logs
docker logs -f kidney-classifier

# Stop container
docker stop kidney-classifier

# Remove container
docker rm kidney-classifier
```

### Access Application

Once running, access the application at:
- **Web UI**: http://localhost:8080
- **Model Status API**: http://localhost:8080/model-status

---

## 📦 Docker Image Details

### Base Image
- **Python 3.10 slim** - Lightweight Python base image

### Installed Packages
- PyTorch 2.0+ with CPU support
- Flask and Flask-CORS
- All dependencies from `requirements.txt`
- System libraries for image processing

### Image Structure
```
/app
├── src/              # Source code
├── config/           # Configuration files
├── templates/        # HTML templates
├── artifacts/        # Model artifacts (created at runtime)
└── logs/            # Application logs
```

---

## 🔧 Advanced Usage

### Run with Volume Mounts

Mount local directories for persistent storage:

```bash
docker run -d \
  --name kidney-classifier \
  -p 8080:8080 \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/logs:/app/logs \
  kidney-classifier:latest
```

This allows:
- Models to persist between container restarts
- Logs to be accessible from host
- Data to persist after container removal

### Run with Environment Variables

```bash
docker run -d \
  --name kidney-classifier \
  -p 8080:8080 \
  -e FLASK_ENV=development \
  -e PYTHONUNBUFFERED=1 \
  kidney-classifier:latest
```

### Run in Interactive Mode

```bash
# Run with interactive shell
docker run -it --rm \
  --name kidney-classifier \
  -p 8080:8080 \
  kidney-classifier:latest \
  /bin/bash

# Inside container, you can:
# - Run training: python main.py
# - Run app: python app.py
# - Inspect files: ls -la
```

### View Container Logs

```bash
# Follow logs in real-time
docker logs -f kidney-classifier

# View last 100 lines
docker logs --tail 100 kidney-classifier

# View logs with timestamps
docker logs -t kidney-classifier
```

### Execute Commands in Running Container

```bash
# Check model status
docker exec kidney-classifier python -c "import os; print('Model exists:', os.path.exists('artifacts/training/model.pth'))"

# Train model inside container
docker exec -it kidney-classifier python main.py

# Access shell
docker exec -it kidney-classifier /bin/bash
```

---

## 🏗️ Building Options

### Build with Custom Tag

```bash
docker build -t kidney-classifier:v1.0 .
docker build -t my-registry/kidney-classifier:latest .
```

### Build with Build Arguments

You can modify the Dockerfile to accept build arguments for customization:

```dockerfile
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim
```

Build with:
```bash
docker build --build-arg PYTHON_VERSION=3.9 -t kidney-classifier .
```

### Build for Different Platforms

```bash
# Build for ARM64 (Apple Silicon)
docker build --platform linux/arm64 -t kidney-classifier:arm64 .

# Build for AMD64 (Intel/AMD)
docker build --platform linux/amd64 -t kidney-classifier:amd64 .
```

---

## 📊 Health Check

The Dockerfile includes a health check that monitors the application:

```bash
# Check container health
docker ps

# Inspect health check status
docker inspect --format='{{.State.Health.Status}}' kidney-classifier
```

Health check endpoints:
- Checks `/model-status` endpoint every 30 seconds
- Container marked unhealthy after 3 consecutive failures

---

## 🚨 Troubleshooting

### Container Won't Start

1. **Check logs:**
   ```bash
   docker logs kidney-classifier
   ```

2. **Check if port is already in use:**
   ```bash
   # On Linux/Mac
   lsof -i :8080
   
   # Or use different port
   docker run -p 8081:8080 kidney-classifier:latest
   ```

3. **Check disk space:**
   ```bash
   docker system df
   ```

### Model Not Found Error

If you see "Model not found" error:
1. Train the model inside the container:
   ```bash
   docker exec -it kidney-classifier python main.py
   ```

2. Or mount a directory with pre-trained model:
   ```bash
   docker run -v /path/to/model:/app/artifacts/training/model.pth ...
   ```

### Out of Memory

If container runs out of memory:
```bash
# Increase memory limit
docker run --memory="2g" kidney-classifier:latest
```

### Permission Issues

If you encounter permission issues with volumes:
```bash
# Fix permissions
docker run --user $(id -u):$(id -g) kidney-classifier:latest
```

---

## 🐛 Debugging

### Inspect Container

```bash
# Check container configuration
docker inspect kidney-classifier

# Check running processes
docker top kidney-classifier

# Check resource usage
docker stats kidney-classifier
```

### Access Container Shell

```bash
# Get interactive shell
docker exec -it kidney-classifier /bin/bash

# Inside container:
# - Check Python version: python --version
# - Check packages: pip list
# - Check files: ls -la
# - Run Python: python -c "from cnnClassifier import logger"
```

---

## 🚢 Deployment

### Push to Docker Registry

```bash
# Login to registry
docker login

# Tag image
docker tag kidney-classifier:latest your-registry/kidney-classifier:latest

# Push image
docker push your-registry/kidney-classifier:latest
```

### Deploy to AWS ECR

```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag
docker tag kidney-classifier:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/kidney-classifier:latest

# Push
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/kidney-classifier:latest
```

---

## 📝 Docker Compose (Optional)

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  kidney-classifier:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./artifacts:/app/artifacts
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8080/model-status')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

Run with:
```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

---

## 🔐 Security Best Practices

1. **Don't run as root** (if possible, though slim images usually use non-root)
2. **Use specific image tags** instead of `latest`
3. **Scan images for vulnerabilities:**
   ```bash
   docker scan kidney-classifier:latest
   ```
4. **Keep base images updated**
5. **Use secrets management** for sensitive data (don't hardcode)

---

## 📈 Performance Tips

1. **Use multi-stage builds** for smaller images (if needed)
2. **Layer caching** - requirements.txt is copied first for better caching
3. **Use .dockerignore** to exclude unnecessary files
4. **Limit resource usage:**
   ```bash
   docker run --cpus="2" --memory="2g" kidney-classifier:latest
   ```

---

## ✅ Verification

After building and running, verify:

1. **Container is running:**
   ```bash
   docker ps | grep kidney-classifier
   ```

2. **Application is accessible:**
   ```bash
   curl http://localhost:8080/model-status
   ```

3. **Health check is passing:**
   ```bash
   docker inspect --format='{{.State.Health.Status}}' kidney-classifier
   ```

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)

---

## 🆘 Need Help?

If you encounter issues:
1. Check container logs: `docker logs kidney-classifier`
2. Verify Docker is running: `docker ps`
3. Check disk space: `docker system df`
4. Review Dockerfile for any missing dependencies

