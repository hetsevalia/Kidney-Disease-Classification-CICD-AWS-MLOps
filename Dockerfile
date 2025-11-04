
FROM python:3.11-slim



# Install system dependencies including build tools
RUN apt update -y && apt install -y \
    awscli \
    gcc \
    g++ \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy model folder first (for better caching)
COPY model/ /app/model/

# Copy rest of the application
COPY . /app

RUN pip install --upgrade pip --default-timeout=100
RUN pip install --default-timeout=100 -r requirements.txt
RUN pip install -e .

CMD ["python3", "app.py"]
