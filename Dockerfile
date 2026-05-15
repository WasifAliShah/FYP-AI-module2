FROM python:3.10-slim

WORKDIR /app

# 1. Install system dependencies + python3-dev (the "Visual Studio" headers for Linux)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*
    
COPY requirements.txt .
COPY video_processing_temporal/requirements.txt ./video_processing_temporal/

# 2. Install build tools and pre-requisites
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy==1.26.4 cython

# 3. Install requirements using --no-build-isolation
# This tells pip: "Use the numpy I just installed, don't try to find it in a sandbox."
RUN pip install --default-timeout=1000 --no-cache-dir --no-build-isolation -r requirements.txt && \
    pip install --default-timeout=1000 --no-cache-dir -r video_processing_temporal/requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "video_processing_temporal/worker.py"]