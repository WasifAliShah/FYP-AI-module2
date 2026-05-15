FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for OpenCV, PyTorch, and git dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
    
# Copy requirements files first to leverage Docker cache
COPY requirements.txt .
COPY video_processing_temporal/requirements.txt ./video_processing_temporal/

# Install python dependencies
# FIX: Pre-install numpy, cython, and build tools so git-based packages can compile their wheels
RUN pip install --default-timeout=1000 --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --default-timeout=1000 --no-cache-dir numpy cython && \
    pip install --default-timeout=1000 --no-cache-dir -r requirements.txt && \
    pip install --default-timeout=1000 --no-cache-dir -r video_processing_temporal/requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure Python can find local modules
ENV PYTHONPATH=/app

# By default, start the Temporal Worker
CMD ["python", "video_processing_temporal/worker.py"]