FROM python:3.10-slim

WORKDIR /app

# Install system dependencies + python3-dev
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*
    
COPY requirements.txt .
COPY video_processing_temporal/requirements.txt ./video_processing_temporal/

# Base build tools and numpy (required for Cython extensions)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy==1.26.4 cython

# --- THE FIX: Separate standard PyPI packages from Git packages ---
# This pulls out all normal packages into 'pypi_reqs' and git links into 'git_reqs'
RUN sed '/git+/d' requirements.txt > pypi_reqs.txt && \
    sed '/git+/d' video_processing_temporal/requirements.txt > pypi_temporal_reqs.txt && \
    sed -n '/git+/p' requirements.txt > git_reqs.txt && \
    sed -n '/git+/p' video_processing_temporal/requirements.txt >> git_reqs.txt

# Phase 1: Install standard packages FIRST (scipy, torch, etc.)
RUN pip install --default-timeout=1000 --no-cache-dir -r pypi_reqs.txt && \
    pip install --default-timeout=1000 --no-cache-dir -r pypi_temporal_reqs.txt

# Phase 2: Install Git packages AFTER
# Now when deep-person-reid tries to import scipy or torch during setup, they actually exist!
RUN if [ -s git_reqs.txt ]; then pip install --default-timeout=1000 --no-cache-dir --no-build-isolation -r git_reqs.txt; fi

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "video_processing_temporal/worker.py"]