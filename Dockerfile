# ============================================================
# RL-NBV Trainer Dockerfile
# Base: CUDA 12.8 on Ubuntu 20.04
# Runs: train.py with config.yaml
# ============================================================

FROM nvidia/cuda:12.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies + Python 3.11 via deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        git \
        build-essential \
        ninja-build \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set working directory
WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv sync --no-dev --python 3.11

# Copy only the files needed for training
COPY config.yaml train.py custom_callback.py ./
COPY envs/ envs/
COPY models/__init__.py models/pointnet2_cls_ssg.py models/pointnet2_utils.py models/
COPY models/pretrained/ models/pretrained/
COPY optim/ optim/
COPY distance/chamfer_distance.py distance/chamfer_distance.cpp distance/chamfer_distance.cu distance/

# Default: run training with config.yaml
CMD ["uv", "run", "--no-dev", "python", "train.py", "--config", "config.yaml"]
