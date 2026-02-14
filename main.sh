#!/bin/bash

# Main setup and training script for RL-NBV
# This script sets up the environment and starts the training process

set -e  # Exit on error

echo "========================================="
echo "RL-NBV Setup and Training Script"
echo "========================================="

echo ""
echo "[Step 1] Installing/checking uv package manager..."
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    echo "✓ uv installed successfully"
else
    echo "✓ uv is already installed"
fi

echo ""
echo "[Step 2] Setting up Python 3.8 environment with uv..."
echo "Syncing dependencies with Python 3.8..."
uv sync --python 3.8

echo ""
echo "[Step 3] Building CUDA distance modules..."
uv run python setup.py build_ext --inplace

echo ""
echo "[Step 4] Verifying CUDA modules build..."
if ls distance/*.so 1> /dev/null 2>&1; then
    echo "✓ CUDA modules successfully built:"
    ls -la distance/*.so
else
    echo "✗ ERROR: No .so files found in distance/ directory!"
    echo "  Build may have failed. Please check the error messages above."
    exit 1
fi

echo ""
echo "[Step 5] Dataset splitting (optional)..."
read -p "Do you want to split the dataset? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running dataset splitting with config.yaml..."
    uv run python split_dataset.py --config config.yaml
else
    echo "Skipping dataset splitting."
fi

echo ""
echo "[Step 6] Checking data directories..."
if [ ! -d "./data/train" ]; then
    echo "⚠ Warning: ./data/train/ directory not found"
    echo "  Please ensure your training data is in ./data/train/"
fi
if [ ! -d "./data/verify" ]; then
    echo "⚠ Warning: ./data/verify/ directory not found"
    echo "  Please ensure your verification data is in ./data/verify/"
fi
if [ ! -d "./data/test" ]; then
    echo "⚠ Warning: ./data/test/ directory not found"
    echo "  Please ensure your test data is in ./data/test/"
fi

echo ""
echo "[Step 7] Replay buffer generation (optional)..."
read -p "Do you want to generate replay buffer with oracle policy? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Generating replay buffer using oracle policy with config.yaml..."
    uv run python generate_replay_buffer.py --config config.yaml
else
    echo "Skipping replay buffer generation."
fi

echo ""
echo "[Step 8] Starting training..."
uv run python train.py --config config.yaml

echo ""
echo "========================================="
echo "Training completed successfully!"
echo "========================================="
