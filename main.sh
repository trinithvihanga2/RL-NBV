#!/bin/bash

# Main setup and training script for RL-NBV
# This script sets up the environment and starts the training process

set -e  # Exit on error

echo "========================================="
echo "RL-NBV Setup and Training Script"
echo "========================================="

echo ""
echo "[Step 1] Creating and activating virtual environment..."
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo "Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"

echo ""
echo "[Step 2] Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo ""
echo "[Step 3] Building CUDA distance modules..."
python setup.py build_ext --inplace

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
    python split_dataset.py --config config.yaml
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
    python generate_replay_buffer.py --config config.yaml
else
    echo "Skipping replay buffer generation."
fi

echo ""
echo "[Step 8] Starting training..."
python train.py --config config.yaml

echo ""
echo "========================================="
echo "Training completed successfully!"
echo "========================================="
