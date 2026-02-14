#!/bin/bash

# Main setup and training script for RL-NBV
# This script sets up the environment and starts the training process

set -e  # Exit on error

echo "========================================="
echo "RL-NBV Setup and Training Script"
echo "========================================="

# Step 1: Install Python dependencies
echo ""
echo "[Step 1/7] Installing Python dependencies..."
pip install -r requirements.txt

# Step 2: Build CUDA distance modules (CRITICAL!)
echo ""
echo "[Step 2/7] Building CUDA distance modules..."
python setup.py build_ext --inplace

# Step 3: Verify build succeeded
echo ""
echo "[Step 3/7] Verifying CUDA modules build..."
if ls distance/*.so 1> /dev/null 2>&1; then
    echo "✓ CUDA modules successfully built:"
    ls -la distance/*.so
else
    echo "✗ ERROR: No .so files found in distance/ directory!"
    echo "  Build may have failed. Please check the error messages above."
    exit 1
fi

# Step 4: Optional dataset splitting
echo ""
echo "[Step 4/7] Dataset splitting (optional)..."
read -p "Do you want to split the dataset? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running dataset splitting with config.yaml..."
    python split_dataset.py --config config.yaml
else
    echo "Skipping dataset splitting."
fi

# Step 5: Data preparation check
echo ""
echo "[Step 5/7] Checking data directories..."
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

# Step 6: Optional replay buffer generation
echo ""
echo "[Step 6/7] Replay buffer generation (optional)..."
read -p "Do you want to generate replay buffer with oracle policy? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Generating replay buffer using oracle policy with config.yaml..."
    python generate_replay_buffer.py --config config.yaml
else
    echo "Skipping replay buffer generation."
fi

# Step 7: Start training
echo ""
echo "[Step 7/7] Starting training..."
python train.py --config config.yaml

echo ""
echo "========================================="
echo "Training completed successfully!"
echo "========================================="
