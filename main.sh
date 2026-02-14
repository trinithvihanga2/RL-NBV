#!/bin/bash

# Main setup and training script for RL-NBV
# This script sets up the environment and starts the training process

set -e  # Exit on error

echo "========================================="
echo "RL-NBV Setup and Training Script"
echo "========================================="

# Step 1: Install Python dependencies
echo ""
echo "[Step 1/6] Installing Python dependencies..."
pip install -r requirements.txt

# Step 2: Build CUDA distance modules (CRITICAL!)
echo ""
echo "[Step 2/6] Building CUDA distance modules..."
python setup.py build_ext --inplace

# Step 3: Verify build succeeded
echo ""
echo "[Step 3/6] Verifying CUDA modules build..."
if ls distance/*.so 1> /dev/null 2>&1; then
    echo "✓ CUDA modules successfully built:"
    ls -la distance/*.so
else
    echo "✗ ERROR: No .so files found in distance/ directory!"
    echo "  Build may have failed. Please check the error messages above."
    exit 1
fi

# Step 4: Data preparation check
echo ""
echo "[Step 4/6] Checking data directories..."
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

# Step 5: Optional replay buffer generation
echo ""
echo "[Step 5/6] Replay buffer generation (optional)..."
read -p "Do you want to generate replay buffer with oracle policy? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "./run_generate_replay_buffer.sh" ]; then
        echo "Running replay buffer generation script..."
        ./run_generate_replay_buffer.sh
    else
        echo "Running replay buffer generation directly..."
        python generate_replay_buffer.py \
            --data_path ./data/train \
            --buffer_size 1000000 \
            --view_num 33 \
            --observation_space_dim 1024 \
            --step_size 10 \
            --env_num 1 \
            --is_ratio_reward 0 \
            --save_path ideal_policy
    fi
else
    echo "Skipping replay buffer generation."
fi

# Step 6: Start training
echo ""
echo "[Step 6/6] Starting training..."
if [ -f "config.yaml" ]; then
    python train.py --config config.yaml
else
    echo "✗ ERROR: config.yaml not found!"
    echo "  Please ensure config.yaml exists in the current directory."
    exit 1
fi

echo ""
echo "========================================="
echo "Training completed successfully!"
echo "========================================="
