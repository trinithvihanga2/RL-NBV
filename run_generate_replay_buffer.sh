#!/bin/bash
rm -rf env_*.log
CONFIG_PATH=${RL_NBV_CONFIG:-config.yaml}
python generate_replay_buffer.py --config "$CONFIG_PATH"
