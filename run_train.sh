#!/bin/bash
rm -rf env_*log
rm -rf model_detail.log
CONFIG_PATH=${RL_NBV_CONFIG:-config.yaml}
python train.py --config "$CONFIG_PATH"
