import envs.rl_nbv_env
import argparse
import logging
import yaml
import os
import sys
from copy import deepcopy
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.save_util import save_to_pkl, load_from_pkl
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import numpy as np
import random


# ============================================================================
# CONFIG
# ============================================================================
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_to_args(config):
    env = config.get("environment", {})
    rbg = config.get("replay_buffer_generation", {})

    return {
        # Environment
        "data_path": rbg.get("data_path", env.get("data_path")),
        "view_num": env.get("view_num", 33),
        "observation_space_dim": env.get("observation_space_dim", 1024),
        "step_size": env.get("step_size", 10),
        "env_num": env.get("env_num", 1),
        # Replay Buffer Generation
        "buffer_size": rbg.get("buffer_size", 1000000),
        "save_path": rbg.get("save_path", "ideal_policy"),
        "log_path": rbg.get("log_path", "replay_buffer.log"),
        "is_load_buffer": rbg.get("is_load_buffer", 0),
        "load_path": rbg.get("load_path", None),
        "is_add_negative_exp": rbg.get("is_add_negative_exp", 0),
        "negative_exp_factor": rbg.get("negative_exp_factor", 0.03),
        "is_ratio_reward": rbg.get("is_ratio_reward", 1),
        "is_reward_with_cur_coverage": rbg.get("is_reward_with_cur_coverage", 1),
        "cur_coverage_ratio": rbg.get("cur_coverage_ratio", 1.0),
    }


# ============================================================================
# LOGGER SETUP
# ============================================================================
def setup_logger(log_path="replay_buffer.log"):
    logger = logging.getLogger("generate_replay_buffer")
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handle = logging.FileHandler(log_path)
    file_handle.setFormatter(log_format)
    file_handle.setLevel(logging.DEBUG)
    logger.addHandler(file_handle)

    console_handle = logging.StreamHandler()
    console_handle.setFormatter(log_format)
    console_handle.setLevel(logging.INFO)
    logger.addHandler(console_handle)

    return logger


# ============================================================================
# ENVIRONMENT FACTORY
# ============================================================================
def make_env(data_path, env_id, args, logger):
    def _f():
        if args.env_num == 1:
            env = envs.rl_nbv_env.PointCloudNextBestViewEnv(
                data_path=data_path,
                view_num=args.view_num,
                observation_space_dim=args.observation_space_dim,
                log_level=logging.INFO,
                is_ratio_reward=(args.is_ratio_reward == 1),
                is_reward_with_cur_coverage=(args.is_reward_with_cur_coverage == 1),
                cur_coverage_ratio=args.cur_coverage_ratio,
            )
            return env
        if args.is_ratio_reward == 1:
            env = envs.rl_nbv_env.PointCloudNextBestViewEnv(
                data_path=data_path,
                view_num=args.view_num,
                observation_space_dim=args.observation_space_dim,
                env_id=env_id,
                log_level=logging.INFO,
                is_ratio_reward=True,
                is_reward_with_cur_coverage=(args.is_reward_with_cur_coverage == 1),
                cur_coverage_ratio=args.cur_coverage_ratio,
            )
            logger.info("is_ratio_reward is True")
        else:
            env = envs.rl_nbv_env.PointCloudNextBestViewEnv(
                data_path=data_path,
                view_num=args.view_num,
                observation_space_dim=args.observation_space_dim,
                env_id=env_id,
                log_level=logging.INFO,
                is_ratio_reward=False,
                is_reward_with_cur_coverage=(args.is_reward_with_cur_coverage == 1),
                cur_coverage_ratio=args.cur_coverage_ratio,
            )
            logger.info("is_ratio_reward is False")
        return env

    return _f


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and print resolved values without generating",
    )
    cli = parser.parse_args()

    config_path = cli.config
    if not os.path.exists(config_path):
        raise FileNotFoundError("Config not found: {}".format(config_path))

    config = load_config(config_path)
    args = argparse.Namespace(**config_to_args(config))

    if cli.dry_run:
        print("Config: {}".format(config_path))
        for arg, value in sorted(vars(args).items()):
            print("  {:30s}: {}".format(arg, value))
        sys.exit(0)

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = setup_logger(args.log_path)
    logger.info("=" * 60)
    logger.info("REPLAY BUFFER GENERATION START")
    logger.info("=" * 60)
    for arg, value in sorted(vars(args).items()):
        logger.info("  {:30s}: {}".format(arg, value))
    logger.info("=" * 60)

    # ── Environments ──────────────────────────────────────────────────────────
    logger.info("Building environments...")
    env_list = []
    for i in range(args.env_num):
        env_list.append(make_env(args.data_path, i, args, logger))
    env_vec = DummyVecEnv(env_list)
    logger.info("Environments ready ✅")

    # ── Replay Buffer ─────────────────────────────────────────────────────────
    replay_buffer = None
    if args.is_load_buffer == 1:
        if args.load_path is None:
            logger.error("is_load_buffer=1 but load_path is None")
            sys.exit(1)
        replay_buffer = load_from_pkl(args.load_path, verbose=1)
        logger.info("Replay buffer loaded ✅")
    else:
        replay_buffer = DictReplayBuffer(
            buffer_size=args.buffer_size,
            observation_space=env_vec.observation_space,
            action_space=env_vec.action_space,
            device="cuda:1",
            n_envs=args.env_num,
        )
        logger.info("New replay buffer created (size={})".format(args.buffer_size))

    # ── Core Loop ─────────────────────────────────────────────────────────────
    experience_size = 0
    negative_experience_size = 0
    model_size = env_vec.envs[0].shapenet_reader.model_num
    logger.info("begin execution, model size: {}".format(model_size * args.env_num))

    for model_id in range(model_size):
        logger.info("handle {} model".format(model_id * args.env_num))
        last_obs = env_vec.reset()
        actions = np.zeros((args.env_num,), dtype=np.int32)
        for step in range(args.step_size - 1):
            experience_size += args.env_num
            for env_id in range(args.env_num):
                view_state = last_obs["view_state"][env_id]
                action = 0
                cover_add_max = 0
                if args.is_add_negative_exp == 1:
                    factor = random.random()
                    if factor <= args.negative_exp_factor:
                        negative_experience_size += 1
                        action = random.randint(0, args.view_num - 1)
                        while view_state[action] != 1:
                            action = random.randint(0, args.view_num - 1)
                        actions[env_id] = action
                        continue
                for i in range(args.view_num):
                    if view_state[i] == 1:
                        continue
                    cover_add_cur = env_vec.envs[env_id].try_step(i)
                    if cover_add_cur >= cover_add_max:
                        cover_add_max = cover_add_cur
                        action = i
                    actions[env_id] = action
            obs, reward, done, info = env_vec.step(actions)

            # Avoid modification by reference
            obs_ = deepcopy(obs)
            info_ = deepcopy(info)
            last_obs_ = deepcopy(last_obs)
            actions_ = deepcopy(actions)

            replay_buffer.add(last_obs_, obs_, actions_, reward, done, info_)
            if np.any(np.isnan(last_obs["current_point_cloud"])):
                logger.error("current_point_cloud has nan")
                for env_id in range(args.env_num):
                    logger.error(
                        "model name: {}".format(env_vec.envs[env_id].model_name)
                    )
            if np.any(np.isnan(last_obs["view_state"])):
                logger.error("view_state has nan")
                for env_id in range(args.env_num):
                    logger.error(
                        "model name: {}".format(env_vec.envs[env_id].model_name)
                    )
            last_obs = obs

    # ── Save ──────────────────────────────────────────────────────────────────
    logger.info("save as pkl file")
    save_to_pkl(args.save_path, replay_buffer, verbose=1)
    logger.info(
        "model size: {}, experience size: {}, negative experience size: {}".format(
            model_size, experience_size, negative_experience_size
        )
    )

    logger.info("=" * 60)
    logger.info("REPLAY BUFFER GENERATION DONE ✅")
    logger.info("=" * 60)
