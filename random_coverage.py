import argparse
import logging
import os
import random
import sys

import numpy as np
import yaml

import envs.rl_nbv_env


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logger(log_file="random_coverage.log"):
    logger = logging.getLogger("random_coverage")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    log_format = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    file_handle = logging.FileHandler(log_file)
    file_handle.setFormatter(log_format)
    file_handle.setLevel(logging.DEBUG)

    console_handle = logging.StreamHandler()
    console_handle.setFormatter(log_format)
    console_handle.setLevel(logging.INFO)

    logger.addHandler(file_handle)
    logger.addHandler(console_handle)
    return logger


# ============================================================================
# CONFIG
# ============================================================================
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_to_args(config):
    ds = config.get("dataset", {})
    env = config.get("environment", {})
    train = config.get("training", {})
    out = train.get("output", {})
    log = train.get("logging", {})

    return {
        "train_data_path": ds.get("train_data_path"),
        "verify_data_path": ds.get("verify_data_path"),
        "test_data_path": ds.get("test_data_path"),
        "view_num": env.get("view_num", 33),
        "observation_space_dim": env.get("observation_space_dim", 1024),
        "terminated_coverage": env.get("terminated_coverage", 0.97),
        "viewpoints_path": env["viewpoints_path"],
        "sun_position_config": env.get("sun_position", {}),
        "step_size": train.get("step_size", 10),
        "is_ratio_reward": train.get("is_ratio_reward", 1),
        "output_file": out.get("output_file", "train_result.txt"),
        "log_file": log.get("log_file", "random_coverage.log"),
    }


def choose_action(env, sample_unvisited=True):
    if env.continuous_mode:
        # Continuous mode: just sample from action space
        return env.action_space.sample()

    if not sample_unvisited:
        return env.action_space.sample()

    unvisited = np.where(env.view_state == 0)[0]
    if unvisited.size == 0:
        return env.action_space.sample()
    return int(random.choice(unvisited.tolist()))


def run_random_coverage(env, step_size, episode_count, logger, sample_unvisited=True):
    per_step_coverage_sum = np.zeros(step_size, dtype=np.float64)
    final_coverages = []

    for ep in range(episode_count):
        obs = env.reset(init_step=-1)
        episode_coverages = np.zeros(step_size, dtype=np.float64)
        episode_coverages[0] = env.current_coverage

        for step_idx in range(1, step_size):
            action = choose_action(env, sample_unvisited=sample_unvisited)
            obs, reward, done, info = env.step(action)
            episode_coverages[step_idx] = info["current_coverage"]
            if done:
                # Keep the trajectory length fixed to step_size for fair averaging.
                episode_coverages[step_idx:] = info["current_coverage"]
                break

        per_step_coverage_sum += episode_coverages
        final_coverages.append(float(episode_coverages[-1]))
        logger.info(
            "Episode {:4d} | final coverage: {:.2f}%".format(
                ep + 1, episode_coverages[-1] * 100
            )
        )

    avg_per_step = (per_step_coverage_sum / float(episode_count)) * 100.0
    avg_final = float(np.mean(final_coverages) * 100.0) if final_coverages else 0.0
    return avg_per_step, avg_final


def append_results(output_file, avg_per_step, avg_final, episode_count, step_size):
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "a+", encoding="utf-8") as f:
        f.write("------ Random Policy Coverage ------\n")
        f.write("episodes: {} | step_size: {}\n".format(episode_count, step_size))
        f.write("average_coverage: ")
        for i in range(step_size):
            f.write("[{}]:{:.2f} ".format(i + 1, avg_per_step[i]))
        f.write("\n")
        f.write("average_final_coverage: {:.2f}\n".format(avg_final))


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
        "--episodes",
        type=int,
        default=1,
        help="Number of 10-step random-policy episodes",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=10,
        help="Steps per episode (default: 10)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "verify", "test"],
        default="test",
        help="Dataset split path from config",
    )
    parser.add_argument(
        "--allow-repeat-actions",
        action="store_true",
        help="If set, random actions may revisit already selected views",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and print resolved values without running episodes",
    )
    cli = parser.parse_args()

    if not os.path.exists(cli.config):
        raise FileNotFoundError("Config not found: {}".format(cli.config))

    config = load_config(cli.config)
    args = argparse.Namespace(**config_to_args(config))

    split_to_path = {
        "train": args.train_data_path,
        "verify": args.verify_data_path,
        "test": args.test_data_path,
    }
    data_path = split_to_path[cli.split]

    step_size = int(cli.step_size)
    if step_size <= 0:
        raise ValueError("step-size must be > 0")
    if cli.episodes <= 0:
        raise ValueError("episodes must be > 0")

    if cli.dry_run:
        print("Config: {}".format(cli.config))
        print("  {:30s}: {}".format("split", cli.split))
        print("  {:30s}: {}".format("data_path", data_path))
        print("  {:30s}: {}".format("episodes", cli.episodes))
        print("  {:30s}: {}".format("step_size", step_size))
        print(
            "  {:30s}: {}".format("sample_unvisited", str(not cli.allow_repeat_actions))
        )
        for arg, value in sorted(vars(args).items()):
            print("  {:30s}: {}".format(arg, value))
        sys.exit(0)

    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    logger = setup_logger(args.log_file)

    logger.info("=" * 60)
    logger.info("RANDOM COVERAGE EVALUATION START")
    logger.info("=" * 60)
    logger.info("split             : {}".format(cli.split))
    logger.info("data_path         : {}".format(data_path))
    logger.info("episodes          : {}".format(cli.episodes))
    logger.info("step_size         : {}".format(step_size))
    logger.info("sample_unvisited  : {}".format(not cli.allow_repeat_actions))

    env = envs.rl_nbv_env.PointCloudNextBestViewEnv(
        data_path=data_path,
        viewpoints_path=args.viewpoints_path,
        view_num=args.view_num,
        observation_space_dim=args.observation_space_dim,
        terminated_coverage=args.terminated_coverage,
        logger=logger.getChild("random_env"),
        is_ratio_reward=(args.is_ratio_reward == 1),
        max_step=step_size,
        sun_position_config=args.sun_position_config,
    )

    avg_per_step, avg_final = run_random_coverage(
        env=env,
        step_size=step_size,
        episode_count=cli.episodes,
        logger=logger,
        sample_unvisited=(not cli.allow_repeat_actions),
    )

    append_results(args.output_file, avg_per_step, avg_final, cli.episodes, step_size)

    logger.info(
        "Average coverage by step: {}".format(
            " ".join(
                "[{}]:{:.2f}".format(i + 1, avg_per_step[i]) for i in range(step_size)
            )
        )
    )
    logger.info("Average final coverage: {:.2f}%".format(avg_final))
    logger.info("Results appended to: {}".format(args.output_file))
    logger.info("=" * 60)
    logger.info("ALL DONE")
    logger.info("=" * 60)
