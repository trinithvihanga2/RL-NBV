import argparse
import inspect
import os
from typing import Any

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO

# These imports are needed so custom classes used by the saved PPO model
# are available when Stable-Baselines3 deserialises the checkpoint.
import models.pointnet2_cls_ssg  # noqa: F401
import optim.adamw  # noqa: F401
from envs.rl_nbv_env import PointCloudNextBestViewEnv


class SpiralPolicy:
    """Heuristic baseline that scans the object using a spiral trajectory."""

    def __init__(self, steps_per_episode: int):
        self.steps = max(1, int(steps_per_episode))
        self.current_step = 0

    def reset(self) -> None:
        self.current_step = 0

    def predict(self, obs: Any, deterministic: bool = True):
        del obs, deterministic

        denominator = max(1, self.steps - 1)
        progress = self.current_step / denominator

        # Sweep from north to south.
        theta = -1.0 + 2.0 * progress

        # Complete three azimuth rotations during the episode.
        total_rotations = 3
        phi_progression = progress * total_rotations * 2.0
        phi = (phi_progression % 2.0) - 1.0

        # Move as quickly as the action convention permits.
        time_action = -1.0

        self.current_step += 1
        action = np.array([theta, phi, time_action], dtype=np.float32)
        return action, None


def as_bool(value: Any) -> bool:
    """Convert YAML booleans or legacy 0/1 configuration values to bool."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def create_env(data_path: str, config: dict) -> PointCloudNextBestViewEnv:
    """Create an environment using only arguments supported by its current API."""
    env_config = config.get("environment", {})

    # `is_normalize` was intentionally removed. The current environment
    # constructor does not accept it, which caused:
    # TypeError: unexpected keyword argument 'is_normalize'
    env_kwargs = {
        "data_path": data_path,
        "observation_space_dim": env_config.get("observation_space_dim", 1024),
        "terminated_coverage": env_config.get("terminated_coverage", 0.97),
        "max_step": env_config.get("max_step", 30),
        "is_ratio_reward": as_bool(env_config.get("is_ratio_reward", 1)),
        "is_reward_with_cur_coverage": as_bool(
            env_config.get("is_reward_with_cur_coverage", 0)
        ),
        "cur_coverage_ratio": env_config.get("cur_coverage_ratio", 1.0),
        "time_cost_weight": env_config.get("time_cost_weight", 1.0),
        "fuel_budget": env_config.get("fuel_budget", 100.0),
        "delta_v_weight": env_config.get("delta_v_weight", 1.0),
        "sun_position_config": env_config.get("sun_position", {}),
        "target_orbit_config": env_config.get("target_orbit", {}),
        "state_reward_config": env_config.get("state_reward", {}),
        "scp_planner_config": env_config.get("scp_planner", {}),
    }

    # Protect the benchmark against other stale config keys after future
    # environment API changes. Unsupported keys are reported and omitted.
    signature = inspect.signature(PointCloudNextBestViewEnv.__init__)
    parameters = signature.parameters
    accepts_arbitrary_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    if not accepts_arbitrary_kwargs:
        unsupported = sorted(
            key for key in env_kwargs if key not in parameters
        )
        if unsupported:
            print(
                "Warning: PointCloudNextBestViewEnv does not accept "
                f"{unsupported}; omitting them."
            )
            env_kwargs = {
                key: value
                for key, value in env_kwargs.items()
                if key in parameters
            }

    return PointCloudNextBestViewEnv(**env_kwargs)


def scalar(value: Any, default: float = 0.0) -> float:
    """Convert scalar-like NumPy values to a regular Python float."""
    if value is None:
        return float(default)

    array = np.asarray(value)
    if array.size == 0:
        return float(default)

    return float(array.reshape(-1)[0])


def get_total_time(env: PointCloudNextBestViewEnv) -> float:
    """Read total mission time from either an object-style or dict config."""
    orbit_config = getattr(env, "orbit_config", None)

    if orbit_config is None:
        return 0.0

    if isinstance(orbit_config, dict):
        return float(orbit_config.get("total_time", 0.0))

    return float(getattr(orbit_config, "total_time", 0.0))


def get_vector_component(obj: Any, attribute: str, index: int) -> float:
    """Safely retrieve a component of a vector-valued environment attribute."""
    vector = getattr(obj, attribute, None)
    if vector is None:
        return float("nan")

    array = np.asarray(vector).reshape(-1)
    if index >= array.size:
        return float("nan")

    return float(array[index])


def normalise_action(action: Any, expected_size: int = 3) -> np.ndarray:
    """Return a flat floating-point action and validate its dimension."""
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)

    if action_array.size < expected_size:
        raise ValueError(
            f"Policy returned {action_array.size} action values; "
            f"expected at least {expected_size}."
        )

    return action_array


def initial_record(
    env: PointCloudNextBestViewEnv,
    info: dict,
    split_name: str,
    policy_name: str,
    model_name: str,
    loop_id: int,
) -> dict:
    total_time = get_total_time(env)
    mission_time = scalar(info.get("mission_time"), 0.0)
    coverage = scalar(
        info.get("current_coverage"),
        getattr(env, "current_coverage", 0.0),
    )

    return {
        "dataset_split": split_name,
        "policy": policy_name,
        "model_name": model_name,
        "loop_id": loop_id,
        "step": 0,
        "coverage": coverage,
        "coverage_gain": 0.0,
        "fuel_remaining": scalar(
            info.get("fuel_remaining"),
            getattr(env, "fuel_budget", 0.0),
        ),
        "time_remaining": max(0.0, total_time - mission_time),
        "reward": 0.0,
        "delta_v": 0.0,
        "action_theta": np.nan,
        "action_phi": np.nan,
        "action_time": np.nan,
        "camera_x": get_vector_component(env, "current_position", 0),
        "camera_y": get_vector_component(env, "current_position", 1),
        "camera_z": get_vector_component(env, "current_position", 2),
        "sun_x": get_vector_component(env, "current_sun_position", 0),
        "sun_y": get_vector_component(env, "current_sun_position", 1),
        "sun_z": get_vector_component(env, "current_sun_position", 2),
        "is_terminated": False,
        "is_truncated": False,
    }


def run_evaluation(
    env: PointCloudNextBestViewEnv,
    policy: Any,
    split_name: str,
    policy_name: str,
    num_loops: int = 1,
) -> list[dict]:
    records = []
    model_num = int(env.shapenet_reader.model_num)

    if model_num <= 0:
        return records

    max_steps = int(getattr(env, "max_step", 30))

    for loop_id in range(num_loops):
        # The reader advances to the next model during reset, so placing it at
        # the final model makes the first reset select model zero.
        env.shapenet_reader.set_model_id(model_num - 1)

        for _ in range(model_num):
            obs, info = env.reset()
            info = info or {}

            model_name = str(
                getattr(env.shapenet_reader, "cur_model_name", "unknown")
            )

            if policy is not None and hasattr(policy, "reset"):
                policy.reset()

            records.append(
                initial_record(
                    env=env,
                    info=info,
                    split_name=split_name,
                    policy_name=policy_name,
                    model_name=model_name,
                    loop_id=loop_id,
                )
            )

            terminated = False
            truncated = False
            step = 0

            while not (terminated or truncated):
                if step >= max_steps:
                    print(
                        f"Warning: {policy_name} exceeded max_step={max_steps} "
                        f"on model {model_name}; stopping the episode."
                    )
                    break

                if policy_name == "Random":
                    action = env.action_space.sample()
                else:
                    if policy is None:
                        raise ValueError(
                            f"Policy object is required for {policy_name}."
                        )
                    action, _ = policy.predict(obs, deterministic=True)

                action = normalise_action(action)
                previous_coverage = scalar(
                    info.get("current_coverage"),
                    getattr(env, "current_coverage", 0.0),
                )

                obs, reward, terminated, truncated, info = env.step(action)
                info = info or {}
                terminated = bool(terminated)
                truncated = bool(truncated)
                step += 1

                current_coverage = scalar(
                    info.get("current_coverage"),
                    getattr(env, "current_coverage", previous_coverage),
                )
                total_time = get_total_time(env)
                mission_time = scalar(info.get("mission_time"), 0.0)

                records.append(
                    {
                        "dataset_split": split_name,
                        "policy": policy_name,
                        "model_name": model_name,
                        "loop_id": loop_id,
                        "step": step,
                        "coverage": current_coverage,
                        "coverage_gain": current_coverage - previous_coverage,
                        "fuel_remaining": scalar(
                            info.get("fuel_remaining"),
                            getattr(env, "fuel_budget", 0.0),
                        ),
                        "time_remaining": max(
                            0.0, total_time - mission_time
                        ),
                        "reward": scalar(reward),
                        "delta_v": scalar(info.get("delta_v"), 0.0),
                        "action_theta": float(action[0]),
                        "action_phi": float(action[1]),
                        "action_time": float(action[2]),
                        "camera_x": get_vector_component(
                            env, "current_position", 0
                        ),
                        "camera_y": get_vector_component(
                            env, "current_position", 1
                        ),
                        "camera_z": get_vector_component(
                            env, "current_position", 2
                        ),
                        "sun_x": get_vector_component(
                            env, "current_sun_position", 0
                        ),
                        "sun_y": get_vector_component(
                            env, "current_sun_position", 1
                        ),
                        "sun_z": get_vector_component(
                            env, "current_sun_position", 2
                        ),
                        "is_terminated": terminated,
                        "is_truncated": truncated,
                    }
                )

    return records


def get_data_paths(base_path: str) -> list[str]:
    """Return all integer-named partitions, or the base path if unpartitioned."""
    if not os.path.isdir(base_path):
        return []

    partition_paths = [
        os.path.join(base_path, item)
        for item in os.listdir(base_path)
        if item.isdigit() and os.path.isdir(os.path.join(base_path, item))
    ]

    if partition_paths:
        return sorted(partition_paths, key=lambda path: int(os.path.basename(path)))

    return [base_path]


def model_file_exists(model_path: str) -> bool:
    """Stable-Baselines3 accepts either a .zip path or its filename stem."""
    return os.path.isfile(model_path) or os.path.isfile(f"{model_path}.zip")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the PPO agent and baseline policies on the "
            "train, validation, and test splits."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--model_path",
        "--model-path",
        dest="model_path",
        type=str,
        required=True,
        help="Path to the trained PPO model, with or without .zip.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=str,
        default="./artefacts/benchmark",
        help="Directory for benchmark CSV output.",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=1,
        help="Number of evaluations per object and policy.",
    )
    args = parser.parse_args()

    if args.loops < 1:
        parser.error("--loops must be at least 1.")

    if not os.path.isfile(args.config):
        parser.error(f"Configuration file not found: {args.config}")

    if not model_file_exists(args.model_path):
        parser.error(
            f"PPO model not found: {args.model_path} "
            f"(also checked {args.model_path}.zip)"
        )

    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_config = config.get("dataset", {})
    splits = {
        "Train": dataset_config.get("train_data_path", "./data/train"),
        "Val": dataset_config.get("verify_data_path", "./data/verify"),
        "Test": dataset_config.get("test_data_path", "./data/test"),
    }

    all_records = []
    ppo_model = None

    for split_name, base_path in splits.items():
        print("\n======================================")
        print(f"--- Evaluating Split: {split_name} ---")
        print("======================================")

        data_paths = get_data_paths(base_path)
        if not data_paths:
            print(f"Warning: Data path {base_path} does not exist. Skipping.")
            continue

        for data_path in data_paths:
            print(f"-> Processing partition: {data_path}")
            env = None

            try:
                env = create_env(data_path, config)
                model_num = int(env.shapenet_reader.model_num)

                if model_num <= 0:
                    print(f"Warning: No models found in {data_path}. Skipping.")
                    continue

                if ppo_model is None:
                    print("Loading PPO model...")
                    custom_objects = {
                        "action_space": env.action_space,
                        "observation_space": env.observation_space,
                    }
                    ppo_model = PPO.load(
                        args.model_path,
                        custom_objects=custom_objects,
                        device="auto",
                    )

                print(
                    f"Running PPO on {split_name} dataset "
                    f"({model_num} models)..."
                )
                all_records.extend(
                    run_evaluation(
                        env,
                        ppo_model,
                        split_name,
                        "PPO",
                        args.loops,
                    )
                )

                print(f"Running Random Policy on {split_name} dataset...")
                all_records.extend(
                    run_evaluation(
                        env,
                        None,
                        split_name,
                        "Random",
                        args.loops,
                    )
                )

                print(
                    f"Running Spiral Baseline Policy on "
                    f"{split_name} dataset..."
                )
                spiral_policy = SpiralPolicy(
                    steps_per_episode=getattr(env, "max_step", 30)
                )
                all_records.extend(
                    run_evaluation(
                        env,
                        spiral_policy,
                        split_name,
                        "Spiral",
                        args.loops,
                    )
                )

            finally:
                if env is not None and hasattr(env, "close"):
                    env.close()

    if not all_records:
        raise RuntimeError(
            "Benchmark produced no records. Check the configured dataset paths "
            "and whether the partitions contain models."
        )

    dataframe = pd.DataFrame(all_records)
    csv_path = os.path.join(args.output_dir, "benchmark_raw_data.csv")
    dataframe.to_csv(csv_path, index=False)

    print(f"\nBenchmark complete! Raw data saved to {csv_path}")
    print(f"Rows written: {len(dataframe)}")


if __name__ == "__main__":
    main()