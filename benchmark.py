import argparse
import inspect
import itertools
import json
import logging
import os
import sys
import time
from typing import Any
import zipfile

import numpy as np
import pandas as pd
import torch
import yaml
from stable_baselines3 import PPO

# Custom imports required for SB3 model checkpoint deserialization
import models.pointnet2_cls_ssg  # noqa: F401
import optim.adamw  # noqa: F401
from envs.rl_nbv_env import PointCloudNextBestViewEnv


def setup_logger(log_file: str = "./artefacts/benchmark/benchmark.log") -> logging.Logger:
    """Configure comprehensive logger outputting to both console and log file."""
    log_dir = os.path.dirname(os.path.abspath(log_file))
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(
            f"Warning: Failed to create log directory {log_dir}: {e}. "
            "Falling back to ./benchmark.log"
        )
        log_file = "./benchmark.log"

    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (DEBUG level)
    try:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Attach file handler to root logger and environment loggers so all subsystem logs are preserved
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)

        train_logger = logging.getLogger("train")
        train_logger.setLevel(logging.DEBUG)
        train_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not attach file handler for {log_file}: {e}")

    return logger


def inspect_and_log_model(
    model_path: str, ppo_model: Any, logger: logging.Logger
) -> None:
    """Log file metadata, training progress, and neural network diagnostics to verify model integrity."""
    actual_path = model_path if os.path.isfile(model_path) else f"{model_path}.zip"
    abs_path = os.path.abspath(actual_path)
    file_size_mb = (
        os.path.getsize(abs_path) / (1024 * 1024)
        if os.path.exists(abs_path)
        else 0.0
    )
    mtime = (
        time.ctime(os.path.getmtime(abs_path))
        if os.path.exists(abs_path)
        else "Unknown"
    )
    stem = os.path.splitext(os.path.basename(model_path))[0]

    logger.info("=" * 80)
    logger.info(f"MODEL CHECKPOINT DIAGNOSTICS: {stem}")
    logger.info(f"  Path: {abs_path}")
    logger.info(f"  Size: {file_size_mb:.2f} MB")
    logger.info(f"  Last Modified: {mtime}")

    # Inspect internal SB3 archive metadata if available
    if os.path.isfile(abs_path) and zipfile.is_zipfile(abs_path):
        try:
            with zipfile.ZipFile(abs_path, "r") as archive:
                namelist = archive.namelist()
                if "data" in namelist:
                    data_bytes = archive.read("data")
                    data = json.loads(data_bytes.decode("utf-8"))
                    num_steps = data.get("num_timesteps")
                    tot_steps = data.get("_total_timesteps")
                    if num_steps is not None and tot_steps is not None:
                        pct = (num_steps / tot_steps * 100.0) if tot_steps > 0 else 0.0
                        logger.info(
                            f"  Training Steps: {num_steps:,} / {tot_steps:,} ({pct:.1f}%)"
                        )
                        if num_steps < tot_steps * 0.5:
                            logger.warning(
                                f"  ⚠️  CHECKPOINT WARNING: Model only trained for {num_steps:,} "
                                f"steps ({pct:.1f}% of target {tot_steps:,}). This is likely an incomplete run!"
                            )
                    gamma = data.get("gamma")
                    n_steps = data.get("n_steps")
                    batch_size = data.get("batch_size")
                    logger.info(
                        f"  Hyperparameters: gamma={gamma}, n_steps={n_steps}, batch_size={batch_size}"
                    )

                if "system_info.txt" in namelist:
                    sys_info = archive.read("system_info.txt").decode("utf-8", errors="ignore").strip().splitlines()
                    for line in sys_info:
                        if any(k in line for k in ["PyTorch:", "GPU Enabled:", "OS:"]):
                            logger.info(f"  Train System: {line.strip('- ')}")
        except Exception as err:
            logger.debug(f"  Could not read SB3 archive metadata: {err}")

    if stem == "final":
        logger.warning(
            "⚠️  WARNING: Checkpoint name is 'final' (artefacts/train/final.zip). "
            "An early training run that stalled at ~37% coverage was saved as 'final.zip'. "
            "If this run underperforms, verify if 'final_500_5.zip' (which achieved >80% coverage) "
            "was intended instead!"
        )
        alt_path = os.path.join(os.path.dirname(abs_path), "final_500_5.zip")
        if os.path.exists(alt_path):
            logger.warning(
                f"  Found trained alternative checkpoint: {alt_path}"
            )

    try:
        policy_net = ppo_model.policy
        total_params = sum(p.numel() for p in policy_net.parameters())
        trainable_params = sum(
            p.numel() for p in policy_net.parameters() if p.requires_grad
        )
        logger.info(
            f"  Total Parameters: {total_params:,} (Trainable: {trainable_params:,})"
        )

        if hasattr(policy_net, "action_net"):
            action_w = policy_net.action_net.weight.data
            action_b = (
                policy_net.action_net.bias.data
                if policy_net.action_net.bias is not None
                else None
            )
            w_norm = float(torch.norm(action_w))
            logger.info(f"  action_net weight L2-norm: {w_norm:.6f}")
            if action_b is not None:
                bias_vals = action_b.cpu().numpy().tolist()
                logger.info(
                    f"  action_net bias: [theta={bias_vals[0]:.4f}, phi={bias_vals[1]:.4f}, time={bias_vals[2]:.4f}]"
                )
    except Exception as exc:
        logger.debug(f"  Could not extract detailed network parameters: {exc}")
    logger.info("=" * 80)


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

        # Sweep polar angle from north to south
        theta = -1.0 + 2.0 * progress

        # Complete three azimuth rotations during the episode
        total_rotations = 3
        phi_progression = progress * total_rotations * 2.0
        phi = (phi_progression % 2.0) - 1.0

        # Maximize speed convention
        time_action = -1.0

        self.current_step += 1
        action = np.array([theta, phi, time_action], dtype=np.float32)
        return action, None


def as_bool(value: Any) -> bool:
    """Convert YAML booleans or legacy 0/1 configuration values to bool."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def create_env(
    data_path: str,
    config: dict,
    fuel_budget: float,
    num_orbits: float,
    max_step: int,
    koz_radius: float = 0.95,
    logger: logging.Logger | None = None,
) -> PointCloudNextBestViewEnv:
    """Create an environment with operational parameters explicitly configured via matrix."""
    env_config = config.get("environment", {})

    target_orbit_cfg = (
        env_config.get("target_orbit", {}).copy()
        if isinstance(env_config.get("target_orbit"), dict)
        else {}
    )
    target_orbit_cfg["num_orbits"] = float(num_orbits)

    scp_planner_cfg = (
        env_config.get("scp_planner", {}).copy()
        if isinstance(env_config.get("scp_planner"), dict)
        else {}
    )
    if koz_radius is not None:
        scp_planner_cfg["koz_radius"] = float(koz_radius)

    env_kwargs = {
        "data_path": data_path,
        "observation_space_dim": env_config.get("observation_space_dim", 1024),
        "terminated_coverage": env_config.get("terminated_coverage", 0.97),
        "max_step": int(max_step),
        "is_ratio_reward": as_bool(env_config.get("is_ratio_reward", 1)),
        "is_reward_with_cur_coverage": as_bool(
            env_config.get("is_reward_with_cur_coverage", 0)
        ),
        "cur_coverage_ratio": env_config.get("cur_coverage_ratio", 1.0),
        "time_cost_weight": env_config.get("time_cost_weight", 1.0),
        "fuel_budget": float(fuel_budget),
        "delta_v_weight": env_config.get("delta_v_weight", 1.0),
        "sun_position_config": env_config.get("sun_position", {}),
        "target_orbit_config": target_orbit_cfg,
        "state_reward_config": env_config.get("state_reward", {}),
        "scp_planner_config": scp_planner_cfg,
    }
    if logger is not None:
        env_kwargs["logger"] = logger

    # Filter out unsupported kwargs if the environment constructor signature changes
    signature = inspect.signature(PointCloudNextBestViewEnv.__init__)
    parameters = signature.parameters
    accepts_arbitrary_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    if not accepts_arbitrary_kwargs:
        unsupported = sorted(key for key in env_kwargs if key not in parameters)
        if unsupported:
            msg = f"PointCloudNextBestViewEnv does not accept {unsupported}; omitting them."
            if logger is not None:
                logger.warning(msg)
            else:
                print(f"Warning: {msg}")
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
            f"Policy returned {action_array.size} action values; expected at least {expected_size}."
        )

    return action_array


def initial_record(
    env: PointCloudNextBestViewEnv,
    info: dict,
    split_name: str,
    policy_name: str,
    model_name: str,
    loop_id: int,
    config_fuel_budget: float,
    config_num_orbits: float,
    config_max_step: int,
    config_koz_radius: float,
    model_checkpoint: str = "N/A",
) -> dict:
    total_time = get_total_time(env)
    mission_time = scalar(info.get("mission_time"), 0.0)
    coverage = scalar(
        info.get("current_coverage"),
        getattr(env, "current_coverage", 0.0),
    )
    cam_x = get_vector_component(env, "current_position", 0)
    cam_y = get_vector_component(env, "current_position", 1)
    cam_z = get_vector_component(env, "current_position", 2)
    view_dist = (
        float(np.sqrt(cam_x**2 + cam_y**2 + cam_z**2))
        if not np.isnan(cam_x)
        else np.nan
    )

    return {
        "dataset_split": split_name,
        "policy": policy_name,
        "model_checkpoint": model_checkpoint,
        "model_name": model_name,
        "loop_id": loop_id,
        "config_fuel_budget": config_fuel_budget,
        "config_num_orbits": config_num_orbits,
        "config_max_step": config_max_step,
        "config_koz_radius": config_koz_radius,
        "step": 0,
        "coverage": coverage,
        "coverage_gain": 0.0,
        "cumulative_dv": 0.0,
        "fuel_remaining": config_fuel_budget,
        "fuel_consumed_fraction": 0.0,
        "step_travel_time": 0.0,
        "mission_time": 0.0,
        "time_remaining": max(0.0, total_time - mission_time),
        "reward": 0.0,
        "delta_v": 0.0,
        "action_theta": np.nan,
        "action_phi": np.nan,
        "action_time": np.nan,
        "camera_x": cam_x,
        "camera_y": cam_y,
        "camera_z": cam_z,
        "viewpoint_distance": view_dist,
        "sun_x": get_vector_component(env, "current_sun_position", 0),
        "sun_y": get_vector_component(env, "current_sun_position", 1),
        "sun_z": get_vector_component(env, "current_sun_position", 2),
        "collision_detected": False,
        "collision_min_clearance": np.nan,
        "is_terminated": False,
        "is_truncated": False,
    }


def run_evaluation(
    env: PointCloudNextBestViewEnv,
    policy: Any,
    split_name: str,
    policy_name: str,
    config_params: dict,
    num_loops: int = 1,
    model_checkpoint: str = "N/A",
    logger: logging.Logger | None = None,
) -> list[dict]:
    records = []
    model_num = int(env.shapenet_reader.model_num)

    if model_num <= 0:
        return records

    max_steps = int(config_params.get("max_step", getattr(env, "max_step", 30)))
    config_fuel_budget = float(config_params.get("fuel_budget", env.fuel_budget))
    config_num_orbits = float(config_params.get("num_orbits", 2.0))
    config_koz_radius = float(config_params.get("koz_radius", 0.95))

    for loop_id in range(num_loops):
        # The reader advances to the next model during reset, so set to model_num - 1
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
                    config_fuel_budget=config_fuel_budget,
                    config_num_orbits=config_num_orbits,
                    config_max_step=max_steps,
                    config_koz_radius=config_koz_radius,
                    model_checkpoint=model_checkpoint,
                )
            )

            terminated = False
            truncated = False
            step = 0

            while not (terminated or truncated):
                if step >= max_steps:
                    break

                if policy_name == "Random":
                    action = env.action_space.sample()
                else:
                    if policy is None:
                        raise ValueError(f"Policy object required for {policy_name}.")
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
                cum_dv = scalar(
                    info.get("cumulative_dv"),
                    getattr(env, "cumulative_dv", 0.0),
                )
                fuel_consumed_frac = (
                    float(cum_dv / config_fuel_budget)
                    if config_fuel_budget > 0
                    else 0.0
                )
                cam_x = get_vector_component(env, "current_position", 0)
                cam_y = get_vector_component(env, "current_position", 1)
                cam_z = get_vector_component(env, "current_position", 2)
                view_dist = (
                    float(np.sqrt(cam_x**2 + cam_y**2 + cam_z**2))
                    if not np.isnan(cam_x)
                    else np.nan
                )

                records.append(
                    {
                        "dataset_split": split_name,
                        "policy": policy_name,
                        "model_checkpoint": model_checkpoint,
                        "model_name": model_name,
                        "loop_id": loop_id,
                        "config_fuel_budget": config_fuel_budget,
                        "config_num_orbits": config_num_orbits,
                        "config_max_step": max_steps,
                        "config_koz_radius": config_koz_radius,
                        "step": step,
                        "coverage": current_coverage,
                        "coverage_gain": current_coverage - previous_coverage,
                        "cumulative_dv": cum_dv,
                        "fuel_remaining": scalar(
                            info.get("fuel_remaining"),
                            max(0.0, config_fuel_budget - cum_dv),
                        ),
                        "fuel_consumed_fraction": fuel_consumed_frac,
                        "step_travel_time": scalar(info.get("travel_time"), 0.0),
                        "mission_time": mission_time,
                        "time_remaining": max(0.0, total_time - mission_time),
                        "reward": scalar(reward),
                        "delta_v": scalar(info.get("delta_v"), 0.0),
                        "action_theta": float(action[0]),
                        "action_phi": float(action[1]),
                        "action_time": float(action[2]),
                        "camera_x": cam_x,
                        "camera_y": cam_y,
                        "camera_z": cam_z,
                        "viewpoint_distance": view_dist,
                        "sun_x": get_vector_component(
                            env, "current_sun_position", 0
                        ),
                        "sun_y": get_vector_component(
                            env, "current_sun_position", 1
                        ),
                        "sun_z": get_vector_component(
                            env, "current_sun_position", 2
                        ),
                        "collision_detected": bool(
                            info.get("collision_detected", False)
                        ),
                        "collision_min_clearance": scalar(
                            info.get("collision_min_clearance"), np.nan
                        ),
                        "is_terminated": terminated,
                        "is_truncated": truncated,
                        }
                )

            if logger is not None:
                coll_str = (
                    f" [COLLISION min_clr={info.get('collision_min_clearance', float('nan')):.4f}]"
                    if info.get("collision_detected", False)
                    else ""
                )
                logger.info(
                    f"  [{policy_name:<8}] {split_name:<5} | Model: {model_name:<22} "
                    f"| Final Cov: {current_coverage * 100.0:6.2f}% | dV: {cum_dv:6.2f} m/s "
                    f"| Steps: {step:2d}/{max_steps}{coll_str}"
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
        return sorted(
            partition_paths, key=lambda path: int(os.path.basename(path))
        )

    return [base_path]


def model_file_exists(model_path: str) -> bool:
    """Stable-Baselines3 accepts either a .zip path or its filename stem."""
    return os.path.isfile(model_path) or os.path.isfile(f"{model_path}.zip")


# Default Curated Operational Matrix for Generalizability Evaluation
DEFAULT_PARAMETER_MATRIX = [
    # In-Distribution Baseline (Trained Regime - 10 steps)
    {
        "fuel_budget": 100.0,
        "num_orbits": 2.0,
        "max_step": 10,
        "koz_radius": 0.95,
        "label": "InDist_100m_2orb_Step10",
    },
    # In-Distribution Horizon (Trained Regime - 30 steps)
    {
        "fuel_budget": 100.0,
        "num_orbits": 2.0,
        "max_step": 30,
        "koz_radius": 0.95,
        "label": "InDist_100m_2orb_Step30",
    },
    # Extended Operational Envelope (Out-of-Distribution)
    {
        "fuel_budget": 200.0,
        "num_orbits": 2.0,
        "max_step": 30,
        "koz_radius": 0.95,
        "label": "OOD_200m_2orb_Step30",
    },
    {
        "fuel_budget": 300.0,
        "num_orbits": 3.0,
        "max_step": 30,
        "koz_radius": 0.95,
        "label": "OOD_300m_3orb_Step30",
    },
    {
        "fuel_budget": 500.0,
        "num_orbits": 5.0,
        "max_step": 30,
        "koz_radius": 0.95,
        "label": "OOD_500m_5orb_Step30",
    },
    {
        "fuel_budget": 500.0,
        "num_orbits": 5.0,
        "max_step": 50,
        "koz_radius": 0.95,
        "label": "OOD_500m_5orb_Step50",
    },
    # Safety Standoff Sensitivity Matrix
    {
        "fuel_budget": 500.0,
        "num_orbits": 5.0,
        "max_step": 30,
        "koz_radius": 0.85,
        "label": "KOZ_0.85_Tight",
    },
    {
        "fuel_budget": 500.0,
        "num_orbits": 5.0,
        "max_step": 30,
        "koz_radius": 1.05,
        "label": "KOZ_1.05_Wide",
    },
]


def print_summary_table(
    df: pd.DataFrame, logger: logging.Logger | None = None
) -> None:
    """Print and log a clean ASCII summary table of final coverage across configs and policies."""
    lines = [
        "\n" + "=" * 90,
        f"{'BENCHMARK EVALUATION SUMMARY':^90}",
        "=" * 90,
        f"{'Config Label':<26} | {'Split':<6} | {'Policy':<8} | {'Final Cov (%)':<14} | {'Mean dV (m/s)':<14} | {'Avg Steps':<10}",
        "-" * 90,
    ]

    # Calculate final step metrics per episode
    episode_keys = [
        "dataset_split",
        "policy",
        "model_checkpoint",
        "model_name",
        "loop_id",
        "config_fuel_budget",
        "config_num_orbits",
        "config_max_step",
        "config_koz_radius",
    ]
    final_steps = df.sort_values("step").groupby(episode_keys, as_index=False).last()

    group_cols = [
        "config_fuel_budget",
        "config_num_orbits",
        "config_max_step",
        "config_koz_radius",
        "dataset_split",
        "policy",
    ]
    summary = final_steps.groupby(group_cols, as_index=False).agg({
        "coverage": "mean",
        "cumulative_dv": "mean",
        "step": "mean",
    })

    for _, row in summary.iterrows():
        cfg_name = f"{int(row['config_fuel_budget'])}m_{int(row['config_num_orbits'])}orb_s{int(row['config_max_step'])}_k{row['config_koz_radius']:.2f}"
        cov_pct = row["coverage"] * 100.0
        dv_val = row["cumulative_dv"]
        steps_val = row["step"]
        lines.append(
            f"{cfg_name:<26} | {row['dataset_split']:<6} | {row['policy']:<8} | {cov_pct:>12.2f}% | {dv_val:>12.2f} | {steps_val:>9.1f}"
        )
    lines.append("=" * 90 + "\n")

    for line in lines:
        if logger is not None:
            logger.info(line)
        else:
            print(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PPO agent and baseline policies across a parameter matrix "
            "to systematically evaluate generalizability."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (used for dataset/paths/model structure).",
    )
    parser.add_argument(
        "--model_path",
        "--model-path",
        "--model_paths",
        "--model-paths",
        dest="model_paths",
        nargs="+",
        type=str,
        required=True,
        help="Path(s) to trained PPO checkpoint(s).",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=str,
        default="./artefacts/benchmark",
        help="Output directory for benchmark CSV files.",
    )
    parser.add_argument(
        "--log_file",
        "--log-file",
        "--log_path",
        dest="log_file",
        type=str,
        default="./artefacts/benchmark/benchmark.log",
        help="Path to write comprehensive benchmark execution log file.",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=1,
        help="Number of evaluations per object and policy.",
    )

    # CLI Parameter Matrix Options
    parser.add_argument(
        "--fuel_budgets",
        nargs="+",
        type=float,
        default=None,
        help="List of fuel budgets to evaluate (e.g. 100 200 300 500).",
    )
    parser.add_argument(
        "--num_orbits",
        nargs="+",
        type=float,
        default=None,
        help="List of orbital period horizons to evaluate (e.g. 2 3 5).",
    )
    parser.add_argument(
        "--max_steps",
        nargs="+",
        type=int,
        default=None,
        help="List of max episode step bounds to evaluate (e.g. 10 30 50).",
    )
    parser.add_argument(
        "--koz_radii",
        nargs="+",
        type=float,
        default=None,
        help="List of KOZ standoff radii to evaluate (e.g. 0.85 0.95 1.05).",
    )
    parser.add_argument(
        "--combos",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Explicit parameter tuples in format 'fuel,orbits,koz' or "
            "'fuel,orbits,steps,koz' (e.g. --combos 100,2,10,0.95 500,5,30,0.95)."
        ),
    )
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="If set, evaluates the full Cartesian product grid of CLI parameter lists.",
    )

    args = parser.parse_args()

    logger = setup_logger(args.log_file)
    logger.info("==========================================================================")
    logger.info("          AUTONOMOUS SATELLITE INSPECTION (RL-NBV) BENCHMARK             ")
    logger.info("==========================================================================")
    logger.info(f"Log File: {os.path.abspath(args.log_file)}")
    logger.info(f"PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"Config File: {os.path.abspath(args.config)}")
    logger.info(f"Output Directory: {os.path.abspath(args.output_dir)}")
    logger.info(f"Model Path(s): {args.model_paths}")

    if args.loops < 1:
        logger.error("--loops must be at least 1.")
        parser.error("--loops must be at least 1.")

    if not os.path.isfile(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        parser.error(f"Configuration file not found: {args.config}")

    valid_model_paths = []
    for m_path in args.model_paths:
        if not model_file_exists(m_path):
            logger.error(
                f"PPO model not found: {m_path} (also checked {m_path}.zip)"
            )
            parser.error(
                f"PPO model not found: {m_path} (also checked {m_path}.zip)"
            )
        valid_model_paths.append(m_path)

    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_config = config.get("dataset", {})
    splits = {
        "Train": dataset_config.get("train_data_path", "./data/train"),
        "Val": dataset_config.get("verify_data_path", "./data/verify"),
        "Test": dataset_config.get("test_data_path", "./data/test"),
    }

    # Construct the operational configuration matrix
    if args.combos is not None:
        matrix_configs = []
        for combo_str in args.combos:
            parts = [float(p.strip()) for p in combo_str.split(",") if p.strip()]
            if len(parts) == 3:
                f_b, n_o, k_r = parts
                m_s = 30
            elif len(parts) == 4:
                f_b, n_o, m_s, k_r = parts
                m_s = int(m_s)
            else:
                raise ValueError(
                    f"Invalid combo format '{combo_str}'. Expected 'fuel,orbits,koz' or 'fuel,orbits,steps,koz'."
                )
            matrix_configs.append(
                {
                    "fuel_budget": f_b,
                    "num_orbits": n_o,
                    "max_step": m_s,
                    "koz_radius": k_r,
                    "label": f"Combo_{int(f_b)}m_{int(n_o)}orb_s{m_s}_koz{k_r}",
                }
            )
    elif (
        args.fuel_budgets is not None
        or args.num_orbits is not None
        or args.max_steps is not None
        or args.koz_radii is not None
    ):
        fuel_list = args.fuel_budgets or [100.0]
        orbit_list = args.num_orbits or [2.0]
        step_list = args.max_steps or [30]
        koz_list = args.koz_radii or [0.95]

        if args.grid_search:
            matrix_configs = []
            for f_b, n_o, m_s, k_r in itertools.product(
                fuel_list, orbit_list, step_list, koz_list
            ):
                matrix_configs.append(
                    {
                        "fuel_budget": f_b,
                        "num_orbits": n_o,
                        "max_step": m_s,
                        "koz_radius": k_r,
                        "label": f"Matrix_{int(f_b)}m_{int(n_o)}orb_step{m_s}_koz{k_r}",
                    }
                )
        else:
            matrix_configs = []
            max_len = max(
                len(fuel_list),
                len(orbit_list),
                len(step_list),
                len(koz_list),
            )
            for i in range(max_len):
                f_b = fuel_list[i % len(fuel_list)]
                n_o = orbit_list[i % len(orbit_list)]
                m_s = step_list[i % len(step_list)]
                k_r = koz_list[i % len(koz_list)]
                matrix_configs.append(
                    {
                        "fuel_budget": f_b,
                        "num_orbits": n_o,
                        "max_step": m_s,
                        "koz_radius": k_r,
                        "label": f"Config_{i+1}_{int(f_b)}m_{int(n_o)}orb_s{m_s}",
                    }
                )
    else:
        matrix_configs = DEFAULT_PARAMETER_MATRIX

    logger.info(
        f"\n🚀 System Generalizability Benchmark initialized with {len(matrix_configs)} matrix configurations "
        f"and {len(valid_model_paths)} model checkpoint(s):"
    )
    for idx, cfg in enumerate(matrix_configs, 1):
        logger.info(
            f"   [{idx}] {cfg['label']}: Fuel={cfg['fuel_budget']} m/s, "
            f"Orbits={cfg['num_orbits']}, MaxSteps={cfg['max_step']}, KOZ={cfg['koz_radius']}"
        )

    all_records = []

    for cfg_idx, cfg_params in enumerate(matrix_configs, 1):
        f_budget = cfg_params["fuel_budget"]
        n_orbits = cfg_params["num_orbits"]
        m_step = cfg_params["max_step"]
        k_radius = cfg_params["koz_radius"]
        cfg_label = cfg_params["label"]

        logger.info(
            "\n=========================================================================="
        )
        logger.info(
            f"=== MATRIX CONFIG [{cfg_idx}/{len(matrix_configs)}]: {cfg_label} ==="
        )
        logger.info(
            f"=== Fuel: {f_budget} m/s | Orbits: {n_orbits} | Steps: {m_step} | KOZ: {k_radius} ==="
        )
        logger.info(
            "=========================================================================="
        )

        for split_name, base_path in splits.items():
            logger.info(f"\n--- Evaluating Split: {split_name} ---")

            data_paths = get_data_paths(base_path)
            if not data_paths:
                logger.warning(
                    f"Data path {base_path} does not exist. Skipping."
                )
                continue

            for data_path in data_paths:
                logger.info(f"-> Processing partition: {data_path}")
                env = None

                try:
                    env = create_env(
                        data_path=data_path,
                        config=config,
                        fuel_budget=f_budget,
                        num_orbits=n_orbits,
                        max_step=m_step,
                        koz_radius=k_radius,
                        logger=logger,
                    )
                    model_num = int(env.shapenet_reader.model_num)

                    if model_num <= 0:
                        logger.warning(
                            f"No models found in {data_path}. Skipping."
                        )
                        continue

                    # Evaluate each PPO model checkpoint on this configuration
                    for m_idx, m_path in enumerate(valid_model_paths, 1):
                        ckpt_stem = os.path.splitext(os.path.basename(m_path))[0]
                        policy_tag = "PPO" if len(valid_model_paths) == 1 else f"PPO_{ckpt_stem}"

                        logger.info(
                            f"Loading PPO checkpoint [{m_idx}/{len(valid_model_paths)}]: {m_path}..."
                        )
                        custom_objects = {
                            "action_space": env.action_space,
                            "observation_space": env.observation_space,
                        }
                        ppo_model = PPO.load(
                            m_path,
                            custom_objects=custom_objects,
                            device="auto",
                        )

                        # Inspect model diagnostics on first load
                        if cfg_idx == 1 and split_name == list(splits.keys())[0] and data_path == data_paths[0]:
                            inspect_and_log_model(m_path, ppo_model, logger)

                        logger.info(
                            f"Running {policy_tag} ({ckpt_stem}) on {split_name} ({model_num} models)..."
                        )
                        all_records.extend(
                            run_evaluation(
                                env=env,
                                policy=ppo_model,
                                split_name=split_name,
                                policy_name=policy_tag,
                                config_params=cfg_params,
                                num_loops=args.loops,
                                model_checkpoint=ckpt_stem,
                                logger=logger,
                            )
                        )

                    logger.info(f"Running Random Policy on {split_name}...")
                    all_records.extend(
                        run_evaluation(
                            env=env,
                            policy=None,
                            split_name=split_name,
                            policy_name="Random",
                            config_params=cfg_params,
                            num_loops=args.loops,
                            model_checkpoint="random_baseline",
                            logger=logger,
                        )
                    )

                    logger.info(f"Running Spiral Baseline Policy on {split_name}...")
                    spiral_policy = SpiralPolicy(steps_per_episode=m_step)
                    all_records.extend(
                        run_evaluation(
                            env=env,
                            policy=spiral_policy,
                            split_name=split_name,
                            policy_name="Spiral",
                            config_params=cfg_params,
                            num_loops=args.loops,
                            model_checkpoint="spiral_baseline",
                            logger=logger,
                        )
                    )

                finally:
                    if env is not None and hasattr(env, "close"):
                        env.close()

    if not all_records:
        raise RuntimeError("Benchmark produced no records.")

    dataframe = pd.DataFrame(all_records)
    csv_matrix_path = os.path.join(args.output_dir, "benchmark_raw_data_matrix.csv")
    dataframe.to_csv(csv_matrix_path, index=False)

    # Also save standard benchmark_raw_data.csv for backward compatibility
    csv_standard_path = os.path.join(args.output_dir, "benchmark_raw_data.csv")
    if len(matrix_configs) == 1:
        dataframe.to_csv(csv_standard_path, index=False)

    logger.info(f"\n🎉 Benchmark complete! Raw matrix data saved to {csv_matrix_path}")
    logger.info(f"Total Rows Written: {len(dataframe)}")

    print_summary_table(dataframe, logger=logger)


if __name__ == "__main__":
    main()