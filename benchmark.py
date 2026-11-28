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
try:
    from baselines.classical_nbv import (
        VolumetricGreedyNBVPolicy,
        CostAwareGreedyNBVPolicy,
    )
except ImportError:
    VolumetricGreedyNBVPolicy = None
    CostAwareGreedyNBVPolicy = None

# Canonical Time Unit (TU) conversion factor: 1 m/s = 765.04 m/TU
TIME_UNIT_S = 765.04


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
        logger.propagate = False

        # Attach file handler to root logger and environment loggers so all subsystem logs are preserved
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)

        train_logger = logging.getLogger("train")
        train_logger.setLevel(logging.DEBUG)
        train_logger.addHandler(file_handler)
        train_logger.propagate = False
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
    """Heuristic baseline scanning from the initial camera direction to its antipode along a 3-revolution spiral."""

    def __init__(self, steps_per_episode: int):
        self.steps = max(1, int(steps_per_episode))
        self.current_step = 0
        self._basis: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def reset(self) -> None:
        self.current_step = 0
        self._basis = None

    def _build_basis_from_obs(self, obs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a right-handed orthonormal frame (e1, e2, e3) with e3 along the initial camera position."""
        e3 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if isinstance(obs, dict) and "camera_position" in obs and obs["camera_position"] is not None:
            vec = np.asarray(obs["camera_position"], dtype=np.float64).reshape(-1)
            if vec.size >= 3 and np.all(np.isfinite(vec[:3])):
                norm = float(np.linalg.norm(vec[:3]))
                if norm > 1e-8:
                    e3 = vec[:3] / norm

        # Avoid near-collinearity when e3 aligns with the global Z-axis.
        ref = (
            np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(e3[2]) < 0.9
            else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        )
        e1 = np.cross(ref, e3)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(e3, e1)
        return (e1, e2, e3)

    def predict(self, obs: Any, deterministic: bool = True):
        del deterministic

        if self._basis is None:
            self._basis = self._build_basis_from_obs(obs)

        # Advance from step 1 so the first action leaves the initial pole rather than duplicating it.
        progress = min(1.0, float(self.current_step + 1) / float(self.steps))
        alpha = np.pi * progress
        total_rotations = 3
        beta = 2.0 * np.pi * total_rotations * progress

        e1, e2, e3 = self._basis
        target_dir = np.cos(alpha) * e3 + np.sin(alpha) * (
            np.cos(beta) * e1 + np.sin(beta) * e2
        )
        target_dir = target_dir / max(float(np.linalg.norm(target_dir)), 1e-12)

        # Map unit target direction to normalized environment spherical action [-1, 1].
        theta_rad = float(np.arccos(np.clip(target_dir[2], -1.0, 1.0)))
        theta = float(np.clip((theta_rad / (np.pi / 2.0)) - 1.0, -1.0, 1.0))
        phi_rad = float(np.arctan2(target_dir[1], target_dir[0]) % (2.0 * np.pi))
        phi = float(np.clip((phi_rad / np.pi) - 1.0, -1.0, 1.0))

        # Pace transfer time (~0.5 TU/step across 30 steps for 2-orbit horizon)
        time_action = -0.67

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
        "collision_penalty_weight": float(env_config.get("collision_penalty_weight", 25.0)),
        "collision_check_samples": int(env_config.get("collision_check_samples", 32)),
        "sun_position_config": env_config.get("sun_position", {}),
        "target_orbit_config": target_orbit_cfg,
        "state_reward_config": env_config.get("state_reward", {}),
        "scp_planner_config": scp_planner_cfg,
        "domain_randomization_config": {"enabled": False},
        "reward_scale": float(env_config.get("reward_scale", 0.01)),
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


def get_orbit_attr(env: PointCloudNextBestViewEnv, attr: str, default: float) -> float:
    """Read a numeric attribute from env.orbit_config whether object- or dict-based."""
    orbit_config = getattr(env, "orbit_config", None)
    if orbit_config is None:
        return float(default)
    if isinstance(orbit_config, dict):
        return float(orbit_config.get(attr, default))
    return float(getattr(orbit_config, attr, default))


def get_total_time(env: PointCloudNextBestViewEnv) -> float:
    """Read total mission time from either an object-style or dict config."""
    return get_orbit_attr(env, "total_time", 0.0)


def filter_ppo_action_with_scvx(
    env: PointCloudNextBestViewEnv, raw_action: np.ndarray
) -> tuple[np.ndarray, int, bool]:
    """Verify a PPO action against CW/SCvx feasibility and KOZ/mesh clearance, searching safe fallbacks if needed."""
    base_action = np.clip(normalise_action(raw_action)[:3], -1.0, 1.0).astype(np.float32)

    cw = getattr(env, "cw", None)
    renderer = getattr(env, "renderer", None)
    if cw is None or renderer is None:
        return base_action, 0, False

    norm_theta = float(base_action[0])
    norm_phi = float(base_action[1])
    norm_time = float(base_action[2])

    r = get_orbit_attr(env, "orbit_radius", 1.0)
    t_min = get_orbit_attr(env, "min_transfer_time", 0.2)
    t_max = get_orbit_attr(env, "max_transfer_time", 2.0)
    unit_scale = get_orbit_attr(env, "unit_scale", 1.0)
    current_time = float(getattr(env, "current_time", 0.0))
    remaining_time = max(0.0, get_total_time(env) - current_time)

    current_position = np.asarray(
        getattr(env, "current_position", np.array([r, 0.0, 0.0], dtype=np.float32)),
        dtype=np.float32,
    ).reshape(3)
    r0 = current_position * unit_scale

    def _evaluate(cand_action: np.ndarray) -> tuple[bool, bool, float, np.ndarray]:
        c_theta = float(np.clip(cand_action[0], -1.0, 1.0))
        c_phi = float(np.clip(cand_action[1], -1.0, 1.0))
        c_time = float(np.clip(cand_action[2], -1.0, 1.0))

        theta = (c_theta + 1.0) * (np.pi / 2.0)
        phi = (c_phi + 1.0) * np.pi
        requested_time = t_min + 0.5 * (c_time + 1.0) * (t_max - t_min)

        if remaining_time <= 0.0:
            travel_time = 0.0
        else:
            upper_bound = max(t_min, remaining_time)
            travel_time = float(np.clip(requested_time, t_min, upper_bound))

        new_position = np.array(
            [
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta),
            ],
            dtype=np.float32,
        )
        rf = new_position * unit_scale
        delta_v, _, _ = cw.compute_delta_v(r0, rf, travel_time)

        traj_canonical = (
            cw.last_trajectory / unit_scale
            if (hasattr(cw, "last_trajectory") and cw.last_trajectory is not None)
            else None
        )
        collision_detected, _, min_clearance = renderer.check_transfer_collision(
            current_position, new_position, travel_time, trajectory=traj_canonical
        )
        feasible = bool(np.isfinite(delta_v) and delta_v < np.inf)
        clearance_val = (
            float(min_clearance)
            if (min_clearance is not None and np.isfinite(min_clearance))
            else -float("inf")
        )
        return (feasible and not bool(collision_detected)), feasible, clearance_val, new_position

    scvx_calls = 1
    is_safe, base_feasible, base_clearance, base_new_position = _evaluate(base_action)
    if is_safe:
        return base_action, scvx_calls, False

    candidates: list[np.ndarray] = []
    seen: set[tuple[float, float, float]] = {
        (round(norm_theta, 6), round(norm_phi, 6), round(norm_time, 6))
    }

    def _add_candidate(c_theta: float, c_phi: float, c_time: float) -> None:
        c_theta = float(np.clip(c_theta, -1.0, 1.0))
        if c_phi < -1.0 or c_phi > 1.0:
            c_phi = float(((c_phi + 1.0) % 2.0) - 1.0)
        else:
            c_phi = float(c_phi)
        c_time = float(np.clip(c_time, -1.0, 1.0))
        key = (round(c_theta, 6), round(c_phi, 6), round(c_time, 6))
        if key not in seen:
            seen.add(key)
            candidates.append(np.array([c_theta, c_phi, c_time], dtype=np.float32))

    # a) Same (norm_theta, norm_phi) with relaxed transfer times
    for cand_time in (-0.4, 0.0, 0.5, 1.0):
        _add_candidate(norm_theta, norm_phi, cand_time)

    # b) Small angular offsets at relaxed transfer times + spherical midpoint direction
    for d_theta in (-0.25, 0.25, -0.5, 0.5, 0.0):
        for d_phi in (-0.25, 0.25, -0.5, 0.5, 0.0):
            if d_theta == 0.0 and d_phi == 0.0:
                continue
            for cand_time in (-0.3, 0.2):
                _add_candidate(norm_theta + d_theta, norm_phi + d_phi, cand_time)

    u0 = current_position.astype(np.float64)
    u0_norm = float(np.linalg.norm(u0))
    u0 = u0 / u0_norm if u0_norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

    u1 = base_new_position.astype(np.float64)
    u1_norm = float(np.linalg.norm(u1))
    u1 = u1 / u1_norm if u1_norm > 1e-8 else -u0

    mid = u0 + u1
    mid_norm = float(np.linalg.norm(mid))
    if mid_norm > 1e-6:
        mid_dir = mid / mid_norm
    else:
        ref = (
            np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(u0[2]) < 0.9
            else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        )
        mid_dir = np.cross(ref, u0)
        mid_dir = mid_dir / max(float(np.linalg.norm(mid_dir)), 1e-12)

    mid_theta_rad = float(np.arccos(np.clip(mid_dir[2], -1.0, 1.0)))
    mid_norm_theta = float(np.clip((mid_theta_rad / (np.pi / 2.0)) - 1.0, -1.0, 1.0))
    mid_phi_rad = float(np.arctan2(mid_dir[1], mid_dir[0]) % (2.0 * np.pi))
    mid_norm_phi = float(np.clip((mid_phi_rad / np.pi) - 1.0, -1.0, 1.0))
    _add_candidate(mid_norm_theta, mid_norm_phi, 0.0)

    best_finite_action: np.ndarray | None = base_action if base_feasible else None
    best_finite_clearance: float = base_clearance if base_feasible else -float("inf")
    best_overall_action: np.ndarray = base_action
    best_overall_clearance: float = base_clearance

    for cand_action in candidates:
        scvx_calls += 1
        cand_safe, cand_feasible, cand_clearance, _ = _evaluate(cand_action)
        if cand_safe:
            return cand_action, scvx_calls, True
        if cand_feasible and (
            best_finite_action is None or cand_clearance > best_finite_clearance
        ):
            best_finite_action = cand_action
            best_finite_clearance = cand_clearance
        if cand_clearance > best_overall_clearance:
            best_overall_action = cand_action
            best_overall_clearance = cand_clearance

    fallback_action = (
        best_finite_action if best_finite_action is not None else best_overall_action
    )
    return fallback_action, scvx_calls, True


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

    config_fuel_budget_tu = config_fuel_budget * TIME_UNIT_S

    return {
        "dataset_split": split_name,
        "policy": policy_name,
        "model_checkpoint": model_checkpoint,
        "model_name": model_name,
        "loop_id": loop_id,
        "config_fuel_budget": config_fuel_budget,
        "config_fuel_budget_tu": config_fuel_budget_tu,
        "config_num_orbits": config_num_orbits,
        "config_max_step": config_max_step,
        "config_koz_radius": config_koz_radius,
        "step": 0,
        "coverage": coverage,
        "coverage_gain": 0.0,
        "cumulative_dv": 0.0,
        "cumulative_dv_tu": 0.0,
        "fuel_remaining": config_fuel_budget,
        "fuel_remaining_tu": config_fuel_budget_tu,
        "fuel_consumed_fraction": 0.0,
        "step_travel_time": 0.0,
        "mission_time": 0.0,
        "time_remaining": max(0.0, total_time - mission_time),
        "reward": 0.0,
        "delta_v": 0.0,
        "delta_v_tu": 0.0,
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
        "policy_predict_ms": 0.0,
        "decision_latency_ms": 0.0,
        "scvx_calls": 0,
        "safety_filter_triggered": False,
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
    config_fuel_budget = float(
        config_params.get(
            "fuel_budget", getattr(env, "fuel_budget", 76504.0) / TIME_UNIT_S
        )
    )
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
            current_coverage = scalar(
                info.get("current_coverage"),
                getattr(env, "current_coverage", 0.0),
            )
            cum_dv_ms = 0.0

            while not (terminated or truncated):
                if step >= max_steps:
                    break

                scvx_counter = 0
                safety_filter_triggered = False
                cw = getattr(env, "cw", None)
                cw_dict = getattr(cw, "__dict__", None)
                had_instance_compute_dv = cw is not None and isinstance(cw_dict, dict) and "compute_delta_v" in cw_dict
                orig_compute_delta_v = getattr(cw, "compute_delta_v", None) if cw is not None else None

                if cw is not None and callable(orig_compute_delta_v):
                    def _counted_compute_delta_v(*args: Any, **kwargs: Any) -> Any:
                        nonlocal scvx_counter
                        scvx_counter += 1
                        return orig_compute_delta_v(*args, **kwargs)

                    cw.compute_delta_v = _counted_compute_delta_v

                try:
                    t_decision_start = time.perf_counter()
                    t_predict_start = t_decision_start
                    if policy_name == "Random":
                        action = env.action_space.sample()
                    else:
                        if policy is None:
                            raise ValueError(f"Policy object required for {policy_name}.")
                        action, _ = policy.predict(obs, deterministic=True)
                    t_predict_end = time.perf_counter()
                    policy_predict_ms = (t_predict_end - t_predict_start) * 1000.0

                    action = normalise_action(action)
                    if policy_name.startswith("PPO"):
                        action, _, safety_filter_triggered = filter_ppo_action_with_scvx(
                            env, action
                        )

                    t_decision_end = time.perf_counter()
                    decision_latency_ms = (t_decision_end - t_decision_start) * 1000.0
                finally:
                    if cw is not None and callable(orig_compute_delta_v):
                        if had_instance_compute_dv:
                            cw.compute_delta_v = orig_compute_delta_v
                        elif isinstance(cw_dict, dict) and "compute_delta_v" in cw_dict:
                            delattr(cw, "compute_delta_v")

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
                cum_dv_tu = scalar(
                    info.get("cumulative_dv"),
                    getattr(env, "cumulative_dv", 0.0),
                )
                cum_dv_ms = cum_dv_tu / TIME_UNIT_S
                step_dv_tu = scalar(info.get("delta_v"), 0.0)
                step_dv_ms = step_dv_tu / TIME_UNIT_S
                config_fuel_budget_tu = config_fuel_budget * TIME_UNIT_S
                fuel_consumed_frac = (
                    float(cum_dv_tu / config_fuel_budget_tu)
                    if config_fuel_budget_tu > 0
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
                        "config_fuel_budget_tu": config_fuel_budget_tu,
                        "config_num_orbits": config_num_orbits,
                        "config_max_step": max_steps,
                        "config_koz_radius": config_koz_radius,
                        "step": step,
                        "coverage": current_coverage,
                        "coverage_gain": current_coverage - previous_coverage,
                        "cumulative_dv": cum_dv_ms,
                        "cumulative_dv_tu": cum_dv_tu,
                        "fuel_remaining": max(0.0, config_fuel_budget - cum_dv_ms),
                        "fuel_remaining_tu": max(0.0, config_fuel_budget_tu - cum_dv_tu),
                        "fuel_consumed_fraction": fuel_consumed_frac,
                        "step_travel_time": scalar(info.get("travel_time"), 0.0),
                        "mission_time": mission_time,
                        "time_remaining": max(0.0, total_time - mission_time),
                        "reward": scalar(reward),
                        "delta_v": step_dv_ms,
                        "delta_v_tu": step_dv_tu,
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
                        "policy_predict_ms": float(policy_predict_ms),
                        "decision_latency_ms": float(decision_latency_ms),
                        "scvx_calls": int(scvx_counter),
                        "safety_filter_triggered": bool(safety_filter_triggered),
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
                    f"| Final Cov: {current_coverage * 100.0:6.2f}% | dV: {cum_dv_ms:6.2f} m/s "
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


# Default Curated Operational Matrix for Generalizability Evaluation (Physical Units: m/s)
DEFAULT_PARAMETER_MATRIX = [
    # ── Nominal Baseline (Trained Regime - 100 m/s physical, 2 orbits, 30 steps) ──
    {
        "fuel_budget": 100.0,
        "num_orbits": 2.0,
        "max_step": 30,
        "koz_radius": 0.95,
        "label": "Nominal_100ms_2orb_s30",
    },
    # ── Step Horizon Sensitivity (10, 15, and 50 steps at nominal 100 m/s) ──
    {
        "fuel_budget": 100.0,
        "num_orbits": 2.0,
        "max_step": 10,
        "koz_radius": 0.95,
        "label": "Horizon_100ms_2orb_s10",
    },
    {
        "fuel_budget": 100.0,
        "num_orbits": 2.0,
        "max_step": 15,
        "koz_radius": 0.95,
        "label": "Horizon_100ms_2orb_s15",
    },
    {
        "fuel_budget": 100.0,
        "num_orbits": 5.0,
        "max_step": 50,
        "koz_radius": 0.95,
        "label": "Horizon_100ms_5orb_s50",
    },
    # ── Fuel Budget Sweep: [0.1, 0.2, 0.3, 0.5, 1.0, 100.0] m/s (Steps=30, Orbits=2, KOZ=0.95) ──
    {
        "fuel_budget": 0.1,
        "num_orbits": 2.0,
        "max_step": 30,
        "koz_radius": 0.95,
        "label": "FuelSweep_0.1ms_2orb_s30",
    },
    {
        "fuel_budget": 0.2,
        "num_orbits": 2.0,
        "max_step": 30,
        "koz_radius": 0.95,
        "label": "FuelSweep_0.2ms_2orb_s30",
    },
    {
        "fuel_budget": 0.3,
        "num_orbits": 2.0,
        "max_step": 30,
        "koz_radius": 0.95,
        "label": "FuelSweep_0.3ms_2orb_s30",
    },
    {
        "fuel_budget": 0.5,
        "num_orbits": 2.0,
        "max_step": 30,
        "koz_radius": 0.95,
        "label": "FuelSweep_0.5ms_2orb_s30",
    },
    {
        "fuel_budget": 1.0,
        "num_orbits": 2.0,
        "max_step": 30,
        "koz_radius": 0.95,
        "label": "FuelSweep_1.0ms_2orb_s30",
    },
    # ── Safety Standoff (KOZ) Sensitivity Matrix at nominal 100 m/s ──
    {
        "fuel_budget": 100.0,
        "num_orbits": 2.0,
        "max_step": 30,
        "koz_radius": 0.85,
        "label": "KOZ_0.85_Tight",
    },
    {
        "fuel_budget": 100.0,
        "num_orbits": 2.0,
        "max_step": 30,
        "koz_radius": 1.05,
        "label": "KOZ_1.05_Relaxed",
    },
]


def print_summary_table(
    df: pd.DataFrame, logger: logging.Logger | None = None
) -> None:
    """Print and log a clean ASCII summary table of coverage, AUC, fuel, steps, collisions, and latency."""
    header = (
        f"{'Config Label':<26} | {'Split':<6} | {'Policy':<8} | {'Final Cov (%)':>13} | "
        f"{'Cov@10 (%)':>10} | {'AUC (%)':>8} | {'Mean dV (m/s)':>13} | {'Avg Steps':>9} | "
        f"{'St@90%':>7} | {'>=97% (%)':>9} | {'Collisions':>10} | {'Latency (ms)':>12}"
    )
    width = len(header)
    lines = [
        "\n" + "=" * width,
        f"{'BENCHMARK EVALUATION SUMMARY':^{width}}",
        "=" * width,
        header,
        "-" * width,
    ]

    if df is None or df.empty:
        lines.append("=" * width + "\n")
        for line in lines:
            if logger is not None:
                logger.info(line)
            else:
                print(line)
        return

    episode_keys = [
        col
        for col in [
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
        if col in df.columns
    ]

    episode_rows = []
    for _, ep_df in df.groupby(episode_keys, sort=False):
        ep_sorted = ep_df.sort_values("step")
        last_row = ep_sorted.iloc[-1]

        max_s = (
            int(last_row["config_max_step"])
            if "config_max_step" in ep_sorted.columns
            else int(ep_sorted["step"].max())
        )
        max_s = max(max_s, int(ep_sorted["step"].max()), 1)

        step_cov_map = {
            int(s): float(c) * 100.0
            for s, c in zip(ep_sorted["step"], ep_sorted["coverage"])
            if 0 <= int(s) <= max_s
        }
        ff_cov = np.zeros(max_s + 1, dtype=np.float64)
        last_cov = 0.0
        for s_idx in range(max_s + 1):
            if s_idx in step_cov_map:
                last_cov = step_cov_map[s_idx]
            ff_cov[s_idx] = last_cov

        cov_at_10 = float(ff_cov[min(10, max_s)])
        auc_pct = float(0.5 * (ff_cov[:-1] + ff_cov[1:]).sum() / max(max_s, 1))

        reached_90 = ep_sorted[ep_sorted["coverage"] >= 0.90]
        steps_to_90 = float(reached_90["step"].iloc[0]) if not reached_90.empty else np.nan
        hit_97 = 100.0 if float(last_row["coverage"]) >= 0.97 else 0.0

        if "collision_detected" in ep_sorted.columns:
            col_vals = ep_sorted["collision_detected"].fillna(False)
            if col_vals.dtype == object:
                col_vals = (
                    col_vals.astype(str)
                    .str.strip()
                    .str.lower()
                    .isin({"true", "1", "yes"})
                )
            ep_collisions = int(bool(col_vals.astype(bool).any()))
        else:
            ep_collisions = 0

        episode_rows.append(
            {
                "config_fuel_budget": float(last_row["config_fuel_budget"]),
                "config_num_orbits": float(last_row["config_num_orbits"]),
                "config_max_step": int(last_row["config_max_step"]),
                "config_koz_radius": float(last_row["config_koz_radius"]),
                "dataset_split": str(last_row["dataset_split"]),
                "policy": str(last_row["policy"]),
                "coverage": float(last_row["coverage"]),
                "cov_at_10": cov_at_10,
                "auc_pct": auc_pct,
                "cumulative_dv": float(last_row["cumulative_dv"]),
                "step": float(last_row["step"]),
                "steps_to_90": steps_to_90,
                "hit_97": hit_97,
                "collisions": ep_collisions,
            }
        )

    ep_summary_df = pd.DataFrame(episode_rows)
    group_cols = [
        "config_fuel_budget",
        "config_num_orbits",
        "config_max_step",
        "config_koz_radius",
        "dataset_split",
        "policy",
    ]
    summary = ep_summary_df.groupby(group_cols, as_index=False).agg(
        {
            "coverage": "mean",
            "cov_at_10": "mean",
            "auc_pct": "mean",
            "cumulative_dv": "mean",
            "step": "mean",
            "steps_to_90": "mean",
            "hit_97": "mean",
            "collisions": "sum",
        }
    )

    latency_map: dict[tuple[float, float, int, float, str, str], float] = {}
    if "decision_latency_ms" in df.columns:
        step_rows = df[df["step"] > 0]
        if not step_rows.empty:
            lat_series = step_rows.groupby(group_cols)["decision_latency_ms"].mean()
            for key_vals, mean_lat in lat_series.items():
                norm_key = (
                    float(key_vals[0]),
                    float(key_vals[1]),
                    int(key_vals[2]),
                    float(key_vals[3]),
                    str(key_vals[4]),
                    str(key_vals[5]),
                )
                latency_map[norm_key] = float(mean_lat) if pd.notna(mean_lat) else 0.0

    for _, row in summary.iterrows():
        cfg_name = f"{row['config_fuel_budget']:g}ms_{int(row['config_num_orbits'])}orb_s{int(row['config_max_step'])}_k{row['config_koz_radius']:.2f}"
        cov_pct = row["coverage"] * 100.0
        cov10_pct = row["cov_at_10"]
        auc_pct = row["auc_pct"]
        dv_val = row["cumulative_dv"]
        steps_val = row["step"]
        st90_str = f"{row['steps_to_90']:>7.1f}" if pd.notna(row["steps_to_90"]) else f"{'N/A':>7}"
        hit97_pct = row["hit_97"]
        collisions_val = int(row["collisions"])
        lat_key = (
            float(row["config_fuel_budget"]),
            float(row["config_num_orbits"]),
            int(row["config_max_step"]),
            float(row["config_koz_radius"]),
            str(row["dataset_split"]),
            str(row["policy"]),
        )
        latency_val = latency_map.get(lat_key, 0.0)
        lines.append(
            f"{cfg_name:<26} | {row['dataset_split']:<6} | {row['policy']:<8} | "
            f"{cov_pct:>12.2f}% | {cov10_pct:>9.2f}% | {auc_pct:>7.2f}% | "
            f"{dv_val:>13.2f} | {steps_val:>9.1f} | {st90_str} | {hit97_pct:>8.1f}% | "
            f"{collisions_val:>10d} | {latency_val:>12.2f}"
        )
    lines.append("=" * width + "\n")

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
        nargs="*",
        type=str,
        default=None,
        help="Path(s) to trained PPO checkpoint(s). Defaults to artefacts/train/final.zip or final.zip if omitted.",
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
    parser.add_argument(
        "--include_classical_baselines",
        "--eval_classical",
        dest="include_classical_baselines",
        action="store_true",
        default=True,
        help="Evaluate Volumetric Greedy NBV (G-NBV) and Cost-Aware Greedy NBV (CA-NBV) baselines (default: True).",
    )
    parser.add_argument(
        "--skip_classical_baselines",
        "--no_classical_baselines",
        dest="include_classical_baselines",
        action="store_false",
        help="Disable evaluation of classical NBV baselines.",
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

    if not args.model_paths:
        # Auto-detect final.zip in standard locations
        candidates = [
            "./artefacts/train/final.zip",
            "./artefacts/train/final",
            "./final.zip",
            "./final",
        ]
        found = None
        for cand in candidates:
            if model_file_exists(cand):
                found = cand
                break
        if found is not None:
            args.model_paths = [found]
        else:
            args.model_paths = ["./artefacts/train/final.zip"]

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
                    "label": f"Combo_{f_b:g}ms_{int(n_o)}orb_s{m_s}_koz{k_r}",
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
                        "label": f"Matrix_{f_b:g}ms_{int(n_o)}orb_step{m_s}_koz{k_r}",
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
                        "label": f"Config_{i+1}_{f_b:g}ms_{int(n_o)}orb_s{m_s}",
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
        f_budget_ms = float(cfg_params["fuel_budget"])
        f_budget_tu = f_budget_ms * TIME_UNIT_S
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
            f"=== Fuel: {f_budget_ms:g} m/s ({f_budget_tu:.1f} m/TU) | Orbits: {n_orbits} | Steps: {m_step} | KOZ: {k_radius} ==="
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
                        fuel_budget=f_budget_tu,
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

                    if args.include_classical_baselines:
                        if VolumetricGreedyNBVPolicy is None or CostAwareGreedyNBVPolicy is None:
                            logger.warning(
                                "Classical NBV baselines requested, but could not import from baselines.classical_nbv. Skipping."
                            )
                        else:
                            logger.info(f"Running Volumetric Greedy NBV Policy (G-NBV) on {split_name}...")
                            g_nbv_policy = VolumetricGreedyNBVPolicy(
                                env=env,
                                num_candidates=64,
                                orbit_radius=float(env.orbit_config.orbit_radius),
                            )
                            all_records.extend(
                                run_evaluation(
                                    env=env,
                                    policy=g_nbv_policy,
                                    split_name=split_name,
                                    policy_name="G-NBV",
                                    config_params=cfg_params,
                                    num_loops=args.loops,
                                    model_checkpoint="gnbv_baseline",
                                    logger=logger,
                                )
                            )

                            logger.info(f"Running Cost-Aware Greedy NBV Policy (CA-NBV) on {split_name}...")
                            ca_nbv_policy = CostAwareGreedyNBVPolicy(
                                env=env,
                                num_candidates=32,
                                orbit_radius=float(env.orbit_config.orbit_radius),
                                fuel_budget=f_budget_tu,
                                total_time=get_total_time(env),
                            )
                            all_records.extend(
                                run_evaluation(
                                    env=env,
                                    policy=ca_nbv_policy,
                                    split_name=split_name,
                                    policy_name="CA-NBV",
                                    config_params=cfg_params,
                                    num_loops=args.loops,
                                    model_checkpoint="canbv_baseline",
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