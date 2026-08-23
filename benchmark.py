import argparse
import inspect
import itertools
import os
from typing import Any

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO

# Custom imports required for SB3 model checkpoint deserialization
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
    object_scale: float = 0.8,
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
        "object_scale": float(object_scale),
        "sun_position_config": env_config.get("sun_position", {}),
        "target_orbit_config": target_orbit_cfg,
        "state_reward_config": env_config.get("state_reward", {}),
        "scp_planner_config": scp_planner_cfg,
    }

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
            print(
                f"Warning: PointCloudNextBestViewEnv does not accept {unsupported}; omitting them."
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
    config_object_scale: float = 0.8,
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
        "model_name": model_name,
        "loop_id": loop_id,
        "config_fuel_budget": config_fuel_budget,
        "config_num_orbits": config_num_orbits,
        "config_max_step": config_max_step,
        "config_koz_radius": config_koz_radius,
        "config_object_scale": config_object_scale,
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
) -> list[dict]:
    records = []
    model_num = int(env.shapenet_reader.model_num)

    if model_num <= 0:
        return records

    max_steps = int(config_params.get("max_step", getattr(env, "max_step", 30)))
    config_fuel_budget = float(config_params.get("fuel_budget", env.fuel_budget))
    config_num_orbits = float(config_params.get("num_orbits", 2.0))
    config_koz_radius = float(config_params.get("koz_radius", 0.95))
    config_object_scale = float(config_params.get("object_scale", getattr(env, "object_scale", 0.8)))

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
                    config_object_scale=config_object_scale,
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
                        "model_name": model_name,
                        "loop_id": loop_id,
                        "config_fuel_budget": config_fuel_budget,
                        "config_num_orbits": config_num_orbits,
                        "config_max_step": max_steps,
                        "config_koz_radius": config_koz_radius,
                        "config_object_scale": config_object_scale,
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
    # In-Distribution Baseline (Trained Regime: Scale 0.80)
    {
        "fuel_budget": 100.0,
        "num_orbits": 2.0,
        "max_step": 30,
        "koz_radius": 0.95,
        "object_scale": 0.80,
        "label": "InDist_100m_2orb",
    },
    # Extended Operational Envelope (Out-of-Distribution)
    {
        "fuel_budget": 200.0,
        "num_orbits": 2.0,
        "max_step": 30,
        "koz_radius": 0.95,
        "object_scale": 0.80,
        "label": "OOD_200m_2orb",
    },
    {
        "fuel_budget": 300.0,
        "num_orbits": 3.0,
        "max_step": 30,
        "koz_radius": 0.95,
        "object_scale": 0.80,
        "label": "OOD_300m_3orb",
    },
    {
        "fuel_budget": 500.0,
        "num_orbits": 5.0,
        "max_step": 30,
        "koz_radius": 0.95,
        "object_scale": 0.80,
        "label": "OOD_500m_5orb",
    },
    {
        "fuel_budget": 500.0,
        "num_orbits": 5.0,
        "max_step": 50,
        "koz_radius": 0.95,
        "object_scale": 0.80,
        "label": "OOD_500m_5orb_Step50",
    },
    # Safety Standoff Sensitivity Matrix
    {
        "fuel_budget": 100.0,
        "num_orbits": 2.0,
        "max_step": 30,
        "koz_radius": 0.85,
        "object_scale": 0.80,
        "label": "KOZ_0.85_Tight",
    },
    {
        "fuel_budget": 100.0,
        "num_orbits": 2.0,
        "max_step": 30,
        "koz_radius": 1.05,
        "object_scale": 0.80,
        "label": "KOZ_1.05_Wide",
    },
    # Object Scale Generalizability Matrix (Base Scale = 0.80, +/- 0.05)
    {
        "fuel_budget": 100.0,
        "num_orbits": 2.0,
        "max_step": 30,
        "koz_radius": 0.95,
        "object_scale": 0.75,
        "label": "ObjectScale_0.75_Small",
    },
    {
        "fuel_budget": 100.0,
        "num_orbits": 2.0,
        "max_step": 30,
        "koz_radius": 0.95,
        "object_scale": 0.85,
        "label": "ObjectScale_0.85_Large",
    },
]


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
        dest="model_path",
        type=str,
        required=True,
        help="Path to trained PPO checkpoint.",
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
        help="List of max episode step bounds to evaluate (e.g. 30 50).",
    )
    parser.add_argument(
        "--koz_radii",
        nargs="+",
        type=float,
        default=None,
        help="List of KOZ standoff radii to evaluate (e.g. 0.85 0.95 1.05).",
    )
    parser.add_argument(
        "--object_scales",
        nargs="+",
        type=float,
        default=None,
        help="List of object normalization scales to evaluate (e.g. 0.75 0.80 0.85).",
    )
    parser.add_argument(
        "--combos",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Explicit parameter tuples in format 'fuel,orbits,koz', "
            "'fuel,orbits,koz,scale' or 'fuel,orbits,steps,koz,scale' "
            "(e.g. --combos 100,2,0.95,0.75 100,2,0.95,0.85)."
        ),
    )
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="If set, evaluates the full Cartesian product grid of CLI parameter lists.",
    )

    args = parser.parse_args()

    if args.loops < 1:
        parser.error("--loops must be at least 1.")

    if not os.path.isfile(args.config):
        parser.error(f"Configuration file not found: {args.config}")

    if not model_file_exists(args.model_path):
        parser.error(
            f"PPO model not found: {args.model_path} (also checked {args.model_path}.zip)"
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

    # Construct the operational configuration matrix
    if args.combos is not None:
        matrix_configs = []
        for combo_str in args.combos:
            parts = [float(p.strip()) for p in combo_str.split(",") if p.strip()]
            if len(parts) == 3:
                f_b, n_o, k_r = parts
                m_s = 30
                o_s = 0.80
            elif len(parts) == 4:
                f_b, n_o, k_r, o_s = parts
                m_s = 30
            elif len(parts) == 5:
                f_b, n_o, m_s, k_r, o_s = parts
                m_s = int(m_s)
            else:
                raise ValueError(
                    f"Invalid combo format '{combo_str}'. Expected 'fuel,orbits,koz', "
                    "'fuel,orbits,koz,scale' or 'fuel,orbits,steps,koz,scale'."
                )
            matrix_configs.append(
                {
                    "fuel_budget": f_b,
                    "num_orbits": n_o,
                    "max_step": m_s,
                    "koz_radius": k_r,
                    "object_scale": o_s,
                    "label": f"Combo_{int(f_b)}m_{int(n_o)}orb_koz{k_r}_scale{o_s:.2f}",
                }
            )
    elif (
        args.fuel_budgets is not None
        or args.num_orbits is not None
        or args.max_steps is not None
        or args.koz_radii is not None
        or args.object_scales is not None
    ):
        fuel_list = args.fuel_budgets or [100.0]
        orbit_list = args.num_orbits or [2.0]
        step_list = args.max_steps or [30]
        koz_list = args.koz_radii or [0.95]
        scale_list = args.object_scales or [0.80]

        if args.grid_search:
            matrix_configs = []
            for f_b, n_o, m_s, k_r, o_s in itertools.product(
                fuel_list, orbit_list, step_list, koz_list, scale_list
            ):
                matrix_configs.append(
                    {
                        "fuel_budget": f_b,
                        "num_orbits": n_o,
                        "max_step": m_s,
                        "koz_radius": k_r,
                        "object_scale": o_s,
                        "label": f"Matrix_{int(f_b)}m_{int(n_o)}orb_step{m_s}_koz{k_r}_scale{o_s:.2f}",
                    }
                )
        else:
            matrix_configs = []
            max_len = max(
                len(fuel_list),
                len(orbit_list),
                len(step_list),
                len(koz_list),
                len(scale_list),
            )
            for i in range(max_len):
                f_b = fuel_list[i % len(fuel_list)]
                n_o = orbit_list[i % len(orbit_list)]
                m_s = step_list[i % len(step_list)]
                k_r = koz_list[i % len(koz_list)]
                o_s = scale_list[i % len(scale_list)]
                matrix_configs.append(
                    {
                        "fuel_budget": f_b,
                        "num_orbits": n_o,
                        "max_step": m_s,
                        "koz_radius": k_r,
                        "object_scale": o_s,
                        "label": f"Config_{i+1}_{int(f_b)}m_{int(n_o)}orb_scale{o_s:.2f}",
                    }
                )
    else:
        matrix_configs = DEFAULT_PARAMETER_MATRIX

    print(
        f"\n🚀 System Generalizability Benchmark initialized with {len(matrix_configs)} matrix configurations:"
    )
    for idx, cfg in enumerate(matrix_configs, 1):
        print(
            f"   [{idx}] {cfg['label']}: Fuel={cfg['fuel_budget']} m/s, "
            f"Orbits={cfg['num_orbits']}, MaxSteps={cfg['max_step']}, KOZ={cfg['koz_radius']}"
        )

    all_records = []
    ppo_model = None

    for cfg_idx, cfg_params in enumerate(matrix_configs, 1):
        f_budget = cfg_params["fuel_budget"]
        n_orbits = cfg_params["num_orbits"]
        m_step = cfg_params["max_step"]
        k_radius = cfg_params["koz_radius"]
        o_scale = cfg_params.get("object_scale", 0.80)
        cfg_label = cfg_params["label"]

        print(
            f"\n=========================================================================="
        )
        print(
            f"=== MATRIX CONFIG [{cfg_idx}/{len(matrix_configs)}]: {cfg_label} ==="
        )
        print(
            f"=== Fuel: {f_budget} m/s | Orbits: {n_orbits} | Steps: {m_step} | KOZ: {k_radius} | Scale: {o_scale} ==="
        )
        print(
            f"=========================================================================="
        )

        for split_name, base_path in splits.items():
            print(f"\n--- Evaluating Split: {split_name} ---")

            data_paths = get_data_paths(base_path)
            if not data_paths:
                print(
                    f"Warning: Data path {base_path} does not exist. Skipping."
                )
                continue

            for data_path in data_paths:
                print(f"-> Processing partition: {data_path}")
                env = None

                try:
                    env = create_env(
                        data_path=data_path,
                        config=config,
                        fuel_budget=f_budget,
                        num_orbits=n_orbits,
                        max_step=m_step,
                        koz_radius=k_radius,
                        object_scale=o_scale,
                    )
                    model_num = int(env.shapenet_reader.model_num)

                    if model_num <= 0:
                        print(
                            f"Warning: No models found in {data_path}. Skipping."
                        )
                        continue

                    if ppo_model is None:
                        print("Loading PPO model checkpoint...")
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
                        f"Running PPO on {split_name} ({model_num} models)..."
                    )
                    all_records.extend(
                        run_evaluation(
                            env=env,
                            policy=ppo_model,
                            split_name=split_name,
                            policy_name="PPO",
                            config_params=cfg_params,
                            num_loops=args.loops,
                        )
                    )

                    print(f"Running Random Policy on {split_name}...")
                    all_records.extend(
                        run_evaluation(
                            env=env,
                            policy=None,
                            split_name=split_name,
                            policy_name="Random",
                            config_params=cfg_params,
                            num_loops=args.loops,
                        )
                    )

                    print(f"Running Spiral Baseline Policy on {split_name}...")
                    spiral_policy = SpiralPolicy(steps_per_episode=m_step)
                    all_records.extend(
                        run_evaluation(
                            env=env,
                            policy=spiral_policy,
                            split_name=split_name,
                            policy_name="Spiral",
                            config_params=cfg_params,
                            num_loops=args.loops,
                        )
                    )

                finally:
                    if env is not None and hasattr(env, "close"):
                        env.close()

    if not all_records:
        raise RuntimeError("Benchmark produced no records.")

    dataframe = pd.DataFrame(all_records)
    csv_path = os.path.join(args.output_dir, "benchmark_raw_data_matrix.csv")
    dataframe.to_csv(csv_path, index=False)

    print(f"\n🎉 Benchmark complete! Raw matrix data saved to {csv_path}")
    print(f"Total Rows Written: {len(dataframe)}")


if __name__ == "__main__":
    main()