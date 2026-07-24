import os
import argparse
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import models.pointnet2_cls_ssg
from envs.rl_nbv_env import PointCloudNextBestViewEnv

class SpiralPolicy:
    """A heuristic baseline that systematically scans the object in a spiral pattern."""
    def __init__(self, steps_per_episode):
        self.steps = steps_per_episode
        self.current_step = 0
        
    def reset(self):
        self.current_step = 0
        
    def predict(self, obs, deterministic=True):
        # Theta slowly sweeps from North (-1.0) to South (1.0)
        theta = -1.0 + 2.0 * (self.current_step / max(1, self.steps - 1))
        
        # Phi rotates rapidly (e.g., 3 full rotations over the episode)
        total_rotations = 3
        phi_progression = (self.current_step / max(1, self.steps - 1)) * total_rotations * 2.0
        phi = (phi_progression % 2.0) - 1.0 # Wrap to [-1.0, 1.0]
        
        # Time constraint: default to moving as fast as possible to save mission time
        time_action = -1.0 
        
        self.current_step += 1
        return np.array([theta, phi, time_action], dtype=np.float32), None

def create_env(data_path, config):
    env_config = config.get("environment", {})
    env_kwargs = {
        "data_path": data_path,
        "observation_space_dim": env_config.get("observation_space_dim", 1024),
        "is_normalize": env_config.get("is_normalize", 1) == 1,
        "terminated_coverage": env_config.get("terminated_coverage", 0.97),
        "max_step": env_config.get("max_step", 30),
        "is_ratio_reward": env_config.get("is_ratio_reward", 1) == 1,
        "is_reward_with_cur_coverage": env_config.get("is_reward_with_cur_coverage", 0) == 1,
        "cur_coverage_ratio": env_config.get("cur_coverage_ratio", 1.0),
        "time_cost_weight": env_config.get("time_cost_weight", 1.0),
        "fuel_budget": env_config.get("fuel_budget", 50.0),
        "delta_v_weight": env_config.get("delta_v_weight", 1.0),
        "sun_position_config": env_config.get("sun_position", {}),
        "target_orbit_config": env_config.get("target_orbit", {}),
        "state_reward_config": env_config.get("state_reward", {}),
        "scp_planner_config": env_config.get("scp_planner", {}),
    }
    return PointCloudNextBestViewEnv(**env_kwargs)

def run_evaluation(env, policy, split_name, policy_name, num_loops=1):
    records = []
    model_num = env.shapenet_reader.model_num
    
    for loop_id in range(num_loops):
        env.shapenet_reader.set_model_id(model_num - 1)
        for model_id in range(model_num):
            obs, info = env.reset()
            cur_model_name = env.shapenet_reader.cur_model_name
            
            if hasattr(policy, 'reset'):
                policy.reset()
            
            done = False
            step = 0
            
            # Initial state record (Step 0)
            records.append({
                "dataset_split": split_name,
                "policy": policy_name,
                "model_name": cur_model_name,
                "loop_id": loop_id,
                "step": step,
                "coverage": info.get("current_coverage", 0.0),
                "coverage_gain": 0.0,
                "fuel_remaining": info.get("fuel_remaining", env.fuel_budget),
                "time_remaining": max(0.0, env.orbit_config.total_time - info.get("mission_time", 0.0)),
                "reward": 0.0,
                "delta_v": 0.0,
                "action_theta": None,
                "action_phi": None,
                "action_time": None,
                "camera_x": env.current_position[0],
                "camera_y": env.current_position[1],
                "camera_z": env.current_position[2],
                "sun_x": env.current_sun_position[0],
                "sun_y": env.current_sun_position[1],
                "sun_z": env.current_sun_position[2],
                "is_terminated": False,
                "is_truncated": False
            })
            
            while not done:
                if policy_name == "Random":
                    action = env.action_space.sample()
                else:
                    action, _ = policy.predict(obs, deterministic=True)
                
                prev_coverage = info.get("current_coverage", 0.0)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1
                
                cur_coverage = info.get("current_coverage", 0.0)
                
                records.append({
                    "dataset_split": split_name,
                    "policy": policy_name,
                    "model_name": cur_model_name,
                    "loop_id": loop_id,
                    "step": step,
                    "coverage": cur_coverage,
                    "coverage_gain": cur_coverage - prev_coverage,
                    "fuel_remaining": info.get("fuel_remaining", 0.0),
                    "time_remaining": max(0.0, env.orbit_config.total_time - info.get("mission_time", 0.0)),
                    "reward": reward,
                    "delta_v": info.get("delta_v", 0.0),
                    "action_theta": action[0],
                    "action_phi": action[1],
                    "action_time": action[2],
                    "camera_x": env.current_position[0],
                    "camera_y": env.current_position[1],
                    "camera_z": env.current_position[2],
                    "sun_x": env.current_sun_position[0],
                    "sun_y": env.current_sun_position[1],
                    "sun_z": env.current_sun_position[2],
                    "is_terminated": terminated,
                    "is_truncated": truncated
                })
    return records

def get_data_paths(base_path):
    """Returns a list of valid data paths. If the base path contains integer-named 
    subdirectories (like 0, 1, 2 for the partitioned train set), it returns them all.
    Otherwise, it returns the base_path itself."""
    paths = []
    has_partitions = False
    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            if item.isdigit() and os.path.isdir(os.path.join(base_path, item)):
                paths.append(os.path.join(base_path, item))
                has_partitions = True
    if not has_partitions:
        paths.append(base_path)
    return sorted(paths)

def main():
    parser = argparse.ArgumentParser(description="Benchmark the RL agent and Baseline Policies on Train/Val/Test splits.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PPO model (.zip)")
    parser.add_argument("--output_dir", type=str, default="./artefacts/benchmark", help="Directory to save raw benchmark CSVs")
    parser.add_argument("--loops", type=int, default=1, help="Number of times to evaluate each object")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    dataset_cfg = config.get("dataset", {})
    splits = {
        "Train": dataset_cfg.get("train_data_path", "./data/train"),
        "Val": dataset_cfg.get("verify_data_path", "./data/verify"),
        "Test": dataset_cfg.get("test_data_path", "./data/test"),
    }
    
    all_records = []
    ppo_model = None
    
    for split_name, base_path in splits.items():
        print(f"\n======================================")
        print(f"--- Evaluating Split: {split_name} ---")
        print(f"======================================")
        if not os.path.exists(base_path):
            print(f"Warning: Data path {base_path} does not exist. Skipping.")
            continue
            
        data_paths = get_data_paths(base_path)
        
        for data_path in data_paths:
            print(f"-> Processing partition: {data_path}")
            env = create_env(data_path, config)
            if env.shapenet_reader.model_num == 0:
                print(f"Warning: No models found in {data_path}. Skipping.")
                continue
            
            # 1. PPO Policy
            if ppo_model is None:
                print(f"Loading PPO model...")
                custom_objects = {
                    "action_space": env.action_space,
                    "observation_space": env.observation_space,
                }
                ppo_model = PPO.load(args.model_path, env=env, custom_objects=custom_objects)
            
            print(f"Running PPO on {split_name} dataset ({env.shapenet_reader.model_num} models)...")
            ppo_records = run_evaluation(env, ppo_model, split_name, "PPO", args.loops)
            all_records.extend(ppo_records)
            
            # 2. Random Policy
            print(f"Running Random Policy on {split_name} dataset...")
            random_records = run_evaluation(env, None, split_name, "Random", args.loops)
            all_records.extend(random_records)
            
            # 3. Spiral Heuristic Policy
            print(f"Running Spiral Baseline Policy on {split_name} dataset...")
            spiral_policy = SpiralPolicy(steps_per_episode=env.max_step)
            spiral_records = run_evaluation(env, spiral_policy, split_name, "Spiral", args.loops)
            all_records.extend(spiral_records)
        
    df = pd.DataFrame(all_records)
    csv_path = os.path.join(args.output_dir, "benchmark_raw_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nBenchmark complete! Raw data saved to {csv_path}")

if __name__ == "__main__":
    main()
