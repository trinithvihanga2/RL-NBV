import argparse
import os
from PIL import Image

from envs.rl_nbv_env import PointCloudNextBestViewEnv
from stable_baselines3 import PPO

def main():
    parser = argparse.ArgumentParser(description="Visually debug the RL agent")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PPO model (.zip)")
    parser.add_argument("--data_path", type=str, default="data/shapenet", help="Path to ShapeNet dataset")
    parser.add_argument("--output_dir", type=str, default="debug_renders", help="Directory to save PNGs")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to visualize")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    env_config = config.get("environment", {})

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Environment
    env_kwargs = {
        "data_path": args.data_path,
        "render_mode": "rgb_array",
        "observation_space_dim": env_config.get("observation_space_dim", 1024),
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
    }
    
    # We use a single env instead of a vectorized env for easier rendering extraction
    env = PointCloudNextBestViewEnv(**env_kwargs)
    
    print(f"Loading model from {args.model_path}")
    model = PPO.load(args.model_path, env=env)

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        step = 0
        
        # Save initial state
        img_arr = env.render()
        if img_arr is not None:
            Image.fromarray(img_arr).save(os.path.join(args.output_dir, f"ep_{ep}_step_{step:02d}.png"))

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            img_arr = env.render()
            if img_arr is not None:
                save_path = os.path.join(args.output_dir, f"ep_{ep}_step_{step:02d}.png")
                Image.fromarray(img_arr).save(save_path)
                print(f"Saved {save_path} - Coverage: {info.get('current_coverage', 0)*100:.1f}%")
                
        print(f"Episode {ep} finished at step {step} with final coverage: {info.get('current_coverage', 0)*100:.1f}%")

if __name__ == "__main__":
    main()
