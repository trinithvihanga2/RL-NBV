import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
        "scp_planner_config": env_config.get("scp_planner", {}),
    }
    
    # We use a single env instead of a vectorized env for easier rendering extraction
    env = PointCloudNextBestViewEnv(**env_kwargs)
    
    print(f"Loading model from {args.model_path}")
    custom_objects = {
        "_last_obs": None,
        "action_space": env.action_space,
        "observation_space": env.observation_space,
    }
    model = PPO.load(args.model_path, env=env, custom_objects=custom_objects)

    for ep in range(args.episodes):
        print(f"Starting episode {ep}...", flush=True)
        obs, info = env.reset()
        done = False
        step = 0
        endpoints = [env.current_position.copy()]
        
        # Save initial state
        print("Rendering initial state...", flush=True)
        img_arr = env.render()
        if img_arr is not None:
            save_path = os.path.join(args.output_dir, f"ep_{ep}_step_{step:02d}.png")
            Image.fromarray(img_arr).save(save_path)
            print(f"Saved {save_path} - Initial", flush=True)

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            endpoints.append(env.current_position.copy())

            img_arr = env.render()
            if img_arr is not None:
                save_path = os.path.join(args.output_dir, f"ep_{ep}_step_{step:02d}.png")
                Image.fromarray(img_arr).save(save_path)
                
            fuel_pct = (info.get('fuel_remaining', 0) / env.fuel_budget) * 100
            time_pct = (max(0.0, env.orbit_config.total_time - info.get('mission_time', 0)) / env.orbit_config.total_time) * 100
            cov_pct = info.get('current_coverage', 0) * 100
            
            print(f"Step {step:02d} | Action: {np.array2string(action, precision=2, suppress_small=True)} "
                  f"| Reward: {reward:6.2f} | Cov: {cov_pct:4.1f}% | Fuel Rem: {fuel_pct:4.1f}% | Time Rem: {time_pct:4.1f}% "
                  f"| Pos: {np.array2string(env.current_position, precision=2)}", flush=True)
                
        print(f"Episode {ep} finished at step {step} with final coverage: {info.get('current_coverage', 0)*100:.1f}%", flush=True)
        
        # Plot trajectory on sphere
        print(f"Plotting trajectory for episode {ep}...", flush=True)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw wireframe sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        r = env.orbit_config.orbit_radius
        x = r * np.cos(u) * np.sin(v)
        y = r * np.sin(u) * np.sin(v)
        z = r * np.cos(v)
        ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)
        
        # Draw trajectory
        traj = np.array(env.full_trajectory)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r--', marker='o', markersize=4, linewidth=2, label='Trajectory')
        
        # Highlight start and end
        endpoints_arr = np.array(endpoints)
        ax.scatter(endpoints_arr[:, 0], endpoints_arr[:, 1], endpoints_arr[:, 2], color='orange', marker='X', s=100, label='Action Endpoints', zorder=4)
        
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='green', s=150, label='Start', zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='blue', s=150, label='End', zorder=5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Episode {ep} Trajectory (Coverage: {info.get("current_coverage", 0)*100:.1f}%)')
        ax.legend()
        
        traj_save_path = os.path.join(args.output_dir, f"ep_{ep}_trajectory.png")
        plt.savefig(traj_save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved trajectory plot to {traj_save_path}", flush=True)

if __name__ == "__main__":
    main()
