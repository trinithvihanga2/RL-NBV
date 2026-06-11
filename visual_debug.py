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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Environment
    env_kwargs = {
        "data_path": args.data_path,
        "render_mode": "rgb_array",
        "is_normalize": True,
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
