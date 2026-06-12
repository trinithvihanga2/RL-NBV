import envs.rl_nbv_env
import models.pointnet2_cls_ssg
import numpy as np
import argparse
import yaml
import stable_baselines3
import stable_baselines3.common.vec_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import time
import os
import sys
import torch
import copy
import logging
import gc
import torch.backends.cudnn as cudnn
from torch.profiler import profile, record_function, ProfilerActivity
from custom_callback import NextBestViewCustomCallback
from typing import Callable
from tqdm import tqdm
import optim.adamw

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logger(log_file="train_detail.log"):
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
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


def setup_worker_logger(logger_name, log_file):
    """Create process-local logger handlers for SubprocVecEnv workers."""
    logger = logging.getLogger(logger_name)
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
    # Avoid propagation to root logger in worker processes.
    logger.propagate = False
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
    ppo = train.get("ppo", {})
    pre = train.get("pretrained", {})
    out = train.get("output", {})
    log = train.get("logging", {})

    args = {
        # Dataset
        "train_data_path": ds.get("train_data_path"),
        "verify_data_path": ds.get("verify_data_path"),
        "test_data_path": ds.get("test_data_path"),
        # Environment
        "observation_space_dim": env.get("observation_space_dim", 1024),
        "is_normalize": env.get("is_normalize", 1),
        "terminated_coverage": env.get("terminated_coverage", 0.97),
        "max_step": env.get("max_step", 11),
        "is_ratio_reward": env.get("is_ratio_reward", train.get("is_ratio_reward", 1)),
        "is_reward_with_cur_coverage": env.get("is_reward_with_cur_coverage", 0),
        "cur_coverage_ratio": env.get("cur_coverage_ratio", 1.0),
        "time_cost_weight": env.get("time_cost_weight", 1.0),
        "delta_v_weight": env.get("delta_v_weight", 1.0),
        "fuel_budget": env.get("fuel_budget", 50.0),
        "target_orbit_config": env.get("target_orbit", {}),
        "state_reward_config": env.get("state_reward", {}),
        "is_vec_env": env.get("is_vec_env", 0),
        "env_num": env.get("env_num", 8),
        "sun_position_config": env.get("sun_position", {}),
        # Training
        "step_size": train.get("step_size", 10),
        "is_profile": train.get("is_profile", 0),
        "resume": train.get("resume", 0),
        # PPO config
        "ppo_learning_rate": ppo.get("learning_rate", 3e-4),
        "ppo_n_steps": ppo.get("n_steps", 2048),
        "ppo_batch_size": ppo.get("batch_size", 256),
        "ppo_n_epochs": ppo.get("n_epochs", 10),
        "ppo_gamma": ppo.get("gamma", 0.99),
        "ppo_gae_lambda": ppo.get("gae_lambda", 0.95),
        "ppo_clip_range": ppo.get("clip_range", 0.2),
        "ppo_ent_coef": ppo.get("ent_coef", 0.01),
        "ppo_vf_coef": ppo.get("vf_coef", 0.5),
        "ppo_max_grad_norm": ppo.get("max_grad_norm", 0.5),
        "ppo_total_steps": ppo.get("total_steps", 1000000),
        # Pretrained
        "is_transform": pre.get("is_transform", 0),
        "pretrained_model_path": pre.get("model_path", "null"),
        "is_freeze_fe": pre.get("is_freeze_fe", 0),
        # Output
        "output_file": out.get("output_file", "train_result.txt"),
        "checkpoint_path": out.get("checkpoint_path", "checkpoints/rl_nbv"),
        "final_model_path": out.get("final_model_path", "rl_nbv"),
        "save_freq": out.get("save_freq", 10000),
        "eval_freq": out.get("eval_freq", 10000),
        "is_save_model": out.get("is_save_model", 1),
        # Logging
        "log_file": log.get("log_file", "train_detail.log"),
        "coverage_log_freq_normal": log.get("coverage_log_freq_normal"),
        "coverage_log_freq_profile": log.get("coverage_log_freq_profile"),
        "progress_log_interval": log.get("progress_log_interval", 10),
    }
    
    return args


# ============================================================================
# GPU MEMORY
# ============================================================================
def setup_gpu(device_str, logger):
    cudnn.benchmark = True
    cudnn.deterministic = False
    if torch.cuda.is_available():
        device_id = int(device_str.split(":")[-1]) if ":" in device_str else 0
        torch.cuda.set_device(device_id)
        gpu_name = torch.cuda.get_device_name(device_id)
        total_mem = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
        logger.info("GPU        : {}".format(gpu_name))
        logger.info("Total VRAM : {:.2f} GB".format(total_mem))
    else:
        logger.warning("CUDA not available — using CPU")


def clear_gpu_memory(logger):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    logger.debug("GPU memory cleared")


def log_gpu_memory(logger, tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        logger.info(
            "GPU Memory {} | Allocated: {:.2f}GB | Reserved: {:.2f}GB | Free: {:.2f}GB".format(
                tag, allocated, reserved, free
            )
        )


# ============================================================================
# COVERAGE
# ============================================================================
def caculate_average_coverage(env, model, step_size, output_file, logger):
    model_size = env.shapenet_reader.model_num
    average_coverage = np.zeros(step_size)
    target_coverage = float(env.terminated_coverage)
    reached_view_counts = []
    logger.info("Calculating average coverage over {} models...".format(model_size))

    for model_id in range(model_size):
        obs = env.reset()
        coverages = np.zeros(step_size)
        coverages[0] = env.current_coverage
        average_coverage[0] += coverages[0]
        for step_id in range(step_size - 1):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            coverages[step_id + 1] = info["current_coverage"]
            average_coverage[step_id + 1] += coverages[step_id + 1]

        reached_indices = np.where(coverages >= target_coverage)[0]
        if reached_indices.size > 0:
            reached_view_counts.append(int(reached_indices[0] + 1))

    average_coverage = (average_coverage / model_size) * 100
    avg_optimal_views = None
    if reached_view_counts:
        avg_optimal_views = float(np.mean(reached_view_counts))

    with open(output_file, "a+", encoding="utf-8") as f:
        f.write("average_coverage: ")
        for i in range(step_size):
            f.write("[{}]:{:.2f} ".format(i + 1, average_coverage[i]))
        f.write("\n")
        if avg_optimal_views is not None:
            f.write(
                "optimal_view_count(target={:.2f}%): {:.2f} (reached_models={}/{})\n".format(
                    target_coverage * 100,
                    avg_optimal_views,
                    len(reached_view_counts),
                    model_size,
                )
            )
        else:
            f.write(
                "optimal_view_count(target={:.2f}%): not_reached (reached_models=0/{})\n".format(
                    target_coverage * 100,
                    model_size,
                )
            )

    logger.info(
        "Average coverage: "
        + " ".join(
            "[{}]:{:.2f}".format(i + 1, average_coverage[i]) for i in range(step_size)
        )
    )
    if avg_optimal_views is not None:
        logger.info(
            "Optimal view count (target={:.2f}%): {:.2f} (reached_models={}/{})".format(
                target_coverage * 100,
                avg_optimal_views,
                len(reached_view_counts),
                model_size,
            )
        )
    else:
        logger.info(
            "Optimal view count (target={:.2f}%): not reached (reached_models=0/{})".format(
                target_coverage * 100,
                model_size,
            )
        )
    return average_coverage


# ============================================================================
# CHECKPOINT
# ============================================================================
def save_checkpoint(model, checkpoint_path, logger, timestep=None):
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    model.save(checkpoint_path)
    logger.info(
        "✅ Checkpoint saved: {} (timestep={})".format(
            checkpoint_path, timestep or "unknown"
        )
    )


def load_checkpoint(checkpoint_path, train_env, policy_kwargs, logger):
    if not os.path.exists(checkpoint_path + ".zip"):
        logger.warning("No checkpoint found at: {}".format(checkpoint_path))
        return None
    model = stable_baselines3.PPO.load(
        path=checkpoint_path,
        env=train_env,
        device=train_env.device if hasattr(train_env, 'device') else 'auto'
    )
    logger.info("✅ Resumed from checkpoint: {}".format(checkpoint_path))
    return model


# ============================================================================
# ENVIRONMENT
# ============================================================================
def make_env(data_path, env_id, logger_name, log_file, args):
    def _f():
        worker_logger = setup_worker_logger(logger_name, log_file)
        env = envs.rl_nbv_env.PointCloudNextBestViewEnv(
            data_path=data_path,
            observation_space_dim=args.observation_space_dim,
            is_normalize=(args.is_normalize == 1),
            terminated_coverage=args.terminated_coverage,
            max_step=args.max_step,
            env_id=env_id,
            logger=worker_logger,
            is_ratio_reward=(args.is_ratio_reward == 1),
            is_reward_with_cur_coverage=(args.is_reward_with_cur_coverage == 1),
            cur_coverage_ratio=args.cur_coverage_ratio,
            time_cost_weight=args.time_cost_weight,
            fuel_budget=args.fuel_budget,
            delta_v_weight=args.delta_v_weight,
            sun_position_config=args.sun_position_config,
            target_orbit_config=args.target_orbit_config,
            state_reward_config=args.state_reward_config,
        )
        return env

    return _f


# ============================================================================
# LEARNING RATE SCHEDULE
# ============================================================================
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        if progress_remaining > 0.05:
            return progress_remaining * initial_value
        return 0.05 * initial_value

    return func


# ============================================================================
# PROGRESS BAR CALLBACK
# ============================================================================
class ProgressCallback(BaseCallback):
    def __init__(
        self,
        total_timesteps: int,
        logger,
        log_interval: int = 10,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_logger = logger
        self.log_interval = max(1, int(log_interval))
        self.next_log_step = self.log_interval
        self.pbar = None
        self.use_tqdm = False

    def _on_training_start(self) -> None:
        self.use_tqdm = sys.stdout.isatty()
        if self.use_tqdm:
            self.pbar = tqdm(
                total=self.total_timesteps,
                desc="Training",
                unit="step",
            )
        else:
            self.progress_logger.info(
                "Training progress logging enabled (interval={} steps)".format(
                    self.log_interval
                )
            )

    def _on_step(self) -> bool:
        if self.use_tqdm and self.pbar is not None:
            delta = self.num_timesteps - self.pbar.n
            if delta > 0:
                self.pbar.update(delta)
            return True

        if (
            self.num_timesteps >= self.next_log_step
            or self.num_timesteps >= self.total_timesteps
        ):
            progress = min(
                100.0,
                100.0 * float(self.num_timesteps) / float(max(1, self.total_timesteps)),
            )
            self.progress_logger.info(
                "Training progress: {}/{} steps ({:.2f}%)".format(
                    self.num_timesteps,
                    self.total_timesteps,
                    progress,
                )
            )
            while self.num_timesteps >= self.next_log_step:
                self.next_log_step += self.log_interval
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


# ============================================================================
# PRETRAINED WEIGHT TRANSFER
# ============================================================================
def transfer_pretrained_weights(model, args, logger):
    if not os.path.exists(args.pretrained_model_path):
        logger.error(
            "Pretrained model not found: {}".format(args.pretrained_model_path)
        )
        raise FileNotFoundError(
            f"Pretrained model not found: {args.pretrained_model_path}"
        )

    checkpoint = torch.load(args.pretrained_model_path, weights_only=False)
    pretrained_dict = checkpoint["model_state_dict"]
    qnet_dict = model.policy.q_net.state_dict()
    update_dict = copy.deepcopy(qnet_dict)
    updated_keys = []
    missing_keys = []

    for key in sorted(pretrained_dict.keys()):
        if key in ("fc3.bias", "fc3.weight"):
            continue
        key_in_qnet = "features_extractor." + key
        if key_in_qnet in update_dict:
            update_dict[key_in_qnet] = pretrained_dict[key]
            updated_keys.append(key_in_qnet)
        else:
            missing_keys.append(key_in_qnet)

    model.policy.q_net.load_state_dict(update_dict)
    model.policy.q_net_target.load_state_dict(update_dict)
    model.q_net.load_state_dict(update_dict)
    model.q_net_target.load_state_dict(update_dict)

    logger.info(
        "Pretrained weights loaded. Updated: {}, Missing: {}".format(
            len(updated_keys), len(missing_keys)
        )
    )
    if missing_keys:
        logger.error("Missing keys: {}".format(missing_keys))
        raise RuntimeError(
            f"Not all pretrained weights were transferred. Missing: {missing_keys}"
        )
    return model


# ============================================================================
# FREEZE FEATURE EXTRACTOR
# ============================================================================
def freeze_feature_extractor(model, logger):
    for net in [
        model.policy.q_net,
        model.policy.q_net_target,
        model.q_net,
        model.q_net_target,
    ]:
        for layer in [
            net.features_extractor.sa1,
            net.features_extractor.sa2,
            net.features_extractor.sa3,
        ]:
            for param in layer.parameters():
                param.requires_grad = False
    logger.info("Feature extractor frozen (sa1, sa2, sa3)")


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
        help="Load config and print resolved values without training",
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

    # Ensure parent directories for log and output files exist to avoid crashes
    try:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    except Exception:
        pass
    try:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    except Exception:
        pass

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = setup_logger(args.log_file)
    logger.info("=" * 60)
    logger.info("TRAINING START")
    logger.info("=" * 60)
    for arg, value in sorted(vars(args).items()):
        logger.info("  {:30s}: {}".format(arg, value))
    logger.info("=" * 60)

    # ── GPU Setup ─────────────────────────────────────────────────────────────
    setup_gpu("cuda:0", logger)
    log_gpu_memory(logger, tag="[startup]")

    # ── Training config ───────────────────────────────────────────────────────
    if args.is_profile == 0:
        coverage_log_freq = (
            args.coverage_log_freq_normal
            if args.coverage_log_freq_normal is not None
            else args.eval_freq
        )
    else:
        coverage_log_freq = (
            args.coverage_log_freq_profile
            if args.coverage_log_freq_profile is not None
            else 200
        )
    total_steps = args.ppo_total_steps
    if args.is_profile == 1:
        total_steps = 2000
    logger.info("Total steps       : {}".format(total_steps))
    logger.info("Coverage log freq (eval) : {}".format(coverage_log_freq))

    # ── Environments ──────────────────────────────────────────────────────────
    logger.info("Building environments...")
    if args.is_vec_env:
        env_list = [
            make_env(
                args.train_data_path,
                i,
                "train.train_env_{}".format(i),
                args.log_file,
                args,
            )
            for i in range(args.env_num)
        ]
        train_env = stable_baselines3.common.vec_env.SubprocVecEnv(env_list)
        logger.info("VecEnv: {} workers".format(args.env_num))
    else:
        train_env = envs.rl_nbv_env.PointCloudNextBestViewEnv(
            data_path=args.train_data_path,
            observation_space_dim=args.observation_space_dim,
            is_normalize=(args.is_normalize == 1),
            terminated_coverage=args.terminated_coverage,
            max_step=args.max_step,
            logger=logger.getChild("train_env"),
            is_ratio_reward=(args.is_ratio_reward == 1),
            is_reward_with_cur_coverage=(args.is_reward_with_cur_coverage == 1),
            cur_coverage_ratio=args.cur_coverage_ratio,
            time_cost_weight=args.time_cost_weight,
            fuel_budget=args.fuel_budget,
            delta_v_weight=args.delta_v_weight,
            sun_position_config=args.sun_position_config,
            target_orbit_config=args.target_orbit_config,
            state_reward_config=args.state_reward_config,
        )

    verify_env = envs.rl_nbv_env.PointCloudNextBestViewEnv(
        data_path=args.verify_data_path,
        observation_space_dim=args.observation_space_dim,
        is_normalize=(args.is_normalize == 1),
        terminated_coverage=args.terminated_coverage,
        max_step=args.max_step,
        logger=logger.getChild("verify_env"),
        is_ratio_reward=(args.is_ratio_reward == 1),
        is_reward_with_cur_coverage=(args.is_reward_with_cur_coverage == 1),
        cur_coverage_ratio=args.cur_coverage_ratio,
        time_cost_weight=args.time_cost_weight,
        fuel_budget=args.fuel_budget,
        delta_v_weight=args.delta_v_weight,
        sun_position_config=args.sun_position_config,
        target_orbit_config=args.target_orbit_config,
        state_reward_config=args.state_reward_config,
    )
    test_env = envs.rl_nbv_env.PointCloudNextBestViewEnv(
        data_path=args.test_data_path,
        observation_space_dim=args.observation_space_dim,
        is_normalize=(args.is_normalize == 1),
        terminated_coverage=args.terminated_coverage,
        max_step=args.max_step,
        logger=logger.getChild("test_env"),
        is_ratio_reward=(args.is_ratio_reward == 1),
        is_reward_with_cur_coverage=(args.is_reward_with_cur_coverage == 1),
        cur_coverage_ratio=args.cur_coverage_ratio,
        time_cost_weight=args.time_cost_weight,
        fuel_budget=args.fuel_budget,
        delta_v_weight=args.delta_v_weight,
        sun_position_config=args.sun_position_config,
        target_orbit_config=args.target_orbit_config,
        state_reward_config=args.state_reward_config,
    )
    logger.info("Environments ready ✅")
    log_gpu_memory(logger, tag="[after envs]")

    # ── Policy ────────────────────────────────────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class=models.pointnet2_cls_ssg.PointNetFeatureExtractionContinuous,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        optimizer_class=optim.adamw.AdamW,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = None
    if args.resume == 1:
        logger.info("Resuming from: {}".format(args.checkpoint_path))
        model = load_checkpoint(args.checkpoint_path, train_env, policy_kwargs, logger)
        if model is None:
            logger.warning("No checkpoint — starting fresh")

    if model is None:
        logger.info("Creating new PPO model...")
        model = stable_baselines3.PPO(
            policy="MultiInputPolicy",
            env=train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=linear_schedule(args.ppo_learning_rate),
            n_steps=args.ppo_n_steps,
            batch_size=args.ppo_batch_size,
            n_epochs=args.ppo_n_epochs,
            gamma=args.ppo_gamma,
            gae_lambda=args.ppo_gae_lambda,
            clip_range=args.ppo_clip_range,
            ent_coef=args.ppo_ent_coef,
            vf_coef=args.ppo_vf_coef,
            max_grad_norm=args.ppo_max_grad_norm,
            verbose=1,
            device="cuda:0",
        )
        logger.info("PPO model created ✅")

    log_gpu_memory(logger, tag="[after model]")

    # ── Pretrained ────────────────────────────────────────────────────────────
    if args.is_transform == 1:
        logger.info("Transferring pretrained weights...")
        model = transfer_pretrained_weights(model, args, logger)
        if args.is_freeze_fe == 1:
            freeze_feature_extractor(model, logger)

    log_gpu_memory(logger, tag="[before training]")

    # ── Callback ──────────────────────────────────────────────────────────────
    custom_callback = NextBestViewCustomCallback(
        args.output_file,
        verify_env,
        test_env,
        check_freq=coverage_log_freq,
        best_model_path=os.path.join(args.checkpoint_path, "best_model_coverage"),
        save_freq=args.save_freq,
        save_path=args.checkpoint_path,
    )
    progress_callback = ProgressCallback(
        total_steps,
        logger=logger,
        log_interval=args.progress_log_interval,
    )
    callbacks = CallbackList([custom_callback, progress_callback])

    # ── Training ──────────────────────────────────────────────────────────────
    logger.info("Training for {} steps...".format(total_steps))
    start_time = time.time()

    try:
        if args.is_profile == 0:
            model.learn(total_steps, callback=callbacks)
        else:
            logger.info("Profiling mode enabled")
            with profile(
                activities=[ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
                with_modules=True,
            ) as prof:
                with record_function("train"):
                    model.learn(total_steps, callback=callbacks)
            logger.info(
                prof.key_averages().table(
                    sort_by="self_cuda_memory_usage", row_limit=400
                )
            )

    except KeyboardInterrupt:
        logger.warning("Interrupted! Saving emergency checkpoint...")
        save_checkpoint(
            model,
            args.checkpoint_path + "_interrupted",
            logger,
            timestep=model.num_timesteps,
        )

    elapsed = time.time() - start_time
    logger.info(
        "Training done in: {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed)))
    )
    log_gpu_memory(logger, tag="[after training]")

    # ── Save + Evaluate ───────────────────────────────────────────────────────
    if args.is_save_model == 1:
        save_checkpoint(
            model, args.final_model_path, logger, timestep=model.num_timesteps
        )

        with open(args.output_file, "a+", encoding="utf-8") as f:
            f.write("------ Before Save ------\n")
        caculate_average_coverage(
            test_env, model, args.step_size, args.output_file, logger
        )

        logger.info("Clearing GPU before reload...")
        del model
        clear_gpu_memory(logger)
        log_gpu_memory(logger, tag="[after del model]")

        model = stable_baselines3.PPO.load(
            path=args.final_model_path,
            env=train_env,
            device="cuda:0"
        )
        logger.info("Model reloaded ✅")

        with open(args.output_file, "a+", encoding="utf-8") as f:
            f.write("------ After Save ------\n")
        caculate_average_coverage(
            test_env, model, args.step_size, args.output_file, logger
        )

    clear_gpu_memory(logger)
    log_gpu_memory(logger, tag="[final]")

    logger.info("=" * 60)
    logger.info("ALL DONE ✅")
    logger.info("=" * 60)
