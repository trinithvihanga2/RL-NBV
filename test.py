import envs.rl_nbv_env
import models.pointnet2_cls_ssg
import numpy as np
import argparse
import yaml
import stable_baselines3
import os
import torch
import logging
import torch.backends.cudnn as cudnn
import optim.adamw

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logger(log_file="train_detail.log"):
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
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
    ds    = config.get("dataset", {})
    env   = config.get("environment", {})
    train = config.get("training", {})
    dqn   = train.get("dqn", {})
    pre   = train.get("pretrained", {})
    rb    = train.get("replay_buffer", {})
    out   = train.get("output", {})
    log   = train.get("logging", {})

    return {
        "train_data_path":           ds.get("train_data_path"),
        "verify_data_path":          ds.get("verify_data_path"),
        "test_data_path":            ds.get("test_data_path"),
        "view_num":                  env.get("view_num", 33),
        "observation_space_dim":     env.get("observation_space_dim", 1024),
        "is_vec_env":                env.get("is_vec_env", 0),
        "env_num":                   env.get("env_num", 8),
        "step_size":                 train.get("step_size", 10),
        "is_ratio_reward":           train.get("is_ratio_reward", 1),
        "is_profile":                train.get("is_profile", 0),
        "resume":                    train.get("resume", 0),
        "device":                    dqn.get("device", "cuda:0"),
        "learning_rate":             dqn.get("learning_rate", 0.001),
        "batch_size":                dqn.get("batch_size", 128),
        "buffer_size":               dqn.get("buffer_size", 100000),
        "learning_starts":           dqn.get("learning_starts", 3000),
        "exploration_fraction":      dqn.get("exploration_fraction", 0.5),
        "exploration_final_eps":     dqn.get("exploration_final_eps", 0.2),
        "gradient_steps":            dqn.get("gradient_steps", 1),
        "train_freq":                dqn.get("train_freq", 16),
        "gamma":                     dqn.get("gamma", 0.1),
        "total_steps":               dqn.get("total_steps", 500000),
        "is_transform":              pre.get("is_transform", 0),
        "pretrained_model_path":     pre.get("model_path", "null"),
        "is_freeze_fe":              pre.get("is_freeze_fe", 0),
        "is_load_replay_buffer":     rb.get("is_load_replay_buffer", 0),
        "load_replay_buffer_path":   rb.get("load_replay_buffer_path", "null"),
        "is_save_replay_buffer":     rb.get("is_save_replay_buffer", 0),
        "save_replay_buffer_path":   rb.get("save_replay_buffer_path", "dqn_replay_buffer.pkl"),
        "output_file":               out.get("output_file", "train_result.txt"),
        "checkpoint_path":           out.get("checkpoint_path", "checkpoints/rl_nbv"),
        "final_model_path":          out.get("final_model_path", "rl_nbv"),
        "save_freq":                 out.get("save_freq", 10000),
        "eval_freq":                 out.get("eval_freq", 10000),
        "is_save_model":             out.get("is_save_model", 1),
        "log_file":                  log.get("log_file", "train_detail.log"),
        "coverage_log_freq_normal":  log.get("coverage_log_freq_normal"),
        "coverage_log_freq_profile": log.get("coverage_log_freq_profile"),
    }


# ============================================================================
# GPU SETUP
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


# ============================================================================
# COVERAGE  (model=None → random policy, model=<DQN> → trained policy)
# ============================================================================
def caculate_average_coverage(env, model, step_size, output_file, logger):
    """
    Runs one full evaluation pass over all test models.
    - If model is None  → actions are sampled randomly from the action space.
    - If model is a DQN → actions are chosen deterministically by the network.
    Returns per-step average coverage as a percentage (0-100).
    """
    label = "random policy" if model is None else "trained model"
    model_size = env.shapenet_reader.model_num
    average_coverage = np.zeros(step_size)
    init_step = 0

    logger.info("Calculating average coverage [{}] over {} models...".format(
        label, model_size))

    for model_id in range(model_size):
        obs = env.reset(init_step=init_step)
        init_step = (init_step + 1) % 33
        average_coverage[0] += env.current_coverage

        for step_id in range(step_size - 1):
            if model is None:
                action = env.action_space.sample()          # random
            else:
                action, _ = model.predict(obs, deterministic=True)  # trained

            obs, rewards, dones, info = env.step(action)
            average_coverage[step_id + 1] += info["current_coverage"]

    average_coverage = (average_coverage / model_size) * 100

    with open(output_file, "a+", encoding="utf-8") as f:
        f.write("{}: ".format(label))
        for i in range(step_size):
            f.write("[{}]:{:.2f} ".format(i + 1, average_coverage[i]))
        f.write("\n")

    logger.info(
        "[{}] Average coverage: ".format(label)
        + " ".join("[{}]:{:.2f}".format(i + 1, average_coverage[i])
                   for i in range(step_size))
    )
    return average_coverage


# ============================================================================
# COMPARISON REPORT
# ============================================================================
def compare_and_report(model_cov, random_cov, step_size, output_file, logger):
    header    = "\n{:<8} {:>14} {:>14} {:>12}".format(
                    "Step", "Model (%)", "Random (%)", "Gap (%)")
    separator = "-" * 52
    rows = []
    for i in range(step_size):
        gap = model_cov[i] - random_cov[i]
        rows.append("  [{:>2}]   {:>10.2f}   {:>10.2f}   {:>+.2f}".format(
            i + 1, model_cov[i], random_cov[i], gap))

    table = "\n".join([header, separator] + rows + [separator])
    logger.info(table)

    final_gap = model_cov[-1] - random_cov[-1]
    logger.info("Final step  |  Model: {:.2f}%  |  Random: {:.2f}%  |  Advantage: {:+.2f}%".format(
        model_cov[-1], random_cov[-1], final_gap))

    with open(output_file, "a+", encoding="utf-8") as f:
        f.write("gap (model - random): ")
        for i in range(step_size):
            f.write("[{}]:{:+.2f} ".format(i + 1, model_cov[i] - random_cov[i]))
        f.write("\n")

    return final_gap


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML config file")
    cli = parser.parse_args()

    if not os.path.exists(cli.config):
        raise FileNotFoundError("Config not found: {}".format(cli.config))

    config = load_config(cli.config)
    args   = argparse.Namespace(**config_to_args(config))

    logger = setup_logger(args.log_file)
    setup_gpu(args.device, logger)

    # Shared test environment
    test_env = envs.rl_nbv_env.PointCloudNextBestViewEnv(
        data_path=args.test_data_path,
        view_num=args.view_num,
        observation_space_dim=args.observation_space_dim,
        log_level=logging.INFO,
    )

    # Load trained DQN
    policy_kwargs = dict(
        features_extractor_class=models.pointnet2_cls_ssg.PointNetFeatureExtraction,
        features_extractor_kwargs=dict(features_dim=128),
        optimizer_class=optim.adamw.AdamW,
    )
    model = stable_baselines3.DQN.load(
        path=args.final_model_path,
        env=test_env,
        policy_kwargs=policy_kwargs,
        policy="MultiInputPolicy",
    )

    # ── STEP 1: Trained model ─────────────────────────────────────────────────
    logger.info("=" * 52)
    logger.info("STEP 1 — Trained model evaluation")
    logger.info("=" * 52)
    model_coverage = caculate_average_coverage(
        test_env, model, args.step_size, args.output_file, logger)

    # ── STEP 2: Random baseline (model=None) ──────────────────────────────────
    logger.info("=" * 52)
    logger.info("STEP 2 — Random policy evaluation")
    logger.info("=" * 52)
    random_coverage = caculate_average_coverage(
        test_env, None, args.step_size, args.output_file, logger)

    # ── STEP 3: Compare ───────────────────────────────────────────────────────
    logger.info("=" * 52)
    logger.info("STEP 3 — Comparison")
    logger.info("=" * 52)
    advantage = compare_and_report(
        model_coverage, random_coverage, args.step_size, args.output_file, logger)

    if advantage > 0:
        logger.info("✅ Trained model outperforms random by {:.2f}% at final step.".format(advantage))
    else:
        logger.warning("⚠️  Random policy matches or beats trained model by {:.2f}%.".format(abs(advantage)))