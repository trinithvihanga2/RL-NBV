import argparse
import logging
import os
import shutil
import sys
import random
import yaml


# ============================================================================
# CONFIG
# ============================================================================
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_to_args(config):
    ds = config.get("dataset", {})
    dss = config.get("dataset_splitting", {})

    return {
        "dataset_path": ds.get("dataset_path", "./dataset"),
        "train_data_path": ds.get("train_data_path", "./data/train"),
        "verify_data_path": ds.get("verify_data_path", "./data/verify"),
        "test_data_path": ds.get("test_data_path", "./data/test"),
        "train_ratio": dss.get("train_ratio", 0.7),
        "verify_ratio": dss.get("verify_ratio", 0.15),
        "test_ratio": dss.get("test_ratio", 0.15),
        "seed": dss.get("seed", 42),
        "log_path": dss.get("log_path", "split_dataset.log"),
    }


# ============================================================================
# LOGGER SETUP
# ============================================================================
def setup_logger(log_path="split_dataset.log"):
    logger = logging.getLogger("split_dataset")
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handle = logging.FileHandler(log_path)
    file_handle.setFormatter(log_format)
    file_handle.setLevel(logging.DEBUG)
    logger.addHandler(file_handle)

    console_handle = logging.StreamHandler()
    console_handle.setFormatter(log_format)
    console_handle.setLevel(logging.INFO)
    logger.addHandler(console_handle)

    return logger


# ============================================================================
# DATASET SPLITTING
# ============================================================================
def get_subdirectories(path):
    """Get list of subdirectory names in a path."""
    if not os.path.exists(path):
        return []
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def copy_directory(src, dst, logger):
    """Copy directory from src to dst."""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    logger.info("Copied {} -> {}".format(src, dst))


def split_dataset(args, logger):
    """
    Split dataset into train, verify, and test sets.
    """
    random.seed(args.seed)

    # Validate source path
    if not os.path.exists(args.dataset_path):
        logger.error("Source dataset path does not exist: {}".format(args.dataset_path))
        sys.exit(1)

    # Get all subdirectories
    subdirs = get_subdirectories(args.dataset_path)
    if not subdirs:
        logger.error("No subdirectories found in {}".format(args.dataset_path))
        sys.exit(1)

    logger.info("Found {} dataset directories".format(len(subdirs)))

    # Shuffle and split
    random.shuffle(subdirs)
    train_count = int(len(subdirs) * args.train_ratio)
    verify_count = int(len(subdirs) * args.verify_ratio)

    train_dirs = subdirs[:train_count]
    verify_dirs = subdirs[train_count : train_count + verify_count]
    test_dirs = subdirs[train_count + verify_count :]

    logger.info(
        "Split: {} train, {} verify, {} test".format(
            len(train_dirs), len(verify_dirs), len(test_dirs)
        )
    )

    # Create output directories
    for path in [args.train_data_path, args.verify_data_path, args.test_data_path]:
        os.makedirs(path, exist_ok=True)

    # Copy directories
    logger.info("Copying train directories...")
    for dir_name in train_dirs:
        src = os.path.join(args.dataset_path, dir_name)
        dst = os.path.join(args.train_data_path, dir_name)
        copy_directory(src, dst, logger)

    logger.info("Copying verify directories...")
    for dir_name in verify_dirs:
        src = os.path.join(args.dataset_path, dir_name)
        dst = os.path.join(args.verify_data_path, dir_name)
        copy_directory(src, dst, logger)

    logger.info("Copying test directories...")
    for dir_name in test_dirs:
        src = os.path.join(args.dataset_path, dir_name)
        dst = os.path.join(args.test_data_path, dir_name)
        copy_directory(src, dst, logger)


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
        help="Load config and print resolved values without splitting",
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

    # Validate split ratios
    total_ratio = args.train_ratio + args.verify_ratio + args.test_ratio
    if not (0.99 < total_ratio < 1.01):  # Allow for floating point errors
        raise ValueError("Split ratios must sum to 1.0, got {}".format(total_ratio))

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = setup_logger(args.log_path)
    logger.info("=" * 60)
    logger.info("DATASET SPLITTING START")
    logger.info("=" * 60)
    logger.info("Config: {}".format(config_path))
    for arg, value in sorted(vars(args).items()):
        logger.info("  {:30s}: {}".format(arg, value))
    logger.info("=" * 60)

    # ── Splitting ─────────────────────────────────────────────────────────────
    split_dataset(args, logger)

    # ── Done ──────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DATASET SPLITTING DONE ✅")
    logger.info("=" * 60)
