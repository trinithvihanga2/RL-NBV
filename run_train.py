import yaml
import argparse
import os
import sys
import subprocess


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def config_to_args(config):
    env   = config.get('environment', {})
    dqn   = config.get('dqn', {})
    pre   = config.get('pretrained', {})
    rb    = config.get('replay_buffer', {})
    out   = config.get('output', {})
    train = config.get('training', {})
    log   = config.get('logging', {})

    return {
        # Environment
        'data_path'              : env.get('data_path'),
        'verify_data_path'       : env.get('verify_data_path'),
        'test_data_path'         : env.get('test_data_path'),
        'view_num'               : env.get('view_num', 33),
        'observation_space_dim'  : env.get('observation_space_dim', 1024),
        'step_size'              : env.get('step_size', 10),
        'is_vec_env'             : env.get('is_vec_env', 0),
        'env_num'                : env.get('env_num', 8),
        'is_ratio_reward'        : env.get('is_ratio_reward', 1),

        # DQN - ✅ NOW PASSED FROM CONFIG
        'device'                 : dqn.get('device', 'cuda:0'),
        'learning_rate'          : dqn.get('learning_rate', 0.001),
        'batch_size'             : dqn.get('batch_size', 128),
        'buffer_size'            : dqn.get('buffer_size', 100000),
        'learning_starts'        : dqn.get('learning_starts', 3000),
        'exploration_fraction'   : dqn.get('exploration_fraction', 0.5),
        'exploration_final_eps'  : dqn.get('exploration_final_eps', 0.2),
        'gradient_steps'         : dqn.get('gradient_steps', 1),
        'train_freq'             : dqn.get('train_freq', 16),
        'gamma'                  : dqn.get('gamma', 0.1),
        'total_steps'            : dqn.get('total_steps', 500000),

        # Pretrained
        'is_transform'           : pre.get('is_transform', 0),
        'pretrained_model_path'  : pre.get('pretrained_model_path', 'null'),
        'is_freeze_fe'           : pre.get('is_freeze_fe', 0),

        # Replay Buffer
        'is_load_replay_buffer'  : rb.get('is_load_replay_buffer', 0),
        'replay_buffer_path'     : rb.get('replay_buffer_path', 'null'),
        'is_save_replay_buffer'  : rb.get('is_save_replay_buffer', 0),

        # Output
        'output_file'            : out.get('output_file', 'train_result.txt'),
        'checkpoint_path'        : out.get('checkpoint_path', 'checkpoints/rl_nbv'),
        'save_freq'              : out.get('save_freq', 10000),
        'eval_freq'              : out.get('eval_freq', 10000),
        'is_save_model'          : out.get('is_save_model', 1),

        # Training
        'is_profile'             : train.get('is_profile', 0),
        'resume'                 : train.get('resume', 0),

        # Logging
        'log_file'               : log.get('log_file', 'train_detail.log'),
    }


def print_summary(args_dict):
    print("\n" + "="*60)
    print("CONFIG SUMMARY")
    print("="*60)
    sections = {
        "Environment"  : ["data_path", "view_num", "observation_space_dim",
                          "step_size", "is_vec_env", "is_ratio_reward"],
        "DQN"          : ["device", "learning_rate", "batch_size", "buffer_size",
                          "learning_starts", "exploration_fraction",
                          "exploration_final_eps", "gradient_steps",
                          "train_freq", "gamma", "total_steps"],
        "Pretrained"   : ["is_transform", "pretrained_model_path", "is_freeze_fe"],
        "Replay Buffer": ["is_load_replay_buffer", "replay_buffer_path",
                          "is_save_replay_buffer"],
        "Output"       : ["output_file", "checkpoint_path", "save_freq", "eval_freq", "is_save_model"],
        "Training"     : ["is_profile", "resume"],
        "Logging"      : ["log_file"],
    }
    for section, keys in sections.items():
        print("\n[{}]".format(section))
        for key in keys:
            if key in args_dict:
                print("  {:30s}: {}".format(key, args_dict[key]))
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  type=str, default='config.yaml')
    parser.add_argument('--dry_run', action='store_true')
    cli = parser.parse_args()

    if not os.path.exists(cli.config):
        print("❌ Config not found: {}".format(cli.config))
        sys.exit(1)

    config   = load_config(cli.config)
    args     = config_to_args(config)
    print_summary(args)

    cmd = [sys.executable, "train.py"]
    for key, val in args.items():
        cmd += ["--{}".format(key), str(val) if val is not None else 'null']

    print("Command:\n  " + " \\\n    ".join(cmd) + "\n")

    if cli.dry_run:
        print("✅ Dry run — not executed")
        sys.exit(0)

    sys.exit(subprocess.run(cmd).returncode)