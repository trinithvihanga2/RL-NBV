import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit


class NextBestViewCustomCallback(BaseCallback):
    def __init__(
        self,
        output_file,
        verify_env,
        test_env,
        check_freq=10000,
        step_size=10,
        best_model_path=None,
        save_freq=None,
        save_path=None,
        verbose: int = 1,
        check_replay_buffer: bool = False,
    ):
        super(NextBestViewCustomCallback, self).__init__(verbose)
        self.output_file = output_file
        self.verify_env = verify_env
        self.test_env = test_env
        self.step_size = step_size
        self.check_freq = check_freq
        self.cnt = 0
        self.best_coverage = -np.inf
        self.check_replay_buffer = check_replay_buffer
        self.best_model_path = best_model_path
        self.save_freq = save_freq
        self.save_path = save_path

    # check the repaly buffer
    def _init_callback(self) -> None:
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
        if not self.check_replay_buffer:
            return
        experience = self.model.replay_buffer.sample(
            32, env=self.model._vec_normalize_env
        )

    def _on_rollout_end(self) -> None:
        if not self.check_replay_buffer:
            return
        experience = self.model.replay_buffer.sample(
            32, env=self.model._vec_normalize_env
        )

    def _on_step(self) -> bool:
        # 1. Periodic Checkpoint
        if self.save_freq and self.n_calls % self.save_freq == 0:
            path = os.path.join(
                self.save_path, f"rl_nbv_periodic_{self.num_timesteps}_steps"
            )
            self.model.save(path)
            if self.verbose >= 1:
                print(
                    f"[Checkpoint] Periodic save at step {self.num_timesteps} to {path}"
                )

        # 2. Evaluation & Best Model Checkpoint
        if self.n_calls % self.check_freq == 0:
            with open(self.output_file, "a+", encoding="utf-8") as f:
                f.write("------ {} ------\n".format(self.cnt))
            self.cnt += 1
            # self._caclulate_policy_detail()
            cur_coverage = self._caculate_average_coverage()
            if cur_coverage > self.best_coverage:
                self.best_coverage = cur_coverage
                if self.best_model_path:
                    if self.verbose >= 1:
                        print(
                            f"[Best Model] New best coverage: {self.best_coverage:.22f}%! Saving to {self.best_model_path}"
                        )
                    self.model.save(self.best_model_path)
        return True

    def _caclulate_policy_detail(self):
        model_size = self.verify_env.shapenet_reader.model_num
        init_step = 0
        for model_id in range(model_size):
            obs = self.verify_env.reset(init_step=init_step)
            init_step = (init_step + 1) % 33
            with open(self.output_file, "a+", encoding="utf-8") as f:
                f.write(
                    "{}: ({}) [0]{:.2f} ".format(
                        self.verify_env.model_name,
                        self.verify_env.current_view,
                        self.verify_env.current_coverage * 100,
                    )
                )
            for step_id in range(self.step_size - 1):
                action, _states = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, info = self.verify_env.step(action)
                with open(self.output_file, "a+", encoding="utf-8") as f:
                    f.write(
                        "({}) [{}]{:.2f} ".format(
                            action, step_id + 1, info["current_coverage"] * 100
                        )
                    )
            with open(self.output_file, "a+", encoding="utf-8") as f:
                f.write("\n")

    def _caculate_average_coverage(self):
        model_size = self.test_env.shapenet_reader.model_num
        init_step = 0
        average_coverage = np.zeros(10)
        for model_id in range(model_size):
            obs = self.test_env.reset(init_step=init_step)
            init_step = (init_step + 1) % 33
            average_coverage[0] += self.test_env.current_coverage
            for step_id in range(self.step_size - 1):
                action, _states = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, info = self.test_env.step(action)
                average_coverage[step_id + 1] += info["current_coverage"]
        average_coverage = average_coverage / model_size
        average_coverage = average_coverage * 100
        with open(self.output_file, "a+", encoding="utf-8") as f:
            f.write("average_coverage: ")
            for i in range(self.step_size):
                f.write("[{}]:{:.2f} ".format(i + 1, average_coverage[i]))
            f.write("\n")
        return average_coverage[9]
