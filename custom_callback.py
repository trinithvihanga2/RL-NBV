import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

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
    ):
        super(NextBestViewCustomCallback, self).__init__(verbose)
        self.output_file = output_file
        self.verify_env = verify_env
        self.test_env = test_env
        self.step_size = getattr(test_env, 'max_step', 30)
        self.check_freq = check_freq
        self.cnt = 0
        self.best_coverage = -np.inf
        self.best_model_path = best_model_path
        self.save_freq = save_freq
        self.save_path = save_path or "."

    def _init_callback(self) -> None:
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
        if self.output_file:
            try:
                os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)
            except Exception:
                pass

    def _on_rollout_end(self) -> None:
        pass

    def _on_step(self) -> bool:
        eval_due_to_checkpoint = False

        # 1. Periodic Checkpoint
        if self.save_freq and self.n_calls % self.save_freq == 0:
            path = os.path.join(
                self.save_path, f"rl_nbv_periodic_{self.num_timesteps}_steps"
            )
            self.model.save(path)
            eval_due_to_checkpoint = True
            if self.verbose >= 1:
                print(
                    f"[Checkpoint] Periodic save at step {self.num_timesteps} to {path}"
                )

        # 2. Evaluation & Best Model Checkpoint
        eval_due_to_check_freq = bool(
            self.check_freq and self.n_calls % self.check_freq == 0
        )

        if eval_due_to_checkpoint or eval_due_to_check_freq:
            with open(self.output_file, "a+", encoding="utf-8") as f:
                f.write("------ {} ------\n".format(self.cnt))
            self.cnt += 1
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

    def _caculate_average_coverage(self):
        model_size = self.test_env.shapenet_reader.model_num
        average_coverage = np.zeros(self.step_size)
        
        # In a generic test env, it may be wrapped. Use its underlying attribute if present, or assume a default.
        if hasattr(self.test_env, 'terminated_coverage'):
            target_coverage = float(self.test_env.terminated_coverage)
        else:
            target_coverage = 0.97
            
        reached_view_counts = []
        for model_id in range(model_size):
            obs, _ = self.test_env.reset()
            coverages = np.zeros(self.step_size)
            
            # test_env.current_coverage might not be directly accessible if it's a VecEnv
            if hasattr(self.test_env, 'current_coverage'):
                coverages[0] = self.test_env.current_coverage
            else:
                coverages[0] = 0.0 # Will be updated in step
                
            average_coverage[0] += coverages[0]
            for step_id in range(self.step_size - 1):
                action, _states = self.model.predict(obs, deterministic=True)
                obs, rewards, terminated, truncated, info = self.test_env.step(action)
                
                # Check if it's a VecEnv info (list of dicts) or a single env info (dict)
                if isinstance(info, list) or isinstance(info, tuple):
                    current_cov = info[0].get("current_coverage", 0.0)
                else:
                    current_cov = info.get("current_coverage", 0.0)
                    
                coverages[step_id + 1] = current_cov
                average_coverage[step_id + 1] += coverages[step_id + 1]

            reached_indices = np.where(coverages >= target_coverage)[0]
            if reached_indices.size > 0:
                reached_view_counts.append(int(reached_indices[0] + 1))

        average_coverage = average_coverage / model_size
        average_coverage = average_coverage * 100

        avg_optimal_views = None
        if reached_view_counts:
            avg_optimal_views = float(np.mean(reached_view_counts))

        with open(self.output_file, "a+", encoding="utf-8") as f:
            f.write("average_coverage: ")
            for i in range(self.step_size):
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

        if self.verbose >= 1:
            print(
                "[Eval] average_coverage: "
                + " ".join(
                    "[{}]:{:.2f}".format(i + 1, average_coverage[i])
                    for i in range(self.step_size)
                )
            )
            if avg_optimal_views is not None:
                print(
                    "[Eval] optimal_view_count(target={:.2f}%): {:.2f} (reached_models={}/{})".format(
                        target_coverage * 100,
                        avg_optimal_views,
                        len(reached_view_counts),
                        model_size,
                    )
                )
            else:
                print(
                    "[Eval] optimal_view_count(target={:.2f}%): not_reached (reached_models=0/{})".format(
                        target_coverage * 100,
                        model_size,
                    )
                )

        return average_coverage[self.step_size - 1]
