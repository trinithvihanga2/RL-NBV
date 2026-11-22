import numpy as np
import gymnasium as gym
from gymnasium import spaces
import envs.shapenet_reader as shapenet_reader
import torch
import sys
import os
from collections.abc import Mapping

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from distance.chamfer_distance import ChamferDistanceFunction
from envs.state_transition import (
    TargetOrbitConfig,
    calculate_sun_position,
)
import logging
import open3d as o3d
from envs.utils import resample_pcd, random_position_on_sphere, estimate_surface_normals
from envs.rendering import EnvironmentRenderer
from envs.state_transition.reward import calculate_continuous_reward
from envs.state_transition.coverage import update_continuous_coverage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from PIL import Image
from envs.state_transition.visibility import filter_lit_points
from envs.state_transition.travel_time import advance_time

# Maximum size for accumulated point cloud in continuous mode (prevents unbounded growth)
MAX_CLOUD_SIZE = 8192



class PointCloudNextBestViewEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        data_path,
        render_mode=None,
        observation_space_dim=-1,
        terminated_coverage=0.97,
        max_step=11,
        env_id=None,
        logger=logging.getLogger(__name__),
        is_ratio_reward=False,
        is_reward_with_cur_coverage=False,
        cur_coverage_ratio=1.0,
        time_cost_weight=1.0,
        fuel_budget=100.0,
        delta_v_weight=1.0,
        collision_penalty_weight=25.0,
        collision_check_samples=32,
        sun_position_config=None,
        target_orbit_config=None,
        state_reward_config=None,
        scp_planner_config=None,
        domain_randomization_config=None,
        reward_scale=0.01,
    ):
        """
        Initialize Point Cloud Next Best View Environment.

        Args:
            time_cost_weight: Weight of travel time penalty in reward calculation.
                            reward = coverage_gain - time_cost_weight * travel_time
                            Default 1.0: equal weight to coverage and time cost
                            Higher value: penalize time more heavily
                            Lower value: focus more on coverage
        """
        self.render_mode = render_mode
        self.COVERAGE_THRESHOLD = 0.00005
        self.is_ratio_reward = is_ratio_reward
        self.is_reward_with_cur_coverage = is_reward_with_cur_coverage
        self.cur_coverage_ratio = cur_coverage_ratio
        self.time_cost_weight = time_cost_weight
        self.delta_v_weight = delta_v_weight
        self.reward_scale = float(reward_scale)
        self.domain_randomization_config = (
            dict(domain_randomization_config)
            if isinstance(domain_randomization_config, Mapping)
            else {}
        )
        self.collision_penalty_weight = collision_penalty_weight
        self.collision_check_samples = max(int(collision_check_samples), 2)
        self.fuel_budget = float(fuel_budget)
        self._base_fuel_budget = float(fuel_budget)
        self.cumulative_dv = 0.0
        self.sun_position_config = sun_position_config or {}
        self.terminated_coverage = terminated_coverage

        from envs.state_transition.cw_utils import CWDynamics

        self.DEVICE = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.logger = logger
        self.logger.info("PointCloudNextBestViewEnv is ok")
        real_data_path = data_path
        if env_id is not None:
            real_data_path = os.path.join(data_path, str(env_id))
        self.data_path = real_data_path
        self.shapenet_reader = shapenet_reader.ShapenetReader(
            real_data_path, self.logger
        )

        self.max_step = int(max_step)
        self._base_max_step = int(max_step)
        self.action_history = []

        # Initialize orbital configuration for travel time calculations
        if target_orbit_config is None:
            target_orbit_config = {}
        if not isinstance(target_orbit_config, Mapping):
            raise TypeError("target_orbit_config must be a mapping")

        self.orbit_config = TargetOrbitConfig(
            orbit_radius=float(target_orbit_config.get("orbit_radius", 1.0)),
            grav_param=float(target_orbit_config.get("grav_param", 1.0)),
            num_orbits=float(target_orbit_config.get("num_orbits", 2.0)),
            min_transfer_time=float(target_orbit_config.get("min_transfer_time", 0.2)),
            unit_scale=float(target_orbit_config.get("unit_scale", 1.0)),
            max_transfer_time=float(target_orbit_config.get("max_transfer_time", 2.0)),
        )
        self._base_num_orbits = float(self.orbit_config.num_orbits)

        self.current_position = random_position_on_sphere(self.orbit_config.orbit_radius)
        self.current_points_cloud = np.zeros((0, 3), dtype=np.float32)
        self.ground_truth_points_cloud = self.shapenet_reader.ground_truth
        self.ground_truth_points_cloud_size = self.ground_truth_points_cloud.shape[0]
        self.ground_truth_tensor = self.shapenet_reader.ground_truth[
            np.newaxis, :, :
        ].astype(np.float32)
        self.ground_truth_tensor = torch.tensor(self.ground_truth_tensor).to(
            self.DEVICE
        )
        self.observation_space_dim = observation_space_dim
        self.current_time = 0.0

        if observation_space_dim == -1:
            shape = (512, 3)
        else:
            shape = (3, observation_space_dim)

        self.observation_space = spaces.Dict(
            {
                "current_point_cloud": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=shape,
                    dtype=np.float64,
                ),
                "camera_position": spaces.Box(
                    low=-1.5, high=1.5, shape=(3,), dtype=np.float32
                ),
                "sun_direction": spaces.Box(
                    low=-1.0, high=1.0, shape=(3,), dtype=np.float32
                ),
                "coverage": spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "fuel_remaining": spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "time_remaining": spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "steps_remaining": spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "visited_directions": spaces.Box(
                    low=-1.0, high=1.0, shape=(6,), dtype=np.float32
                ),
            }
        )
        self.current_coverage = 0.0
        self.coverage_add = 0.0
        self.step_cnt = 1
        self.model_name = self.shapenet_reader.get_model_info()
        if state_reward_config is None:
            state_reward_config = {}
        if not isinstance(state_reward_config, Mapping):
            raise TypeError("state_reward_config must be a mapping")
        self.reward_config = {
            "revisit_penalty": -15.0,
            "coverage_coeff": 1.0,
            "good_coverage_threshold": 0.0,
            "near_distance_threshold": 0.28,
            "polar_diversity_bonus": 8.0,
        }
        self.reward_config.update(dict(state_reward_config))


        self.renderer = EnvironmentRenderer(self.data_path, self.shapenet_reader, self.orbit_config.orbit_radius, self.collision_check_samples, self.collision_penalty_weight, self.logger)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        self.logger.info(
            "TargetOrbitConfig from env config: orbit_radius=%.4f, grav_param=%.4f, num_orbits=%.4f",
            self.orbit_config.orbit_radius,
            self.orbit_config.grav_param,
            self.orbit_config.num_orbits,
        )

        self.scp_planner_config = scp_planner_config or {}
        self.cw = CWDynamics(
            self.orbit_config.mean_motion,
            scp_config=self.scp_planner_config,
            unit_scale=self.orbit_config.unit_scale,
        )

        # Set max normalization bounds based on single-step transfer & orbital limits.
        self.max_travel_time = float(self.orbit_config.max_transfer_time)
        # Upper-bound single-step rest-to-rest CW transfer delta-v on the inspection sphere:
        v_orbit = self.orbit_config.mean_motion * self.orbit_config.orbit_radius
        v_rest_to_rest = (
            2.0
            * np.pi
            * self.orbit_config.orbit_radius
            / max(float(self.orbit_config.min_transfer_time), 0.5)
        )
        self.max_delta_v = float(
            max(2.0 * v_orbit, v_rest_to_rest) * self.orbit_config.unit_scale
        )

        # Current mission time (starts at 0.0, increments as agent moves)
        self.current_time = 0.0
        
        # Track visited positions to penalize revisiting them
        self.good_positions = []

        # ============================================================================
        # SUN POSITION INITIALIZATION
        # ============================================================================
        default_sun_position = np.array([1.0, 0.0, 0.0], dtype=float)
        initial_sun_position = self.sun_position_config.get(
            "initial_direction", default_sun_position
        )
        self.initial_sun_position = np.asarray(initial_sun_position, dtype=float)
        if self.initial_sun_position.shape != (3,):
            raise ValueError(
                "sun_position_config['initial_direction'] must have shape (3,), got {}".format(
                    self.initial_sun_position.shape
                )
            )

        self.sun_orbital_params = dict(self.sun_position_config.get("orbital_params", {}))
        if not self.sun_orbital_params:
            self.sun_orbital_params = {"angular_velocity_rad_per_s": 0.0}
        self._base_sun_orbital_params = dict(self.sun_orbital_params)

        self.current_sun_position = self.initial_sun_position.astype(float)
        self.logger.debug(
            "[SUN] Initialized. current_sun_position={} orbital_params={}".format(
                self.current_sun_position.tolist(), self.sun_orbital_params
            )
        )

        # State-transition runtime structures.
        self._transition_state = None
        self._canonical_points = np.zeros((0, 3), dtype=np.float32)
        self._model_transition_cache = {}
        self._mesh_cache = {}
        self._initialize_state_transition_for_current_model()
        self._initialize_initial_view()
        self.current_points_cloud_from_gt = np.zeros((0, 3), dtype=np.float32)

    def _apply_domain_randomization(self):
        dr = self.domain_randomization_config
        if not bool(dr.get("enabled", False)):
            self.fuel_budget = self._base_fuel_budget
            self.max_step = self._base_max_step
            self.orbit_config.num_orbits = self._base_num_orbits
            self.orbit_config.total_time = (
                self.orbit_config.num_orbits * self.orbit_config.orbital_period
            )
            self.orbit_config.angular_velocity = (
                2.0 * np.pi / self.orbit_config.total_time
            )
            self.sun_orbital_params = dict(self._base_sun_orbital_params)
            return

        # 1. Fuel budget randomization
        nominal_fuel_prob = float(dr.get("nominal_fuel_prob", 0.2))
        if np.random.random() < nominal_fuel_prob:
            self.fuel_budget = self._base_fuel_budget
        else:
            fuel_range = dr.get("fuel_budget_range_tu", [153.008, 1530.08])
            lo = max(float(fuel_range[0]), 1e-6)
            hi = max(float(fuel_range[1]), lo)
            self.fuel_budget = float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))

        # 2. Step horizon & orbital period randomization
        horizon_choices = dr.get(
            "horizon_choices",
            [[10, 2.0], [20, 2.0], [30, 2.0], [40, 3.5], [50, 5.0]],
        )
        if horizon_choices:
            raw_probs = dr.get("horizon_probs", None)
            if raw_probs is not None and len(raw_probs) == len(horizon_choices):
                probs = np.asarray(raw_probs, dtype=np.float64)
                prob_sum = float(np.sum(probs))
                if prob_sum > 0.0:
                    probs = probs / prob_sum
                else:
                    probs = np.ones(len(horizon_choices), dtype=np.float64) / len(horizon_choices)
            elif raw_probs is None and len(horizon_choices) == 5:
                probs = np.asarray([0.15, 0.15, 0.40, 0.15, 0.15], dtype=np.float64)
                probs = probs / np.sum(probs)
            else:
                probs = np.ones(len(horizon_choices), dtype=np.float64) / len(horizon_choices)

            choice_idx = int(np.random.choice(len(horizon_choices), p=probs))
            chosen_max_step, chosen_num_orbits = horizon_choices[choice_idx]
            self.max_step = int(chosen_max_step)
            self.orbit_config.num_orbits = float(chosen_num_orbits)
            self.orbit_config.total_time = (
                self.orbit_config.num_orbits * self.orbit_config.orbital_period
            )
            self.orbit_config.angular_velocity = (
                2.0 * np.pi / self.orbit_config.total_time
            )

        # 3. Sun inclination, ascending node, and initial phase randomization
        sun_params = dict(self._base_sun_orbital_params)
        sun_zero_incl_prob = float(dr.get("sun_zero_inclination_prob", 0.3))
        if np.random.random() < sun_zero_incl_prob:
            sun_params["inclination_rad"] = 0.0
            sun_params["ascending_node_rad"] = 0.0
        else:
            max_inc_deg = float(dr.get("sun_inclination_max_deg", 35.0))
            max_inc_rad = float(np.deg2rad(abs(max_inc_deg)))
            sun_params["inclination_rad"] = float(
                np.random.uniform(-max_inc_rad, max_inc_rad)
            )
            sun_params["ascending_node_rad"] = float(
                np.random.uniform(0.0, 2.0 * np.pi)
            )
        if bool(dr.get("randomize_sun_phase", True)):
            sun_params["initial_phase_rad"] = float(
                np.random.uniform(0.0, 2.0 * np.pi)
            )
        self.sun_orbital_params = sun_params

        self.logger.debug(
            "[DR] fuel_budget=%.2f max_step=%d num_orbits=%.2f sun_params=%s",
            self.fuel_budget,
            self.max_step,
            self.orbit_config.num_orbits,
            self.sun_orbital_params,
        )

    def _accumulate_point_cloud(self, new_view_points: np.ndarray, voxel_size: float = 0.025):
        if new_view_points is None or new_view_points.shape[0] == 0:
            return
        if self.current_points_cloud.shape[0] > 0:
            combined = np.vstack([self.current_points_cloud, new_view_points])
        else:
            combined = np.asarray(new_view_points, dtype=np.float32)

        opcd = o3d.geometry.PointCloud()
        opcd.points = o3d.utility.Vector3dVector(combined.astype(np.float64))
        voxel_pcd = opcd.voxel_down_sample(voxel_size=voxel_size)
        downsampled = np.asarray(voxel_pcd.points, dtype=np.float32)

        if downsampled.shape[0] > MAX_CLOUD_SIZE:
            idx = np.random.choice(downsampled.shape[0], MAX_CLOUD_SIZE, replace=False)
            downsampled = downsampled[idx]
        self.current_points_cloud = downsampled

    def _initialize_initial_view(self):
        self.current_points_cloud = np.zeros((0, 3), dtype=np.float32)
        initial_points = self.renderer.get_points_from_position(
            self.current_position, self._canonical_points
        )
        if initial_points.shape[0] > 0:
            if (
                hasattr(self.renderer, "last_rendered_normals")
                and self.renderer.last_rendered_normals.shape == initial_points.shape
            ):
                normals = self.renderer.last_rendered_normals
            else:
                normals = estimate_surface_normals(
                    initial_points, camera_position=self.current_position
                )
            initial_points = filter_lit_points(
                initial_points, normals, self.current_sun_position
            )
        self._accumulate_point_cloud(initial_points)
        self._coverage_map, self.current_coverage, _ = update_continuous_coverage(
            initial_points,
            self._canonical_tensor,
            self._coverage_map,
            self.current_coverage,
            self.ground_truth_points_cloud_size,
            self.COVERAGE_THRESHOLD,
            ChamferDistanceFunction,
            self.DEVICE,
        )
        self.good_positions = [self.current_position.copy()]

    def _initialize_state_transition_for_current_model(self):
        model_name = self.shapenet_reader.get_model_info()
        cached_geometry = self._model_transition_cache.get(model_name)

        if cached_geometry is None:
            self._canonical_points = np.asarray(
                self.shapenet_reader.ground_truth, dtype=np.float32
            )
            canonical_tensor = torch.tensor(
                self._canonical_points[np.newaxis, :, :].astype(np.float32)
            ).to(self.DEVICE)
            
            cached_geometry = {
                "canonical_points": self._canonical_points,
                "canonical_tensor": canonical_tensor,
            }
            self._model_transition_cache[model_name] = cached_geometry
        else:
            self._canonical_points = cached_geometry["canonical_points"]
            canonical_tensor = cached_geometry["canonical_tensor"]

        self.ground_truth_points_cloud_size = self._canonical_points.shape[0]
        self._canonical_tensor = canonical_tensor
        self._coverage_map = np.zeros(self.ground_truth_points_cloud_size, dtype=bool)
        self.current_coverage = 0.0
        self.coverage_add = 0.0
        self.ground_truth_points_cloud = self._canonical_points
        self.current_points_cloud_from_gt = np.zeros((0, 3), dtype=np.float32)

    def step(self, action):
        # action: [norm_theta, norm_phi, norm_transfer_time] all in [-1, 1]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 3:
            raise ValueError(
                "continuous action must have 3 elements, got {}".format(
                    action.shape[0]
                )
            )

        # Denormalize action from [-1, 1] to physical ranges
        norm_theta = float(np.clip(action[0], -1.0, 1.0))
        norm_phi = float(np.clip(action[1], -1.0, 1.0))
        norm_time = float(np.clip(action[2], -1.0, 1.0))
        
        theta = (norm_theta + 1.0) * (np.pi / 2.0)
        phi = (norm_phi + 1.0) * np.pi
        t_min = float(self.orbit_config.min_transfer_time)
        t_max = float(self.orbit_config.max_transfer_time)
        requested_transfer_time = t_min + 0.5 * (norm_time + 1.0) * (t_max - t_min)

        # Convert spherical to Cartesian coordinates
        r = self.orbit_config.orbit_radius
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        new_position = np.array([x, y, z], dtype=np.float32)

        # 2. Resolve commanded travel time (bounded by remaining mission horizon)
        remaining_time = max(0.0, self.orbit_config.total_time - self.current_time)
        if remaining_time <= 0.0:
            travel_time = 0.0
        else:
            upper_bound = max(
                float(self.orbit_config.min_transfer_time), float(remaining_time)
            )
            travel_time = float(
                np.clip(
                    requested_transfer_time,
                    self.orbit_config.min_transfer_time,
                    upper_bound,
                )
            )

        # 3. Compute Δv via CW dynamics (using pre-initialized instance)
        # Apply the real-world SI unit scale (e.g. 6.25m) so Delta-V is in SI units
        r0 = self.current_position * self.orbit_config.unit_scale
        rf = new_position * self.orbit_config.unit_scale
        delta_v, _, _ = self.cw.compute_delta_v(r0, rf, travel_time)
        if delta_v == np.inf:
            delta_v = self.max_delta_v  # fallback for singular transfers

        traj_canonical = (
            self.cw.last_trajectory / self.orbit_config.unit_scale
            if (hasattr(self.cw, "last_trajectory") and self.cw.last_trajectory is not None)
            else None
        )
        collision_detected, collision_penalty, collision_min_clearance = (
            self.renderer.check_transfer_collision(
                self.current_position, new_position, travel_time, trajectory=traj_canonical
            )
        )
        if collision_detected:
            reward = calculate_continuous_reward(
                cover_add=0.0,
                current_coverage=self.current_coverage,
                step_cnt=self.step_cnt,
                travel_time=travel_time,
                max_travel_time=self.max_travel_time,
                delta_v=delta_v,
                max_delta_v=self.max_delta_v,
                collision_penalty=collision_penalty,
                is_reward_with_cur_coverage=self.is_reward_with_cur_coverage,
                is_ratio_reward=self.is_ratio_reward,
                time_cost_weight=self.time_cost_weight,
                delta_v_weight=self.delta_v_weight,
            )
            reward = float(reward * self.reward_scale)
            self.logger.debug(
                f"[REWARD] coverage_gain=0.0, travel_time={travel_time:7.4f}, "
                f"delta_v={delta_v:7.4f}, collision_penalty={collision_penalty:7.4f}, final={reward:7.4f}"
            )
            terminated = True
            observation = self._get_observation_space()
            info = self._get_info(
                travel_time,
                delta_v,
                requested_transfer_time,
                collision_detected=True,
                collision_penalty=collision_penalty,
                collision_min_clearance=collision_min_clearance,
            )
            self.logger.warning(
                "[COLLISION] action=%s travel_time=%.6f delta_v=%.6f clearance=%.6f penalty=%.6f",
                np.array2string(action, precision=4),
                travel_time,
                delta_v,
                collision_min_clearance,
                collision_penalty,
            )
            truncated = False
            return observation, reward, terminated, truncated, info

        # Advance time and sun position to arrival
        self.current_time = advance_time(
            self.current_time,
            travel_time,
            self.orbit_config.total_time,
            False
        )
        self.current_sun_position = calculate_sun_position(
            new_time=self.current_time,
            prev_sun_position=self.initial_sun_position,
            orbital_params=self.sun_orbital_params,
        )

        # 4. Get visible points from new position
        new_view_points = self.renderer.get_points_from_position(new_position, self._canonical_points)
        if new_view_points.shape[0] > 0:
            if (
                hasattr(self.renderer, "last_rendered_normals")
                and self.renderer.last_rendered_normals.shape == new_view_points.shape
            ):
                normals = self.renderer.last_rendered_normals
            else:
                normals = estimate_surface_normals(
                    new_view_points, camera_position=new_position
                )
            new_view_points = filter_lit_points(new_view_points, normals, self.current_sun_position)

        # 5. Update coverage & accumulate voxel-downsampled point cloud
        self._accumulate_point_cloud(new_view_points)
        self._coverage_map, self.current_coverage, coverage_gain = update_continuous_coverage(
            new_view_points,
            self._canonical_tensor,
            self._coverage_map,
            self.current_coverage,
            self.ground_truth_points_cloud_size,
            self.COVERAGE_THRESHOLD,
            ChamferDistanceFunction,
            self.DEVICE,
        )

        # 5.5 Check for revisiting near previously visited positions
        revisit_penalty_val = 0.0
        min_dist_to_visited = float("inf")
        if len(self.good_positions) > 0:
            distances = np.linalg.norm(np.array(self.good_positions) - new_position, axis=1)
            min_dist_to_visited = float(np.min(distances))
            if min_dist_to_visited < self.reward_config.get("near_distance_threshold", 0.28):
                revisit_penalty_val = float(self.reward_config.get("revisit_penalty", -15.0))
                if revisit_penalty_val < 0.0 and coverage_gain < 0.005:
                    revisit_penalty_val *= 1.5
                self.logger.debug(f"[REVISIT] Near a visited position! Penalty: {revisit_penalty_val}")

        # Remember visited position to prevent stationary hovering (avoid exact duplicates)
        if (
            coverage_gain >= self.reward_config.get("good_coverage_threshold", 0.0)
            and min_dist_to_visited >= 1e-4
        ):
            self.good_positions.append(new_position.copy())

        # 6. Update state
        self.current_position = new_position
        self.cumulative_dv += delta_v
        
        # 6.5 Update dense trajectory for plotting
        if hasattr(self.cw, 'last_trajectory') and self.cw.last_trajectory is not None:
            # Append all points except the first one (which matches current_position)
            for pt in self.cw.last_trajectory[1:]:
                self.full_trajectory.append(pt.copy() / self.orbit_config.unit_scale)
        else:
            self.full_trajectory.append(self.current_position.copy())
        self.step_cnt += 1

        # 7. Compute reward using _get_reward for consistency
        reward = calculate_continuous_reward(
            cover_add=coverage_gain,
            current_coverage=self.current_coverage,
            step_cnt=self.step_cnt,
            travel_time=travel_time,
            max_travel_time=self.max_travel_time,
            delta_v=delta_v,
            max_delta_v=self.max_delta_v,
            collision_penalty=0.0,
            is_reward_with_cur_coverage=self.is_reward_with_cur_coverage,
            is_ratio_reward=self.is_ratio_reward,
            time_cost_weight=self.time_cost_weight,
            delta_v_weight=self.delta_v_weight,
        )

        # Strictly gated polar diversity bonus
        polar_coeff = float(self.reward_config.get("polar_diversity_bonus", 8.0))
        abs_z_frac = abs(float(new_position[2])) / max(float(r), 1e-8)
        if (
            polar_coeff > 0.0
            and revisit_penalty_val == 0.0
            and coverage_gain >= 0.002
            and abs_z_frac > 0.35
        ):
            polar_bonus_val = polar_coeff * abs_z_frac
        else:
            polar_bonus_val = 0.0

        # Apply revisit penalty and polar diversity bonus
        reward += revisit_penalty_val + polar_bonus_val

        # 8. Check termination using _get_terminated for consistency
        terminated = self._get_terminated()

        # 9. Add completion bonus if the episode successfully terminated via coverage
        if terminated and self.current_coverage >= self.terminated_coverage:
            reward += 100.0
            self.logger.debug("[REWARD] Added +100.0 completion bonus!")

        reward = float(reward * self.reward_scale)
        self.logger.debug(
            f"[REWARD] coverage_gain={coverage_gain:7.4f}, travel_time={travel_time:7.4f}, "
            f"delta_v={delta_v:7.4f}, revisit={revisit_penalty_val}, polar={polar_bonus_val:6.3f}, final={reward:7.4f}"
        )

        observation = self._get_observation_space()
        info = self._get_info(
            travel_time,
            delta_v,
            requested_transfer_time,
            collision_detected=False,
            collision_penalty=0.0,
            collision_min_clearance=None,
        )

        truncated = False
        return observation, reward, terminated, truncated, info

    # for greedy policy test
    def try_step(self, action):
        # ============================================================================
        # TRY_STEP: Simulate action without updating state (for planning/evaluation)
        # ============================================================================
        # This method tests the value of an action without committing to it.
        # Used for greedy policy evaluation and planning.
        raise NotImplementedError("try_step is not supported in continuous mode")

    def reset(self, *, seed=None, options=None):
        self.shapenet_reader.get_next_model()
        self.action_history.clear()
        self.good_positions.clear()
        self.step_cnt = 1
        self.model_name = self.shapenet_reader.get_model_info()

        # Apply domain randomization (or restore base parameters if disabled)
        self._apply_domain_randomization()

        # Reset mission time and fuel
        self.current_time = 0.0
        self.cumulative_dv = 0.0
        self.logger.debug(
            f"[reset] Mission time reset to 0.0. Horizon: {self.orbit_config.total_time:.6f} time units"
        )

        self.current_position = random_position_on_sphere(self.orbit_config.orbit_radius)
        self.full_trajectory = [self.current_position.copy()]

        # Re-initialize and synchronize sun direction with reset mission time.
        self.current_sun_position = calculate_sun_position(
            new_time=self.current_time,
            prev_sun_position=self.initial_sun_position,
            orbital_params=self.sun_orbital_params,
        )
        self.logger.debug(
            "[SUN] reset: t={:.6f} dir=[{:.6f}, {:.6f}, {:.6f}]".format(
                self.current_time,
                self.current_sun_position[0],
                self.current_sun_position[1],
                self.current_sun_position[2],
            )
        )

        self._initialize_state_transition_for_current_model()
        self._initialize_initial_view()

        observation = self._get_observation_space()
        info = self._get_info()
        self.logger.debug("[reset] pass")
        return observation, info

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        if self.render_mode != "rgb_array" and mode != "rgb_array":
            return None

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the full model as light transparent background
        if hasattr(self, 'shapenet_reader') and self.shapenet_reader is not None:
            model_pc = self.shapenet_reader.ground_truth
            if model_pc is not None and len(model_pc) > 0:
                ax.scatter(model_pc[:, 0], model_pc[:, 1], model_pc[:, 2], 
                           c='lightgray', s=1, alpha=0.1, label='Ground Truth')

        # Plot the accumulated points
        if hasattr(self, 'current_points_cloud') and self.current_points_cloud is not None:
            curr_pc = self.current_points_cloud
            if len(curr_pc) > 0:
                ax.scatter(curr_pc[:, 0], curr_pc[:, 1], curr_pc[:, 2], 
                           c='blue', s=2, alpha=0.8, label='Accumulated PC')

        # Plot the camera position
        if hasattr(self, 'current_position') and self.current_position is not None:
            cam = np.asarray(self.current_position)
            ax.scatter(cam[0], cam[1], cam[2], 
                       c='red', s=100, marker='*', label='Camera Position')
            ax.plot([0, cam[0]], [0, cam[1]], [0, cam[2]], c='red', linestyle='--', alpha=0.5)

        # Plot the sun direction
        if hasattr(self, 'current_sun_position') and self.current_sun_position is not None:
            sun = np.asarray(self.current_sun_position)
            ax.quiver(0, 0, 0, sun[0]*1.5, sun[1]*1.5, sun[2]*1.5, color='orange', arrow_length_ratio=0.1, label='Sun Direction', linewidth=2)

        time_val = getattr(self, 'current_time', 0.0)
        ax.set_title(f"Step: {self.step_cnt}/{self.max_step} | Cov: {self.current_coverage*100:.1f}% | Time: {time_val:.2f}")
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc='upper right')

        # Save to numpy array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        plt.close(fig)
        
        img = Image.open(buf)
        img_arr = np.array(img)
        return img_arr

    def _caculate_current_coverage(self):
        return self.current_coverage


    def _get_observation_space(self):
        """Get observation with normalized scalars."""
        source_pc = self.current_points_cloud

        if self.observation_space_dim == -1:
            # do not downsample, just for debug
            cur_pc = source_pc.T
        else:
            cur_pc = resample_pcd(
                source_pc,
                self.observation_space_dim,
                self.logger,
                self.model_name,
            )
            cur_pc = cur_pc.T

        if hasattr(self, "current_sun_position") and self.current_sun_position is not None:
            sun_norm = float(np.linalg.norm(self.current_sun_position))
            sun_dir = (self.current_sun_position / max(sun_norm, 1e-8)).astype(np.float32)
        else:
            sun_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        steps_left_frac = float(
            np.clip(
                float(self.max_step - (self.step_cnt - 1)) / max(float(self.max_step), 1.0),
                0.0,
                1.0,
            )
        )
        steps_remaining = np.array([steps_left_frac * 2.0 - 1.0], dtype=np.float32)

        if len(self.good_positions) > 0:
            pos_arr = np.asarray(self.good_positions, dtype=np.float32)
            norms = np.linalg.norm(pos_arr, axis=1, keepdims=True)
            unit_pos = pos_arr / np.maximum(norms, 1e-8)
            counts = np.array(
                [
                    np.sum(np.maximum(0.0, unit_pos[:, 0])),
                    np.sum(np.maximum(0.0, -unit_pos[:, 0])),
                    np.sum(np.maximum(0.0, unit_pos[:, 1])),
                    np.sum(np.maximum(0.0, -unit_pos[:, 1])),
                    np.sum(np.maximum(0.0, unit_pos[:, 2])),
                    np.sum(np.maximum(0.0, -unit_pos[:, 2])),
                ],
                dtype=np.float32,
            )
            visited_directions = np.clip(
                np.tanh(counts / 3.0) * 2.0 - 1.0, -1.0, 1.0
            ).astype(np.float32)
        else:
            visited_directions = np.full(6, -1.0, dtype=np.float32)

        return {
            "current_point_cloud": cur_pc.astype(np.float32),
            "camera_position": self.current_position.astype(np.float32),
            "sun_direction": sun_dir,
            "coverage": np.array([self.current_coverage * 2.0 - 1.0], dtype=np.float32),
            "fuel_remaining": np.array(
                [(max(0.0, self.fuel_budget - self.cumulative_dv) / self.fuel_budget) * 2.0 - 1.0],
                dtype=np.float32,
            ),
            "time_remaining": np.array(
                [
                    (max(0.0, self.orbit_config.total_time - self.current_time)
                    / self.orbit_config.total_time) * 2.0 - 1.0
                ],
                dtype=np.float32,
            ),
            "steps_remaining": steps_remaining,
            "visited_directions": visited_directions,
        }

    def _get_terminated(self):
        if self.step_cnt > self.max_step:
            return True
        if self.current_coverage >= self.terminated_coverage:
            return True
        if self.cumulative_dv > self.fuel_budget:
            return True
        if self.current_time >= self.orbit_config.total_time:
            return True
        return False

    def _get_info(
        self,
        travel_time=0.0,
        delta_v=0.0,
        requested_travel_time=None,
        collision_detected=False,
        collision_penalty=0.0,
        collision_min_clearance=None,
    ):
        return self._get_info_continuous(
            travel_time,
            delta_v,
            requested_travel_time,
            collision_detected,
            collision_penalty,
            collision_min_clearance,
        )

    def _get_info_continuous(
        self,
        travel_time,
        delta_v,
        requested_travel_time=None,
        collision_detected=False,
        collision_penalty=0.0,
        collision_min_clearance=None,
    ):
        """Get info dict for continuous mode."""
        if requested_travel_time is None:
            requested_travel_time = travel_time
        return {
            "cur_points_cloud": self._canonical_points,
            "model_name": self.model_name,
            "current_coverage": self.current_coverage,
            "camera_position": self.current_position.copy(),
            "travel_time": travel_time,
            "requested_travel_time": requested_travel_time,
            "delta_v": delta_v,
            "collision_detected": collision_detected,
            "collision_penalty": collision_penalty,
            "collision_min_clearance": collision_min_clearance,
            "mission_time": self.current_time,
            "cumulative_dv": self.cumulative_dv,
            "fuel_remaining": max(0.0, self.fuel_budget - self.cumulative_dv),
        }

    def _get_debug_info(self):
        self.logger.info(
            "model name:{}, action history: {}".format(
                self.model_name, self.action_history
            )
        )


PointCloudNBVEnvLevel2 = PointCloudNextBestViewEnv
