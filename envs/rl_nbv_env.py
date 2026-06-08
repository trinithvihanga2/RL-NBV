import numpy as np
import gym
from gym import spaces
import envs.shapenet_reader as shapenet_reader
import open3d as o3d
import torch
import sys
import os
from collections.abc import Mapping

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from distance.chamfer_distance import ChamferDistanceFunction
from envs.state_transition import (
    TargetOrbitConfig,
    calculate_sun_position,
    compute_all_travel_times,
    compute_delta_v_matrix,
)
import logging
from envs.utils import resample_pcd, normalize_pc, random_position_on_sphere, estimate_surface_normals
from envs.rendering import EnvironmentRenderer
from envs.state_transition.reward import calculate_continuous_reward
from envs.state_transition.coverage import update_continuous_coverage
from envs.state_transition.visibility import filter_lit_points
from envs.state_transition.travel_time import advance_time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Maximum size for accumulated point cloud in continuous mode (prevents unbounded growth)
MAX_CLOUD_SIZE = 8192



class PointCloudNextBestViewEnv(gym.Env):
    def __init__(
        self,
        data_path,
        viewpoints_path,
        view_num=33,
        begin_view=-1,
        observation_space_dim=-1,
        terminated_coverage=0.97,
        max_step=11,
        env_id=None,
        logger=logging.getLogger(__name__),
        is_normalize=True,
        is_ratio_reward=False,
        is_reward_with_cur_coverage=False,
        cur_coverage_ratio=1.0,
        time_cost_weight=1.0,
        fuel_budget=50.0,
        delta_v_weight=1.0,
        collision_penalty_weight=25.0,
        collision_check_samples=32,
        sun_position_config=None,
        target_orbit_config=None,
        state_reward_config=None,
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
        self.COVERAGE_THRESHOLD = 0.00005
        self.is_ratio_reward = is_ratio_reward
        self.is_reward_with_cur_coverage = is_reward_with_cur_coverage
        self.cur_coverage_ratio = cur_coverage_ratio
        self.time_cost_weight = time_cost_weight
        self.delta_v_weight = delta_v_weight
        self.collision_penalty_weight = collision_penalty_weight
        self.collision_check_samples = max(int(collision_check_samples), 2)
        self.fuel_budget = fuel_budget
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
            real_data_path, view_num, self.logger, True
        )
        self.view_state = np.zeros(view_num, dtype=np.int32)
        self.view_num = view_num
        self.begin_view = begin_view
        self.max_step = max_step
        self.current_view = 0
        self.action_history = []
        self.current_position = random_position_on_sphere()
        self.current_points_cloud = np.zeros((0, 3), dtype=np.float32)
        self.ground_truth_points_cloud = self.shapenet_reader.ground_truth
        self.ground_truth_points_cloud_size = self.ground_truth_points_cloud.shape[0]
        self.ground_truth_tensor = self.shapenet_reader.ground_truth[
            np.newaxis, :, :
        ].astype(np.float32)
        self.ground_truth_tensor = torch.tensor(self.ground_truth_tensor).to(
            self.DEVICE
        )
        self.view_state[self.current_view] = 1
        self.observation_space_dim = observation_space_dim
        self.is_normalize = is_normalize
        self.current_time = 0.0

        if observation_space_dim == -1:
            self.observation_space = spaces.Dict(
                {
                    "current_point_cloud": spaces.Box(
                        low=float("-inf"),
                        high=float("inf"),
                        shape=(512, 3),
                        dtype=np.float64,
                    ),
                    "camera_position": spaces.Box(
                        low=-1.0, high=1.0, shape=(3,), dtype=np.float32
                    ),
                    "coverage": spaces.Box(
                        low=0.0, high=1.0, shape=(1,), dtype=np.float32
                    ),
                    "fuel_remaining": spaces.Box(
                        low=0.0, high=1.0, shape=(1,), dtype=np.float32
                    ),
                    "time_remaining": spaces.Box(
                        low=0.0, high=1.0, shape=(1,), dtype=np.float32
                    ),
                }
            )
        elif self.is_normalize:
            self.observation_space = spaces.Dict(
                {
                    "current_point_cloud": spaces.Box(
                        low=float("-1"),
                        high=float("1"),
                        shape=(3, observation_space_dim),
                        dtype=np.float64,
                    ),
                    "camera_position": spaces.Box(
                        low=-1.0, high=1.0, shape=(3,), dtype=np.float32
                    ),
                    "coverage": spaces.Box(
                        low=0.0, high=1.0, shape=(1,), dtype=np.float32
                    ),
                    "fuel_remaining": spaces.Box(
                        low=0.0, high=1.0, shape=(1,), dtype=np.float32
                    ),
                    "time_remaining": spaces.Box(
                        low=0.0, high=1.0, shape=(1,), dtype=np.float32
                    ),
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "current_point_cloud": spaces.Box(
                        low=float("-inf"),
                        high=float("inf"),
                        shape=(3, observation_space_dim),
                        dtype=np.float64,
                    ),
                    "camera_position": spaces.Box(
                        low=-1.0, high=1.0, shape=(3,), dtype=np.float32
                    ),
                    "coverage": spaces.Box(
                        low=0.0, high=1.0, shape=(1,), dtype=np.float32
                    ),
                    "fuel_remaining": spaces.Box(
                        low=0.0, high=1.0, shape=(1,), dtype=np.float32
                    ),
                    "time_remaining": spaces.Box(
                        low=0.0, high=1.0, shape=(1,), dtype=np.float32
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
            "revisit_penalty": -5.0,
            "coverage_coeff": 1.0,
        }
        self.reward_config.update(dict(state_reward_config))

        # ============================================================================
        # TRAVEL TIME INITIALIZATION
        # ============================================================================
        # Load viewpoints from the required path
        resolved_viewpoints_path = os.path.expanduser(viewpoints_path)
        if not os.path.isabs(resolved_viewpoints_path):
            resolved_viewpoints_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", resolved_viewpoints_path)
            )

        if not os.path.exists(resolved_viewpoints_path):
            raise FileNotFoundError(
                "viewpoints_path does not exist: {}".format(resolved_viewpoints_path)
            )

        self.viewpoints = np.loadtxt(resolved_viewpoints_path)  # Shape: (N, 3)
        self.logger.info(
            f"Loaded {self.viewpoints.shape[0]} viewpoints from {resolved_viewpoints_path}"
        )

        # Initialize orbital configuration for travel time calculations
        # orbit_radius: 1.0 (unit sphere)
        # grav_param: 1.0 (dimensionless, controls orbital dynamics)
        # num_orbits: 2.0 (mission horizon = 2 complete orbits)
        if target_orbit_config is None:
            target_orbit_config = {}
        if not isinstance(target_orbit_config, Mapping):
            raise TypeError("target_orbit_config must be a mapping")

        self.orbit_config = TargetOrbitConfig(
            orbit_radius=float(target_orbit_config.get("orbit_radius", 1.0)),
            grav_param=float(target_orbit_config.get("grav_param", 1.0)),
            num_orbits=float(target_orbit_config.get("num_orbits", 2.0)),
        )
        self.renderer = EnvironmentRenderer(self.data_path, self.shapenet_reader, self.orbit_config.orbit_radius, self.collision_check_samples, self.collision_penalty_weight, self.logger)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array(
                [np.pi, 2 * np.pi, self.orbit_config.total_time], dtype=np.float32
            ),
            shape=(3,),
            dtype=np.float32,
        )
        self.logger.info(
            "TargetOrbitConfig from env config: orbit_radius=%.4f, grav_param=%.4f, num_orbits=%.4f",
            self.orbit_config.orbit_radius,
            self.orbit_config.grav_param,
            self.orbit_config.num_orbits,
        )

        self.cw = CWDynamics(self.orbit_config.mean_motion)

        # Precompute travel times between all pairs of viewpoints
        # Shape: (view_num, view_num) where [i,j] = travel time from view i to view j
        if self.viewpoints is not None:
            self.travel_times = compute_all_travel_times(
                self.viewpoints, self.orbit_config
            )
            self.max_travel_time = max(float(np.max(self.travel_times)), 1e-12)
            self.logger.info(
                f"Precomputed travel times matrix shape: {self.travel_times.shape}"
            )
        else:
            self.travel_times = None
            self.max_travel_time = 1.0
            self.logger.warning("Travel times not computed (viewpoints unavailable)")

        if self.viewpoints is not None and self.travel_times is not None:
            self.delta_v_matrix = compute_delta_v_matrix(
                viewpoints=self.viewpoints,
                travel_times=self.travel_times,
                orbit_radius=self.orbit_config.orbit_radius,
                mean_motion=self.orbit_config.mean_motion,
            )
            self.max_delta_v = max(float(np.max(self.delta_v_matrix)), 1e-12)
        else:
            raise ValueError(
                "Delta-V matrix cannot be computed without viewpoints and travel times"
            )

        # Current mission time (starts at 0.0, increments as agent moves)
        self.current_time = 0.0

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

        self.sun_orbital_params = self.sun_position_config.get("orbital_params", {})
        if not self.sun_orbital_params:
            self.sun_orbital_params = {"angular_velocity_rad_per_s": 0.0}

        self.current_sun_position = self.initial_sun_position.astype(float)
        self.logger.debug(
            "[SUN] Initialized. current_sun_position={} orbital_params={}".format(
                self.current_sun_position.tolist(), self.sun_orbital_params
            )
        )

        # State-transition runtime structures.
        self.geometry_cache = {}
        self._transition_state = None
        self._canonical_points = np.zeros((0, 3), dtype=np.float32)
        self._model_transition_cache = {}
        self._mesh_cache = {}
        self._initialize_state_transition_for_current_model(self.current_view)
        self.current_points_cloud = self.renderer.get_points_from_position(
            self.current_position, self._canonical_points
        )
        self.current_points_cloud_from_gt = np.zeros((0, 3), dtype=np.float32)

    def _build_visible_points_by_view(self, canonical_points):
        point_count = canonical_points.shape[0]
        visible_points = np.zeros((self.view_num, point_count), dtype=bool)
        if point_count == 0:
            return visible_points

        canonical_tensor = canonical_points[np.newaxis, :, :].astype(np.float32)
        canonical_tensor = torch.tensor(canonical_tensor).to(self.DEVICE)

        for view_id in range(self.view_num):
            view_points = self.shapenet_reader.get_point_cloud_by_view_id(view_id)
            if view_points is None or view_points.shape[0] == 0:
                continue
            view_tensor = view_points[np.newaxis, :, :].astype(np.float32)
            view_tensor = torch.tensor(view_tensor).to(self.DEVICE)
            _, dist_to_canonical = ChamferDistanceFunction.apply(
                view_tensor, canonical_tensor
            )
            visible_points[view_id] = (
                dist_to_canonical.detach().cpu().numpy()[0] < self.COVERAGE_THRESHOLD
            )

        return visible_points

    def _sync_legacy_coverage_buffers(self):
        if self._transition_state is None:
            return

        coverage_map = np.asarray(self._transition_state["coverage_map"], dtype=bool)
        self.current_points_cloud_from_gt = self._canonical_points[coverage_map]
        self.ground_truth_points_cloud = self._canonical_points[~coverage_map]
        self.ground_truth_tensor = self.ground_truth_points_cloud[
            np.newaxis, :, :
        ].astype(np.float32)
        self.ground_truth_tensor = torch.tensor(self.ground_truth_tensor).to(
            self.DEVICE
        )

    def _initialize_state_transition_for_current_model(self, initial_view):
        model_name = self.shapenet_reader.get_model_info()
        cached_geometry = self._model_transition_cache.get(model_name)

        if cached_geometry is None:
            self._canonical_points = np.asarray(
                self.shapenet_reader.ground_truth, dtype=np.float32
            )
            surface_normals = estimate_surface_normals(self._canonical_points)
            visible_points_by_view = self._build_visible_points_by_view(
                self._canonical_points
            )
            cached_geometry = {
                "canonical_points": self._canonical_points,
                "surface_normals": surface_normals,
                "visible_points_by_view": visible_points_by_view,
            }
            self._model_transition_cache[model_name] = cached_geometry
        else:
            self._canonical_points = cached_geometry["canonical_points"]

        self.ground_truth_points_cloud_size = self._canonical_points.shape[0]
        self.geometry_cache = {
            "canonical_points": cached_geometry["canonical_points"],
            "surface_normals": cached_geometry["surface_normals"],
            "view_positions": self.viewpoints,
            "travel_times_matrix": self.travel_times,
            "visible_points_by_view": cached_geometry["visible_points_by_view"],
        }
        self._canonical_tensor = torch.tensor(
            self._canonical_points[np.newaxis, :, :].astype(np.float32)
        ).to(self.DEVICE)
        self._coverage_map = np.zeros(self.ground_truth_points_cloud_size, dtype=bool)
        self.current_coverage = 0.0
        self.coverage_add = 0.0
        self.ground_truth_points_cloud = self._canonical_points
        self.current_points_cloud_from_gt = np.zeros((0, 3), dtype=np.float32)

    def step(self, action):
        return self._step_continuous(action)

    def _step_continuous(self, action):
        # action: [theta, phi, transfer_time]
        # theta: polar angle [0, pi], phi: azimuthal angle [0, 2pi]
        # transfer_time: requested time-of-flight in [0, orbit_config.total_time]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 3:
            raise ValueError(
                "continuous action must have 3 elements [theta, phi, transfer_time], got {}".format(
                    action.shape[0]
                )
            )

        theta = float(np.clip(action[0], 0.0, np.pi))
        phi = float(np.clip(action[1], 0.0, 2 * np.pi))
        requested_transfer_time = float(action[2])

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
            travel_time = float(np.clip(requested_transfer_time, 1e-6, remaining_time))

        # 3. Compute Δv via CW dynamics (using pre-initialized instance)
        r0 = self.current_position
        rf = new_position
        delta_v, _, _ = self.cw.compute_delta_v(r0, rf, travel_time)
        if delta_v == np.inf:
            delta_v = self.max_delta_v  # fallback for singular transfers

        collision_detected, collision_penalty, collision_min_clearance = (
            self.renderer.check_transfer_collision(r0, rf, travel_time)
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
            return observation, reward, terminated, info

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
            normals = estimate_surface_normals(new_view_points)
            new_view_points = filter_lit_points(new_view_points, normals, self.current_sun_position)

        # 5. Update coverage
        self.current_points_cloud = np.append(
            self.current_points_cloud, new_view_points, axis=0
        )
        # Cap point cloud size to prevent unbounded growth
        if self.current_points_cloud.shape[0] > MAX_CLOUD_SIZE:
            idx = np.random.choice(
                self.current_points_cloud.shape[0], MAX_CLOUD_SIZE, replace=False
            )
            self.current_points_cloud = self.current_points_cloud[idx]
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

        # 6. Update state
        self.current_position = new_position
        self.cumulative_dv += delta_v
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
        self.logger.debug(
            f"[REWARD] coverage_gain={coverage_gain:7.4f}, travel_time={travel_time:7.4f}, "
            f"delta_v={delta_v:7.4f}, collision_penalty=0.0, final={reward:7.4f}"
        )

        # 8. Check termination using _get_terminated for consistency
        terminated = self._get_terminated()

        observation = self._get_observation_space()
        info = self._get_info(
            travel_time,
            delta_v,
            requested_transfer_time,
            collision_detected=False,
            collision_penalty=0.0,
            collision_min_clearance=None,
        )

        return observation, reward, terminated, info

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
        self.step_cnt = 1
        self.model_name = self.shapenet_reader.get_model_info()

        # Reset mission time and fuel
        self.current_time = 0.0
        self.cumulative_dv = 0.0
        self.logger.debug(
            f"[reset] Mission time reset to 0.0. Horizon: {self.orbit_config.total_time:.6f} time units"
        )

        self.current_position = random_position_on_sphere()
        self.current_view = 0

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

        self._initialize_state_transition_for_current_model(self.current_view)

        self.current_points_cloud = self.renderer.get_points_from_position(
            self.current_position, self._canonical_points
        )

        observation = self._get_observation_space()
        info = self._get_info()
        self.logger.debug("[reset] pass, init step: {}".format(self.current_view))
        return observation, info

    def close(self):
        pass

    def render(sellf):
        pass

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
            if self.is_normalize:
                cur_pc = normalize_pc(cur_pc, self.logger, self.model_name)
            cur_pc = cur_pc.T

        return {
            "current_point_cloud": cur_pc.astype(np.float32),
            "camera_position": self.current_position.astype(np.float32),
            "coverage": np.array([self.current_coverage], dtype=np.float32),
            "fuel_remaining": np.array(
                [max(0.0, self.fuel_budget - self.cumulative_dv) / self.fuel_budget],
                dtype=np.float32,
            ),
            "time_remaining": np.array(
                [
                    max(0.0, self.orbit_config.total_time - self.current_time)
                    / self.orbit_config.total_time
                ],
                dtype=np.float32,
            ),
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
