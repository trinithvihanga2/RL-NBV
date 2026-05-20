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
    get_travel_time,
)
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Maximum size for accumulated point cloud in continuous mode (prevents unbounded growth)
MAX_CLOUD_SIZE = 8192

PARTIAL_RENDER_WIDTH = 640
PARTIAL_RENDER_HEIGHT = 480
PARTIAL_RENDER_FX = 525.0
PARTIAL_RENDER_FY = 525.0
PARTIAL_RENDER_CX = 319.5
PARTIAL_RENDER_CY = 239.5
MODEL_NORMALIZATION_SCALE = 0.8


def resample_pcd(pcd, n, logger, name):
    """Drop or duplicate points so that pcd has exactly n points"""
    if pcd.shape[0] == 0:
        logger.debug("observation source point cloud is empty, model: {}".format(name))
        return np.zeros((n, 3))
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate(
            [idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])]
        )
    logger.debug("resample_pcd from {} to {}, model: {}".format(pcd.shape[0], n, name))
    return pcd[idx[:n]]


def normalize_pc(points, logger, name):
    if points.shape[0] == 0:
        logger.debug("normalize received empty points, model: {}".format(name))
        return points
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))
    if furthest_distance == 0:
        logger.debug(
            "normalize skipped due to zero furthest distance, model: {}".format(name)
        )
        return points
    points /= furthest_distance
    logger.debug(
        "normalize furthest distance: {:.6f}, model: {}".format(furthest_distance, name)
    )
    return points


def random_position_on_sphere():
    """Generate random position on unit sphere using spherical coordinates."""
    theta = np.random.uniform(0, np.pi)  # Polar angle [0, pi]
    phi = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle [0, 2pi]
    r = 1.0  # Unit sphere

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.array([x, y, z], dtype=np.float32)


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
        self.fuel_budget = fuel_budget
        self.cumulative_dv = 0.0
        self.sun_position_config = sun_position_config or {}
        self.terminated_coverage = terminated_coverage

        from envs.state_transition.cw_utils import CWDynamics

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([np.pi, 2 * np.pi]),
            shape=(2,),
            dtype=np.float32,
        )
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
        self.current_points_cloud = self._get_points_from_position(
            self.current_position
        )
        self.current_points_cloud_from_gt = np.zeros((0, 3), dtype=np.float32)

    def _estimate_surface_normals(self, points):
        if points.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        centroid = np.mean(points, axis=0, keepdims=True)
        vectors = points - centroid
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normals = vectors / np.maximum(norms, 1e-12)
        degenerate = norms[:, 0] <= 1e-12
        if np.any(degenerate):
            normals[degenerate] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return normals.astype(np.float32)

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
            surface_normals = self._estimate_surface_normals(self._canonical_points)
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
        # action: spherical coordinates [theta, phi]
        # theta: polar angle [0, pi] (0 to 180 degrees)
        # phi: azimuthal angle [0, 2pi] (0 to 360 degrees)
        theta, phi = action[0], action[1]

        # Convert spherical to Cartesian coordinates
        r = self.orbit_config.orbit_radius
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        new_position = np.array([x, y, z], dtype=np.float32)

        # 2. Compute travel time
        travel_time = get_travel_time(
            self.current_position, new_position, self.orbit_config
        )

        # 3. Compute Δv via CW dynamics (using pre-initialized instance)
        r0 = self.current_position
        rf = new_position
        delta_v, _, _ = self.cw.compute_delta_v(r0, rf, travel_time)
        if delta_v == np.inf:
            delta_v = self.max_delta_v  # fallback for singular transfers

        # 4. Get visible points from new position (nearest view approximation)
        new_view_points = self._get_points_from_position(new_position)

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
        coverage_gain = self._update_coverage(new_view_points)

        # 6. Update state
        self.current_position = new_position
        self.current_time += travel_time
        self.cumulative_dv += delta_v
        self.step_cnt += 1

        # 7. Compute reward using _get_reward for consistency
        reward = self._get_reward(
            cover_add=coverage_gain,
            action=0,  # unused in continuous mode
            travel_time=travel_time,
            delta_v=delta_v,
        )

        # 8. Check termination using _get_terminated for consistency
        terminated = self._get_terminated()

        observation = self._get_observation_space()
        info = self._get_info(travel_time, delta_v)

        return observation, reward, terminated, info

    # for greedy policy test
    def try_step(self, action):
        # ============================================================================
        # TRY_STEP: Simulate action without updating state (for planning/evaluation)
        # ============================================================================
        # This method tests the value of an action without committing to it.
        # Used for greedy policy evaluation and planning.
        raise NotImplementedError("try_step is not supported in continuous mode")

    def reset(self, init_step=-1):
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
            action=0,
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

        self.current_points_cloud = self._get_points_from_position(
            self.current_position
        )

        observation = self._get_observation_space()
        self._get_info()
        self.logger.debug("[reset] pass, init step: {}".format(self.current_view))
        return observation

    def close(self):
        pass

    def render(sellf):
        pass

    def _caculate_current_coverage(self):
        return self.current_coverage

    def _get_reward(self, cover_add, action, travel_time=0.0, delta_v=0.0):
        """
        Calculate reward combining coverage gain and travel time cost.

        Reward = coverage_reward - time_cost_weight * travel_time - delta_v_weight * delta_v

        This encourages agent to:
        1. Maximize new coverage observation
        2. Minimize travel time (explore efficiently)
        3. Balance between distant high-value targets and nearby ones

        Args:
            cover_add: Coverage gained (0.0 to 1.0)
            action: Selected viewpoint action
            travel_time: Time cost to reach this viewpoint (dimensionless units)

        Returns:
            Scalar reward value
        """
        # ========================================================================
        # STEP 1: CALCULATE BASE COVERAGE REWARD
        # ========================================================================
        if self.is_reward_with_cur_coverage:
            # Reward based on current coverage progress
            if self.step_cnt < 4:
                coverage_reward = cover_add * 10
            else:
                if cover_add <= 0:
                    coverage_reward = cover_add * 10
                else:
                    # Reward increases as coverage completes (scarcity-based)
                    remain = 1.0 - (self.current_coverage - cover_add)
                    coverage_reward = (cover_add / remain) * 5 + cover_add * 5
        elif self.is_ratio_reward:
            # Ratio-based reward: prioritize new coverage when near completion
            if cover_add <= 0:
                coverage_reward = cover_add * 10
            else:
                remain = 1.0 - (self.current_coverage - cover_add)
                coverage_reward = (cover_add / remain) * 10
        else:
            # Simple linear reward: reward = coverage * constant
            coverage_reward = cover_add * 10

        # ========================================================================
        # STEP 2: APPLY TRAVEL TIME PENALTY
        # ========================================================================
        # Subtract time cost from coverage gain
        # This makes agent think about efficiency:
        # "Is this viewpoint worth the travel time?"
        #
        # Example:
        # - High coverage gain (0.10) but far away (travel_time=0.5)
        #   reward = 1.0 - 1.0*0.5 = 0.5
        # - Low coverage gain (0.02) but very close (travel_time=0.05)
        #   reward = 0.2 - 1.0*0.05 = 0.15
        #
        # Agent learns to balance:
        # coverage_reward / travel_time = efficiency
        normalized_travel_time = travel_time * 10 / self.max_travel_time
        normalized_delta_v = delta_v * 10 / self.max_delta_v
        time_penalty = self.time_cost_weight * normalized_travel_time
        fuel_penalty = self.delta_v_weight * normalized_delta_v
        final_reward = coverage_reward - time_penalty - fuel_penalty

        self.logger.debug(
            f"[REWARD] action={action:2d}, coverage_reward={coverage_reward:7.4f}, "
            f"time_penalty={time_penalty:7.4f}, fuel_penalty={fuel_penalty:7.4f}, final={final_reward:7.4f}"
        )

        return final_reward

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

    def _get_info(self, travel_time=0.0, delta_v=0.0):
        return self._get_info_continuous(travel_time, delta_v)

    def _get_info_continuous(self, travel_time, delta_v):
        """Get info dict for continuous mode."""
        return {
            "cur_points_cloud": self._canonical_points,
            "model_name": self.model_name,
            "current_coverage": self.current_coverage,
            "camera_position": self.current_position.copy(),
            "travel_time": travel_time,
            "delta_v": delta_v,
            "mission_time": self.current_time,
            "cumulative_dv": self.cumulative_dv,
            "fuel_remaining": max(0.0, self.fuel_budget - self.cumulative_dv),
        }

    def _camera_axes(self, eye, target=None, up=None):
        if target is None:
            target = np.zeros(3, dtype=np.float64)
        if up is None:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)

        fwd = target - eye
        fwd /= np.linalg.norm(fwd)

        if abs(np.dot(fwd, up)) > 0.999:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        up_new = np.cross(right, fwd)

        return right, up_new, fwd, eye

    def _resolve_current_model_mesh_path(self):
        model_name = self.shapenet_reader.get_model_info()
        candidate_paths = [
            os.path.join(self.data_path, model_name, "model.obj"),
            os.path.join(self.data_path, f"{model_name}.obj"),
            os.path.join(self.shapenet_reader.data_path, model_name, "model.obj"),
            os.path.join(self.shapenet_reader.data_path, f"{model_name}.obj"),
        ]
        for candidate_path in candidate_paths:
            if os.path.isfile(candidate_path):
                return candidate_path
        return None

    def _load_current_model_mesh(self):
        model_name = self.shapenet_reader.get_model_info()
        cached_mesh = self._mesh_cache.get(model_name)
        if cached_mesh is not None:
            return cached_mesh

        mesh_path = self._resolve_current_model_mesh_path()
        if mesh_path is None:
            self.logger.debug(
                "[continuous] No model.obj found for model %s; using canonical-point fallback",
                model_name,
            )
            return None

        mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
        if mesh.is_empty() or not mesh.has_vertices():
            self.logger.warning(
                "[continuous] Failed to load mesh from %s; using canonical-point fallback",
                mesh_path,
            )
            return None

        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices)
        centroid = vertices.mean(axis=0)
        mesh.translate(-centroid)
        vertices = np.asarray(mesh.vertices)
        max_dist = float(np.max(np.linalg.norm(vertices, axis=1)))
        if max_dist > 0.0:
            mesh.scale(MODEL_NORMALIZATION_SCALE / max_dist, center=(0, 0, 0))
        mesh.compute_vertex_normals()

        self._mesh_cache[model_name] = mesh
        self.logger.info(
            "[continuous] Loaded mesh for %s from %s", model_name, mesh_path
        )
        return mesh

    def _render_partial_points_from_mesh(self, mesh, position):
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_t)

        right, up_vec, fwd, origin = self._camera_axes(position)

        us = np.arange(PARTIAL_RENDER_WIDTH, dtype=np.float64)
        vs = np.arange(PARTIAL_RENDER_HEIGHT, dtype=np.float64)
        uu, vv = np.meshgrid(us, vs)

        xc = (uu - PARTIAL_RENDER_CX) / PARTIAL_RENDER_FX
        yc = (vv - PARTIAL_RENDER_CY) / PARTIAL_RENDER_FY

        dirs = (
            fwd[np.newaxis, np.newaxis, :]
            + xc[..., np.newaxis] * right[np.newaxis, np.newaxis, :]
            - yc[..., np.newaxis] * up_vec[np.newaxis, np.newaxis, :]
        )

        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        dirs_unit = (dirs / np.maximum(norms, 1e-12)).astype(np.float32)
        origins = np.full(dirs_unit.shape, origin, dtype=np.float32)

        rays = np.concatenate(
            [origins.reshape(-1, 3), dirs_unit.reshape(-1, 3)], axis=1
        )
        rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        result = scene.cast_rays(rays_t)
        t_hit = (
            result["t_hit"].numpy().reshape(PARTIAL_RENDER_HEIGHT, PARTIAL_RENDER_WIDTH)
        )

        valid = np.isfinite(t_hit) & (t_hit > 0.0)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float32)

        t_v = t_hit[valid].astype(np.float64)
        dirs_v = dirs_unit[valid].astype(np.float64)
        pts = origin + t_v[:, np.newaxis] * dirs_v

        keep = np.linalg.norm(pts, axis=1) < 1.2
        pts = pts[keep]
        return pts.astype(np.float32)

    def _get_points_from_position(self, position):
        """Generate the partial point cloud for an arbitrary camera position.

        The primary path mirrors the render pipeline used in the render tools: load the
        current model mesh, ray-cast from the requested camera position, and return the
        visible surface points in world space. If a mesh is not available, fall back to a
        geometric visibility mask over the canonical point cloud so continuous mode still
        works with pre-generated point-cloud datasets.
        """
        mesh = self._load_current_model_mesh()
        if mesh is not None:
            points = self._render_partial_points_from_mesh(mesh, position)
            if points.shape[0] > 0:
                return points

        canonical_points = np.asarray(self._canonical_points, dtype=np.float32)
        if canonical_points.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)

        position = np.asarray(position, dtype=np.float32)
        d = float(np.linalg.norm(position))
        if d < 1e-12:
            return np.zeros((0, 3), dtype=np.float32)

        u_hat = position / d
        chief_radius = float(self.orbit_config.orbit_radius)
        rhs = (chief_radius * chief_radius) / d
        visible_mask = np.asarray(canonical_points @ u_hat >= rhs, dtype=bool)
        return canonical_points[visible_mask]

    def _update_coverage(self, new_points):
        """Update coverage with new points and return coverage gain using persistent coverage map."""
        if new_points.shape[0] == 0:
            return 0.0

        # Convert to tensors
        new_points_tensor = torch.tensor(
            new_points[np.newaxis, :, :].astype(np.float32)
        ).to(self.DEVICE)

        # Calculate distance from new points to ground truth using cached tensor
        _, dist_to_gt = ChamferDistanceFunction.apply(
            new_points_tensor, self._canonical_tensor
        )

        # dist_to_gt[i] = distance from canonical point i to nearest new point
        newly_covered_mask = (
            dist_to_gt.detach().cpu().numpy()[0] < self.COVERAGE_THRESHOLD
        )

        # Merge into persistent coverage map
        self._coverage_map |= newly_covered_mask

        prev_coverage = self.current_coverage
        self.current_coverage = float(np.sum(self._coverage_map)) / max(
            self.ground_truth_points_cloud_size, 1
        )
        return self.current_coverage - prev_coverage

    def _get_debug_info(self):
        self.logger.info(
            "model name:{}, action history: {}".format(
                self.model_name, self.action_history
            )
        )


PointCloudNBVEnvLevel2 = PointCloudNextBestViewEnv
