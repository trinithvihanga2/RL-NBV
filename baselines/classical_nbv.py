"""
Classical Next-Best-View (NBV) Baselines for Autonomous Spacecraft Inspection.

Implements:
1. Volumetric Greedy NBV (G-NBV): Information-gain-based viewpoint selection
   using Rear-Side Voxel (RSV) scoring with solar illumination weighting
   (following Isler et al., ICRA 2016 and Delmerico et al., Auton. Robots 2018).
2. Cost-Aware Greedy NBV (CA-NBV): Couples predicted information gain to the
   minimum-Δv astrodynamic transfer cost returned by the SCvx trajectory planner.

Reference:
- S. Isler et al., "An information gain formulation for active volumetric 3D reconstruction," ICRA 2016.
- J. Delmerico et al., "A comparison of volumetric information gain metrics for active 3D object reconstruction," Auton. Robots 2018.
"""

from __future__ import annotations

from typing import Any, Tuple, List
import numpy as np


def fibonacci_sphere(num_points: int = 64, radius: float = 1.12) -> np.ndarray:
    """Generate approximately uniform points on a sphere using the Fibonacci lattice."""
    points = []
    phi_golden = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio
    for i in range(num_points):
        z = 1.0 - (2.0 * i + 1.0) / num_points
        theta_polar = np.arccos(np.clip(z, -1.0, 1.0))
        phi_azimuth = (2.0 * np.pi * i) / phi_golden
        
        x = radius * np.sin(theta_polar) * np.cos(phi_azimuth)
        y = radius * np.sin(theta_polar) * np.sin(phi_azimuth)
        points.append([x, y, z * radius])
    return np.asarray(points, dtype=np.float32)


def cartesian_to_action(pos: np.ndarray, orbit_radius: float, duration_norm: float = -1.0) -> np.ndarray:
    """Convert Cartesian coordinates on inspection sphere to normalized action [-1, 1]^3."""
    norm_pos = pos / orbit_radius
    x, y, z = norm_pos[0], norm_pos[1], norm_pos[2]
    
    # Polar angle theta in [0, pi] from +z
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    # Azimuth phi in [0, 2pi]
    phi = np.arctan2(y, x)
    if phi < 0:
        phi += 2.0 * np.pi
        
    norm_theta = float((theta / (np.pi / 2.0)) - 1.0)
    norm_phi = float((phi / np.pi) - 1.0)
    norm_time = float(duration_norm)
    
    return np.array([norm_theta, norm_phi, norm_time], dtype=np.float32)


class VolumetricGreedyNBVPolicy:
    """
    Baseline 1: Classical Volumetric Greedy NBV (G-NBV).
    
    Scores candidate views by predicted rear-side voxel information gain under
    projected solar illumination. Starting from the highest-scoring candidate,
    it queries the SCvx planner and selects the first dynamically feasible transfer.
    """

    def __init__(
        self,
        env: Any,
        num_candidates: int = 64,
        orbit_radius: float = 1.12,
    ):
        self.env = env
        self.num_candidates = num_candidates
        self.orbit_radius = orbit_radius
        self.candidate_views = fibonacci_sphere(num_candidates, radius=orbit_radius)
        self.step_idx = 0

    def reset(self) -> None:
        self.step_idx = 0

    def predict(self, obs: dict, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        del deterministic
        pcd = obs["current_point_cloud"].T if obs["current_point_cloud"].shape[0] == 3 else obs["current_point_cloud"]
        cam_pos = obs["camera_position"]
        current_time = getattr(self.env, "current_time", 0.0)
        
        # Candidate duration (nominal feasible multi-node transfer)
        tau = 1.5
        predicted_time = current_time + tau
        sun_dir = self._predict_sun_direction(predicted_time)
        
        # Score each candidate viewpoint
        scored_candidates = []
        for idx, view_pos in enumerate(self.candidate_views):
            gain = self._compute_rear_side_gain(pcd, view_pos, sun_dir)
            scored_candidates.append((gain, view_pos))
            
        # Sort candidates in descending order of information gain
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Check feasibility with SCvx planner
        r0 = cam_pos * self.env.orbit_config.unit_scale
        selected_view = scored_candidates[0][1]
        for gain, cand_pos in scored_candidates:
            rf = cand_pos * self.env.orbit_config.unit_scale
            delta_v, _, _ = self.env.cw.compute_delta_v(r0, rf, tau)
            if delta_v < np.inf:
                selected_view = cand_pos
                break
                
        self.step_idx += 1
        time_norm = float((tau / (getattr(self.env, "total_time", 14.89) / 2.0)) - 1.0)
        action = cartesian_to_action(selected_view, self.orbit_radius, duration_norm=time_norm)
        return action, None

    def _predict_sun_direction(self, t: float) -> np.ndarray:
        omega = 0.1
        theta = -omega * t
        return np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float32)

    def _compute_rear_side_gain(self, pcd: np.ndarray, view_pos: np.ndarray, sun_dir: np.ndarray) -> float:
        """Estimate unobserved information gain for candidate view_pos under sun_dir."""
        if len(pcd) == 0:
            return 1.0
        v_norm = view_pos / np.linalg.norm(view_pos)
        sun_dot = float(np.dot(v_norm, sun_dir))
        # Sun illumination weighting: positions on sunlit hemisphere have higher prospective gain
        sun_factor = max(0.05, (sun_dot + 1.0) / 2.0)

        # Redundancy penalty: fraction of existing reconstruction cloud observed in this viewing cone
        boresight = -v_norm
        d = pcd - view_pos
        d_norms = np.linalg.norm(d, axis=1, keepdims=True)
        d_unit = d / np.maximum(d_norms, 1e-8)
        cos_alpha = np.sum(d_unit * boresight, axis=1)

        in_fov = np.sum(cos_alpha > 0.8)
        redundancy = float(in_fov) / float(len(pcd) + 1e-6)
        novelty = 1.0 - redundancy

        return float(sun_factor * (0.2 + 0.8 * novelty))


class CostAwareGreedyNBVPolicy:
    """
    Baseline 2: SCvx Cost-Aware Greedy NBV (CA-NBV).
    
    Evaluates candidate viewpoints using a coupled utility combining predicted
    information gain, normalized SCvx propellant cost (Δv), and transfer time.
    """

    def __init__(
        self,
        env: Any,
        num_candidates: int = 32,
        orbit_radius: float = 1.12,
        fuel_budget: float = 100.0,
        total_time: float = 14.89,
    ):
        self.env = env
        self.num_candidates = num_candidates
        self.orbit_radius = orbit_radius
        self.fuel_budget = fuel_budget
        self.total_time = total_time
        self.candidate_views = fibonacci_sphere(num_candidates, radius=orbit_radius)
        self.step_idx = 0

    def reset(self) -> None:
        self.step_idx = 0

    def predict(self, obs: dict, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        del deterministic
        pcd = obs["current_point_cloud"].T if obs["current_point_cloud"].shape[0] == 3 else obs["current_point_cloud"]
        cam_pos = obs["camera_position"]
        current_time = getattr(self.env, "current_time", 0.0)
        
        # Duration candidate grid (feasible multi-node transfers N >= 2)
        tau_candidates = [0.8, 1.2, 1.6, 2.0]
        r0 = cam_pos * self.env.orbit_config.unit_scale
        
        best_utility = -float("inf")
        best_view = self.candidate_views[0]
        best_tau = 1.2
        
        for cand_pos in self.candidate_views:
            rf = cand_pos * self.env.orbit_config.unit_scale
            for tau in tau_candidates:
                delta_v, _, _ = self.env.cw.compute_delta_v(r0, rf, tau)
                if delta_v == np.inf:
                    continue
                    
                sun_dir = self._predict_sun_direction(current_time + tau)
                gain = self._compute_rear_side_gain(pcd, cand_pos, sun_dir)
                
                # Matched-objective one-step utility (Eq. 12 in Implementation Guide)
                w_cov = 1000.0
                w_f, c_f = 1.0, 10.0
                w_t, c_t = 1.0, 10.0
                
                utility = (
                    w_cov * gain
                    - (w_f * c_f * delta_v / self.fuel_budget)
                    - (w_t * c_t * tau / self.total_time)
                )
                
                if utility > best_utility:
                    best_utility = utility
                    best_view = cand_pos
                    best_tau = tau
                    
        self.step_idx += 1
        # Map tau to normalized action
        time_norm = float((best_tau / (self.total_time / 2.0)) - 1.0)
        action = cartesian_to_action(best_view, self.orbit_radius, duration_norm=time_norm)
        return action, None

    def _predict_sun_direction(self, t: float) -> np.ndarray:
        omega = 0.1
        theta = -omega * t
        return np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float32)

    def _compute_rear_side_gain(self, pcd: np.ndarray, view_pos: np.ndarray, sun_dir: np.ndarray) -> float:
        """Estimate unobserved information gain for candidate view_pos under sun_dir."""
        if len(pcd) == 0:
            return 1.0
        v_norm = view_pos / np.linalg.norm(view_pos)
        sun_dot = float(np.dot(v_norm, sun_dir))
        # Sun illumination weighting
        sun_factor = max(0.05, (sun_dot + 1.0) / 2.0)

        # Redundancy penalty
        boresight = -v_norm
        d = pcd - view_pos
        d_norms = np.linalg.norm(d, axis=1, keepdims=True)
        d_unit = d / np.maximum(d_norms, 1e-8)
        cos_alpha = np.sum(d_unit * boresight, axis=1)

        in_fov = np.sum(cos_alpha > 0.8)
        redundancy = float(in_fov) / float(len(pcd) + 1e-6)
        novelty = 1.0 - redundancy

        return float(sun_factor * (0.2 + 0.8 * novelty))
