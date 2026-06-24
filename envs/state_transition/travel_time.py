"""
Travel time calculation for movement between viewpoints on a unit sphere.

Uses constant angular velocity model for a camera/observer rotating around a target object.
Points are assumed to be normalized to unit sphere surface in dimensionless units.
All parameters are in abstract units (not kilometers or meters).
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TargetOrbitConfig:
    """
    Orbital mission parameters for rotating around a target object.
    
    All parameters work in dimensionless units (no specific unit conversion).
    Useful for unit sphere coordinates and abstract orbital mechanics simulations.
    """

    def __init__(
        self,
        orbit_radius: float = 1.0,
        grav_param: float = 1.0,
        num_orbits: float = 2.0,
        min_transfer_time: float = 1e-6,
        unit_scale: float = 1.0,
    ):
        """
        Initialize target orbit configuration.

        Args:
            orbit_radius: Orbital radius from target center (dimensionless units).
                         The camera rotates at this radius around the target.
                         Default: 1.0 (unit sphere).
            grav_param: Gravitational parameter (dimensionless units). 
                       This controls orbital dynamics and travel time scaling.
                       Default: 1.0.
            num_orbits: Number of complete orbits for mission horizon.
                       Default: 2.0.
        """
        self.orbit_radius = orbit_radius
        self.grav_param = grav_param
        self.num_orbits = num_orbits
        self.min_transfer_time = min_transfer_time
        self.unit_scale = unit_scale

        # Mean motion (angular velocity of circular orbit)
        # n = sqrt(grav_param / orbit_radius^3)
        self.mean_motion = np.sqrt(grav_param / (orbit_radius**3))

        # Single orbital period (dimensionless time units)
        # P = 2π / n
        self.orbital_period = 2.0 * np.pi / self.mean_motion

        # Total mission time (dimensionless time units)
        # T_total = num_orbits * P
        self.total_time = num_orbits * self.orbital_period

        # Angular velocity for unit sphere traversal (dimensionless)
        # omega = 2π / T_total
        self.angular_velocity = 2.0 * np.pi / self.total_time

        logger.info(
            f"TargetOrbitConfig: r_orbit={self.orbit_radius:.4f}, "
            f"μ={self.grav_param:.4f}, "
            f"P_orbit={self.orbital_period:.4f}, "
            f"T_total={self.total_time:.4f}, "
            f"ω={self.angular_velocity:.6f}"
        )


# Backward compatibility alias
OrbitalConfig = TargetOrbitConfig


def advance_time(
    current_time: float,
    travel_time: float,
    total_mission_time: Optional[float] = None,
    wrap_around: bool = False,
) -> float:
    """
    Calculate new absolute time after traveling for travel_time.

    Args:
        current_time: Current absolute timestamp (seconds).
        travel_time: Time to advance (seconds).
        total_mission_time: Total mission horizon. If provided, constrains result to [0, total_mission_time].
                           If None, time can exceed total_mission_time.
        wrap_around: If True and result exceeds total_mission_time, wraps time around.
                    If False, clamps to total_mission_time.

    Returns:
        New absolute time.

    Raises:
        ValueError: If inputs are invalid.
    """
    if current_time < 0:
        raise ValueError(f"current_time must be >= 0, got {current_time}")
    if travel_time < 0:
        raise ValueError(f"travel_time must be >= 0, got {travel_time}")

    new_time = current_time + travel_time

    # Honor mission time constraint if provided
    if total_mission_time is not None:
        if wrap_around and new_time > total_mission_time:
            new_time = new_time % total_mission_time
        elif new_time > total_mission_time:
            new_time = total_mission_time

    return new_time
