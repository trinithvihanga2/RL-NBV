"""Sun-position transition helpers.

Coordinate frame and units
--------------------------
- Frame: Hill frame (x-hat, y-hat, z-hat).
- Output: unit sun-direction vector r_hat_s(t) with shape (3,).
- Angles: radians.
- Time: seconds.

Sun dynamics model
------------------
This implementation follows the provided formulation:

        r_hat_s = [cos(theta_s), sin(theta_s), 0]
        theta_s(t + dt) = theta_s(t) - n * dt

Equivalent closed form used here:

        theta_s(t) = theta0 - n * (t - t0)

where:
- n comes from either angular_velocity_rad_per_s or period_s.
- t0 is time_offset_s (default: 0).
- theta0 is initial_phase_rad (default: 0).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def _as_finite_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real scalar, got {type(value).__name__}")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value}")
    return scalar


def _validate_prev_sun_position(prev_sun_position: object) -> np.ndarray:
    vector = np.asarray(prev_sun_position, dtype=float)
    if vector.shape != (3,):
        raise ValueError(
            "prev_sun_position must be array-like with shape (3,), "
            f"got shape {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("prev_sun_position must contain finite values")
    return vector


def _get_omega(orbital_params: Mapping[str, object]) -> float:
    if "angular_velocity_rad_per_s" in orbital_params:
        omega = _as_finite_scalar(
            "orbital_params['angular_velocity_rad_per_s']",
            orbital_params["angular_velocity_rad_per_s"],
        )
        if omega < 0.0:
            raise ValueError("angular_velocity_rad_per_s must be >= 0")
        return omega

    if "period_s" in orbital_params:
        period_s = _as_finite_scalar(
            "orbital_params['period_s']", orbital_params["period_s"]
        )
        if period_s <= 0.0:
            raise ValueError("period_s must be > 0")
        return (2.0 * math.pi) / period_s

    raise ValueError(
        "orbital_params must include either 'angular_velocity_rad_per_s' or 'period_s'"
    )


def calculate_sun_position(
    new_time: float,
    prev_sun_position: object,
    orbital_params: Mapping[str, object],
) -> np.ndarray:
    """Compute time-dependent unit sun-direction vector.

    Args:
        new_time: Absolute timestamp in seconds.
        prev_sun_position: Previous sun direction vector, shape (3,).
            Validated for compatibility and future extension, not required by the
            baseline equation.
        orbital_params: Orbital constants/config map. Supported keys:
            - angular_velocity_rad_per_s (float, >= 0) OR period_s (float, > 0)
            - initial_phase_rad (float, default 0)
            - time_offset_s (float, default 0)

    Returns:
        np.ndarray: Unit vector with shape (3,) in the Hill frame.
    """

    _validate_prev_sun_position(prev_sun_position)

    if not isinstance(orbital_params, Mapping):
        raise TypeError("orbital_params must be a mapping")

    t = _as_finite_scalar("new_time", new_time)
    omega = _get_omega(orbital_params)

    t0 = _as_finite_scalar(
        "orbital_params['time_offset_s']", orbital_params.get("time_offset_s", 0.0)
    )
    theta0 = _as_finite_scalar(
        "orbital_params['initial_phase_rad']",
        orbital_params.get("initial_phase_rad", 0.0),
    )

    # Screenshot model: theta(t + dt) = theta(t) - n * dt.
    theta = theta0 - omega * (t - t0)

    direction = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError(
            "computed sun direction has zero norm; check orbital parameters"
        )
    return direction / norm
