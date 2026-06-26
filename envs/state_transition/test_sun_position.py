import math
import numpy as np
import pytest
from envs.state_transition.sun_position import calculate_sun_position

def test_calculate_sun_position_angular_velocity():
    orbital_params = {
        "angular_velocity_rad_per_s": math.pi / 2.0,  # 90 degrees per second
        "initial_phase_rad": 0.0,
        "time_offset_s": 0.0
    }
    prev_pos = [1.0, 0.0, 0.0]

    # t = 0
    pos0 = calculate_sun_position(0.0, prev_pos, orbital_params)
    np.testing.assert_allclose(pos0, [1.0, 0.0, 0.0], atol=1e-7)

    # t = 1.0 -> -90 degrees (theta0 - omega * t)
    pos1 = calculate_sun_position(1.0, prev_pos, orbital_params)
    np.testing.assert_allclose(pos1, [0.0, -1.0, 0.0], atol=1e-7)

def test_calculate_sun_position_period():
    orbital_params = {
        "period_s": 4.0,  # 360 degrees in 4 seconds -> 90 degrees/s
        "initial_phase_rad": math.pi,  # Starts at 180 degrees
    }
    prev_pos = [-1.0, 0.0, 0.0]

    # t = 0
    pos0 = calculate_sun_position(0.0, prev_pos, orbital_params)
    np.testing.assert_allclose(pos0, [-1.0, 0.0, 0.0], atol=1e-7)

    # t = 1.0 -> 180 - 90 = 90 degrees
    pos1 = calculate_sun_position(1.0, prev_pos, orbital_params)
    np.testing.assert_allclose(pos1, [0.0, 1.0, 0.0], atol=1e-7)

def test_calculate_sun_position_validation():
    with pytest.raises(ValueError):
        calculate_sun_position(0.0, [1.0, 0.0], {"period_s": 1.0}) # Wrong prev shape
    
    with pytest.raises(ValueError):
        calculate_sun_position(0.0, [1.0, 0.0, 0.0], {}) # Missing params
