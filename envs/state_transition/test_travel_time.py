import pytest
import numpy as np
from envs.state_transition.travel_time import TargetOrbitConfig, advance_time

def test_target_orbit_config_init():
    config = TargetOrbitConfig(
        orbit_radius=2.0,
        grav_param=8.0,
        num_orbits=3.0,
        min_transfer_time=0.1,
        unit_scale=2.0
    )
    # n = sqrt(8 / 2^3) = sqrt(8 / 8) = 1.0
    assert pytest.approx(config.mean_motion) == 1.0
    
    # P = 2pi / 1.0 = 2pi
    assert pytest.approx(config.orbital_period) == 2 * np.pi
    
    # Total time = 3 * 2pi = 6pi
    assert pytest.approx(config.total_time) == 6 * np.pi

def test_advance_time():
    # Basic
    assert advance_time(1.0, 2.0) == 3.0
    
    # Clamping
    assert advance_time(8.0, 3.0, total_mission_time=10.0, wrap_around=False) == 10.0
    
    # Wrap around
    assert advance_time(8.0, 3.0, total_mission_time=10.0, wrap_around=True) == 1.0

    # Validation
    with pytest.raises(ValueError):
        advance_time(-1.0, 2.0)
    with pytest.raises(ValueError):
        advance_time(1.0, -2.0)
