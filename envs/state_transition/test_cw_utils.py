import numpy as np
from envs.state_transition.cw_utils import CWDynamics

def test_cw_dynamics_zero_time():
    cw = CWDynamics(mean_motion=1.0)
    r0 = np.array([1.0, 0.0, 0.0])
    rf = np.array([0.0, 1.0, 0.0])
    
    dv, v0, vf = cw.compute_delta_v(r0, rf, 0.0)
    assert dv == 0.0
    np.testing.assert_allclose(v0, [0, 0, 0])
    np.testing.assert_allclose(vf, [0, 0, 0])

def test_cw_dynamics_inside_koz():
    cw = CWDynamics(mean_motion=1.0)
    # Start inside the KOZ (radius 0.5 < rKOZ 0.95) -> infeasible
    r0 = np.array([0.5, 0.0, 0.0])
    rf = np.array([1.0, 0.0, 0.0])
    
    dv, v0, vf = cw.compute_delta_v(r0, rf, np.pi)
    assert dv == np.inf
    assert v0 is None
    assert vf is None

def test_cw_dynamics_quarter_orbit():
    cw = CWDynamics(mean_motion=1.0)
    r0 = np.array([1.0, 0.0, 0.0])
    rf = np.array([0.0, 1.0, 0.0])
    t = np.pi / 2.0
    
    dv, v0, vf = cw.compute_delta_v(r0, rf, t)
    assert np.isfinite(dv) and dv > 0.0
    assert cw.last_trajectory is not None
    np.testing.assert_allclose(cw.last_trajectory[0], r0, atol=1e-6)
    np.testing.assert_allclose(cw.last_trajectory[-1], rf, atol=1e-6)
    # Verify all trajectory nodes respect the KOZ radius (0.95)
    radii = np.linalg.norm(cw.last_trajectory, axis=1)
    assert np.all(radii >= 0.95 - 1e-4)

def test_cw_dynamics_short_transfer():
    cw = CWDynamics(mean_motion=0.84364, unit_scale=6.25)
    r0 = np.array([1.12, 0.0, 0.0]) * 6.25
    rf = np.array([0.0, 1.12, 0.0]) * 6.25
    # Short transfer (t = 0.2 < dt = 0.5) must be feasible with terminal braking impulse
    dv, v0, vf = cw.compute_delta_v(r0, rf, 0.2)
    assert np.isfinite(dv) and dv > 0.0
    assert v0 is not None and vf is not None
