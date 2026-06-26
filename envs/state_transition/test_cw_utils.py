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

def test_cw_dynamics_singular():
    cw = CWDynamics(mean_motion=1.0)
    r0 = np.array([1.0, 0.0, 0.0])
    rf = np.array([-1.0, 0.0, 0.0])
    
    # At t = pi/n, the matrix is singular
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
    
    # Check that propagating forward yields rf
    rf_check, _ = cw.compute_final_velocity(r0, v0, t)
    np.testing.assert_allclose(rf_check, rf, atol=1e-7)
