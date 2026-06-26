import numpy as np
from envs.state_transition.visibility import filter_lit_points

def test_filter_lit_points_empty():
    points = np.zeros((0, 3))
    normals = np.zeros((0, 3))
    sun = np.array([1.0, 0.0, 0.0])
    res = filter_lit_points(points, normals, sun)
    assert res.shape == (0, 3)

def test_filter_lit_points_no_sun():
    points = np.array([[1.0, 0.0, 0.0]])
    normals = np.array([[1.0, 0.0, 0.0]])
    sun = np.array([0.0, 0.0, 0.0])
    res = filter_lit_points(points, normals, sun)
    assert res.shape == (1, 3)

def test_filter_lit_points_filtering():
    points = np.array([
        [1.0, 0.0, 0.0],  # normal faces sun
        [-1.0, 0.0, 0.0], # normal faces away
        [0.0, 1.0, 0.0],  # normal orthogonal to sun
    ])
    normals = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    sun = np.array([1.0, 0.0, 0.0])
    res = filter_lit_points(points, normals, sun)
    
    # Only the first point should be kept (dot product > 0)
    assert res.shape == (1, 3)
    np.testing.assert_allclose(res[0], points[0])
