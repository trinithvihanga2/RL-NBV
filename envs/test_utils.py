import numpy as np
import pytest
import logging
from envs.utils import (
    resample_pcd,
    normalize_pc,
    random_position_on_sphere,
    estimate_surface_normals,
    camera_axes,
)

@pytest.fixture
def logger():
    return logging.getLogger("test_logger")

def test_resample_pcd(logger):
    # Test with empty points
    empty_pcd = np.zeros((0, 3))
    res = resample_pcd(empty_pcd, 10, logger, "test")
    assert res.shape == (10, 3)
    assert np.all(res == 0)

    # Test with exact size
    pcd = np.random.rand(10, 3)
    res = resample_pcd(pcd, 10, logger, "test")
    assert res.shape == (10, 3)

    # Test with fewer points (needs upsampling)
    pcd = np.random.rand(5, 3)
    res = resample_pcd(pcd, 10, logger, "test")
    assert res.shape == (10, 3)

    # Test with more points (needs downsampling)
    pcd = np.random.rand(20, 3)
    res = resample_pcd(pcd, 10, logger, "test")
    assert res.shape == (10, 3)

def test_normalize_pc(logger):
    empty_pcd = np.zeros((0, 3))
    res = normalize_pc(empty_pcd, logger, "test")
    assert res.shape == (0, 3)

    # All zeros
    pcd = np.zeros((10, 3))
    res = normalize_pc(pcd, logger, "test")
    assert np.all(res == 0)

    # Normal case
    pcd = np.array([
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0]
    ])
    res = normalize_pc(pcd, logger, "test")
    # Centroid is (0,0,0). Max distance is sqrt(3) ~ 1.732
    max_dist = np.max(np.linalg.norm(res, axis=1))
    np.testing.assert_allclose(max_dist, 1.0)

def test_random_position_on_sphere():
    pos = random_position_on_sphere()
    assert pos.shape == (3,)
    np.testing.assert_allclose(np.linalg.norm(pos), 1.0)

    pos = random_position_on_sphere(radius=5.0)
    np.testing.assert_allclose(np.linalg.norm(pos), 5.0)

def test_estimate_surface_normals():
    empty = np.zeros((0, 3))
    res = estimate_surface_normals(empty)
    assert res.shape == (0, 3)

    pcd = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])
    normals = estimate_surface_normals(pcd)
    assert normals.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)

    # Test degenerate case (all same points)
    pcd_deg = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    normals = estimate_surface_normals(pcd_deg)
    # Fallbacks to [0, 0, 1]
    np.testing.assert_allclose(normals[0], [0.0, 0.0, 1.0])

def test_camera_axes():
    eye = [1.0, 0.0, 0.0]
    right, up, fwd, origin = camera_axes(eye)
    
    np.testing.assert_allclose(origin, eye)
    np.testing.assert_allclose(np.linalg.norm(right), 1.0)
    np.testing.assert_allclose(np.linalg.norm(up), 1.0)
    np.testing.assert_allclose(np.linalg.norm(fwd), 1.0)
    
    # Orthogonality
    assert abs(np.dot(right, fwd)) < 1e-6
    assert abs(np.dot(right, up)) < 1e-6
    assert abs(np.dot(up, fwd)) < 1e-6
