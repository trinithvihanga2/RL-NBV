import numpy as np

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

def estimate_surface_normals(points):
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

def camera_axes(eye, target=None, up=None):
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
