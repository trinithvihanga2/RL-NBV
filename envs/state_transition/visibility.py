"""Visibility and illumination helpers for state transition updates.

This module computes a boolean mask over canonical model points that are both:
1) visible from the selected view using the perception-cone condition, and
2) illuminated according to a Blinn-Phong shading model with thresholded output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _as_vector3(name: str, value: object) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values")
    return vector


def _as_points(name: str, value: object) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain finite values")
    return points


def _normalize_rows(points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / np.maximum(norms, eps)


def _extract_point_count(geometry_cache: Mapping[str, object]) -> int:
    if "canonical_points" in geometry_cache:
        points = _as_points(
            "geometry_cache['canonical_points']", geometry_cache["canonical_points"]
        )
        return int(points.shape[0])

    if "point_count" in geometry_cache:
        point_count = int(geometry_cache["point_count"])
        if point_count < 0:
            raise ValueError("geometry_cache['point_count'] must be >= 0")
        return point_count

    raise ValueError("geometry_cache must include 'canonical_points' or 'point_count'")


def _visible_mask_from_cache(
    action: int, point_count: int, geometry_cache: Mapping[str, object]
) -> np.ndarray:
    if "visible_points_by_view" in geometry_cache:
        visible_points_by_view = np.asarray(geometry_cache["visible_points_by_view"])
        if visible_points_by_view.ndim != 2:
            raise ValueError(
                "geometry_cache['visible_points_by_view'] must have shape (V, N)"
            )
        if action < 0 or action >= visible_points_by_view.shape[0]:
            raise IndexError(
                f"action {action} out of range for visible_points_by_view with {visible_points_by_view.shape[0]} views"
            )
        if visible_points_by_view.shape[1] != point_count:
            raise ValueError(
                "geometry_cache['visible_points_by_view'] point dimension does not match canonical indexing"
            )
        return np.asarray(visible_points_by_view[action], dtype=bool)

    if "visible_indices_by_view" in geometry_cache:
        visible_indices_by_view = geometry_cache["visible_indices_by_view"]
        if not isinstance(visible_indices_by_view, Sequence):
            raise ValueError(
                "geometry_cache['visible_indices_by_view'] must be a sequence"
            )
        if action < 0 or action >= len(visible_indices_by_view):
            raise IndexError(
                f"action {action} out of range for visible_indices_by_view with {len(visible_indices_by_view)} views"
            )
        indices = np.asarray(visible_indices_by_view[action], dtype=int)
        mask = np.zeros(point_count, dtype=bool)
        if indices.size == 0:
            return mask
        if np.any(indices < 0) or np.any(indices >= point_count):
            raise ValueError(
                "visible_indices_by_view contains out-of-range point index"
            )
        mask[indices] = True
        return mask

    canonical_points = _as_points(
        "geometry_cache['canonical_points']", geometry_cache["canonical_points"]
    )
    if point_count == 0:
        return np.zeros(0, dtype=bool)

    if "sensor_position" in geometry_cache:
        p_d = _as_vector3(
            "geometry_cache['sensor_position']", geometry_cache["sensor_position"]
        )
    elif "view_positions" in geometry_cache:
        view_positions = _as_points(
            "geometry_cache['view_positions']", geometry_cache["view_positions"]
        )
        if action < 0 or action >= view_positions.shape[0]:
            raise IndexError(
                f"action {action} out of range for view_positions with {view_positions.shape[0]} views"
            )
        p_d = np.asarray(view_positions[action], dtype=float)
    else:
        raise ValueError(
            "geometry_cache must include either visibility cache ('visible_points_by_view' or 'visible_indices_by_view') "
            "or geometric data ('canonical_points' with 'view_positions' or 'sensor_position')"
        )

    d = float(np.linalg.norm(p_d))
    if d < 1e-12:
        raise ValueError(
            "sensor position norm must be > 0 for perception-cone visibility"
        )

    chief_radius = float(geometry_cache.get("chief_radius", 1.0))
    if not np.isfinite(chief_radius) or chief_radius <= 0.0:
        raise ValueError(
            "geometry_cache['chief_radius'] must be a positive finite scalar"
        )

    u_hat = p_d / d
    rhs = (chief_radius * chief_radius) / d
    lhs = canonical_points @ u_hat
    return np.asarray(lhs >= rhs, dtype=bool)


def _illumination_mask(
    point_count: int,
    new_sun_position: np.ndarray,
    action: int,
    geometry_cache: Mapping[str, object],
) -> np.ndarray:
    if point_count == 0:
        return np.zeros(0, dtype=bool)

    if "surface_normals" not in geometry_cache:
        raise ValueError("geometry_cache must include 'surface_normals'")

    normals = _as_points(
        "geometry_cache['surface_normals']", geometry_cache["surface_normals"]
    )
    if normals.shape[0] != point_count:
        raise ValueError("surface_normals must align to canonical point indexing")

    if "canonical_points" in geometry_cache:
        canonical_points = _as_points(
            "geometry_cache['canonical_points']", geometry_cache["canonical_points"]
        )
    else:
        canonical_points = np.zeros((point_count, 3), dtype=float)

    if "sensor_position" in geometry_cache:
        sensor_position = _as_vector3(
            "geometry_cache['sensor_position']", geometry_cache["sensor_position"]
        )
    elif "view_positions" in geometry_cache:
        view_positions = _as_points(
            "geometry_cache['view_positions']", geometry_cache["view_positions"]
        )
        if action < 0 or action >= view_positions.shape[0]:
            raise IndexError(
                f"action {action} out of range for view_positions with {view_positions.shape[0]} views"
            )
        sensor_position = np.asarray(view_positions[action], dtype=float)
    else:
        raise ValueError(
            "geometry_cache must include 'view_positions' or 'sensor_position' for illumination"
        )

    l_norm = float(np.linalg.norm(new_sun_position))
    if l_norm < 1e-12:
        raise ValueError("new_sun_position must have non-zero norm")

    n_hat = _normalize_rows(normals)
    l_hat = new_sun_position / l_norm

    v_hat = _normalize_rows(sensor_position[None, :] - canonical_points)
    h_hat = _normalize_rows(v_hat + l_hat[None, :])

    ndotl = np.maximum(n_hat @ l_hat, 0.0)
    ndoth = np.maximum(np.einsum("ij,ij->i", n_hat, h_hat), 0.0)

    ia = _as_vector3(
        "light_ambient", geometry_cache.get("light_ambient", (1.0, 1.0, 1.0))
    )
    id_ = _as_vector3(
        "light_diffuse", geometry_cache.get("light_diffuse", (1.0, 1.0, 1.0))
    )
    is_ = _as_vector3(
        "light_specular", geometry_cache.get("light_specular", (1.0, 1.0, 1.0))
    )

    ka = _as_vector3(
        "chief_ambient", geometry_cache.get("chief_ambient", (0.4, 0.4, 0.4))
    )
    kd = _as_vector3(
        "chief_diffuse", geometry_cache.get("chief_diffuse", (0.1, 0.1, 0.1))
    )
    ks = _as_vector3(
        "chief_specular", geometry_cache.get("chief_specular", (1.0, 1.0, 1.0))
    )

    shininess = float(geometry_cache.get("shininess", 100.0))
    if not np.isfinite(shininess) or shininess < 0.0:
        raise ValueError("geometry_cache['shininess'] must be a finite scalar >= 0")

    ambient_rgb = (ka * ia)[None, :]
    diffuse_rgb = ndotl[:, None] * (kd * id_)[None, :]
    specular_rgb = (ndoth[:, None] ** shininess) * (ks * is_)[None, :]
    rgb = ambient_rgb + diffuse_rgb + specular_rgb
    intensity = np.linalg.norm(rgb, axis=1)

    # Ensure physically lit side only; then threshold intensity to strict boolean.
    lit_by_orientation = ndotl > 0.0

    if "dark_thresh" in geometry_cache or "bright_thresh" in geometry_cache:
        dark_thresh = float(geometry_cache.get("dark_thresh", 0.0))
        bright_thresh = float(geometry_cache.get("bright_thresh", np.inf))
        if dark_thresh < 0.0 or not np.isfinite(dark_thresh):
            raise ValueError("geometry_cache['dark_thresh'] must be finite and >= 0")
        if bright_thresh <= 0.0:
            raise ValueError("geometry_cache['bright_thresh'] must be > 0")
        if bright_thresh < dark_thresh:
            raise ValueError("geometry_cache['bright_thresh'] must be >= dark_thresh")
        lit_by_threshold = (intensity >= dark_thresh) & (intensity <= bright_thresh)
    else:
        min_intensity = float(geometry_cache.get("min_lit_intensity", 0.0))
        max_intensity = float(geometry_cache.get("max_lit_intensity", np.inf))
        if min_intensity < 0.0 or not np.isfinite(min_intensity):
            raise ValueError(
                "geometry_cache['min_lit_intensity'] must be finite and >= 0"
            )
        if max_intensity <= 0.0:
            raise ValueError("geometry_cache['max_lit_intensity'] must be > 0")
        if max_intensity < min_intensity:
            raise ValueError(
                "geometry_cache['max_lit_intensity'] must be >= min_lit_intensity"
            )
        lit_by_threshold = (intensity >= min_intensity) & (intensity <= max_intensity)

    return np.asarray(lit_by_orientation & lit_by_threshold, dtype=bool)


def get_lit_visible_points(
    action: int,
    new_sun_position: object,
    geometry_cache: Mapping[str, object],
) -> np.ndarray:
    """Return canonical-point mask for points that are both visible and lit.

    Args:
        action: Selected view index.
        new_sun_position: Sun direction vector at current mission time.
        geometry_cache: Geometry and optional cache data. Supported keys:
            - canonical_points: (N,3) model points defining canonical indexing.
            - point_count: Optional fallback if canonical_points absent.
            - surface_normals: (N,3) normals aligned to canonical points.
            - view_positions: (V,3) sensor positions by action.
            - sensor_position: Optional explicit sensor position for this action.
            - chief_radius: Radius used in perception-cone visibility condition.
            - visible_points_by_view: Optional precomputed (V,N) visibility mask.
            - visible_indices_by_view: Optional per-view point index sequence.
            - Blinn-Phong parameters and thresholds (optional):
              light_ambient/light_diffuse/light_specular,
              chief_ambient/chief_diffuse/chief_specular,
              shininess, dark_thresh/bright_thresh,
              min_lit_intensity/max_lit_intensity.

    Returns:
        np.ndarray: Boolean mask with shape (N,) and dtype bool.
    """

    if not isinstance(action, int) or isinstance(action, bool):
        raise TypeError(f"action must be int, got {type(action).__name__}")

    if not isinstance(geometry_cache, Mapping):
        raise TypeError("geometry_cache must be a mapping")

    sun_position = _as_vector3("new_sun_position", new_sun_position)
    point_count = _extract_point_count(geometry_cache)

    if point_count == 0:
        return np.zeros(0, dtype=bool)

    visible_mask = _visible_mask_from_cache(action, point_count, geometry_cache)
    illumination_mask = _illumination_mask(
        point_count, sun_position, action, geometry_cache
    )

    if visible_mask.shape != (point_count,):
        raise ValueError("visible mask shape does not match canonical point count")
    if illumination_mask.shape != (point_count,):
        raise ValueError("illumination mask shape does not match canonical point count")

    return np.asarray(visible_mask & illumination_mask, dtype=bool)

def filter_lit_points(points: np.ndarray, normals: np.ndarray, sun_position: np.ndarray) -> np.ndarray:
    """Filter physically visible points to keep only those illuminated by the sun."""
    if points.shape[0] == 0:
        return points

    l_norm = float(np.linalg.norm(sun_position))
    if l_norm < 1e-12:
        return points  # If no sun direction, assume all lit

    l_hat = sun_position / l_norm
    n_hat = _normalize_rows(normals)

    ndotl = np.einsum("ij,j->i", n_hat, l_hat)
    lit_mask = ndotl > 0.0

    return points[lit_mask]
