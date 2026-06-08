"""Visibility and illumination helpers for state transition updates.

This module computes a boolean mask over continuous points that are
illuminated by the sun using simple orientation filtering.
"""

from __future__ import annotations

import numpy as np

def _normalize_rows(points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / np.maximum(norms, eps)

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
