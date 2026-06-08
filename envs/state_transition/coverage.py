"""Coverage map update for state transition.

This module computes cumulative coverage over canonical model points,
tracking which points have been observed across the mission history.
Coverage counts only newly lit-visible points without double-counting.
"""

from __future__ import annotations

from typing import NamedTuple, Any

import numpy as np


class CoverageUpdateResult(NamedTuple):
    """Result of coverage map update.

    Attributes:
        coverage_map: Boolean mask (N,) of points covered so far (cumulative).
        newly_covered_count: Integer count of points newly covered in this step.
        newly_covered_ratio: Float ratio of newly covered points to total N_m.
    """

    coverage_map: np.ndarray
    newly_covered_count: int
    newly_covered_ratio: float


def _as_bool_mask(name: str, value: object, expected_size: int) -> np.ndarray:
    """Normalize input to boolean mask of expected size.

    Args:
        name: Parameter name for error messages.
        value: Input value (boolean array or index list).
        expected_size: Expected mask size (N points).

    Returns:
        Boolean mask with shape (expected_size,) and dtype bool.

    Raises:
        ValueError: If input cannot be normalized or size mismatch.
    """
    if isinstance(value, (list, tuple)):
        # Index list: create boolean mask from indices
        indices = np.asarray(value, dtype=int).ravel()
        if indices.size == 0:
            return np.zeros(expected_size, dtype=bool)
        if np.any(indices < 0) or np.any(indices >= expected_size):
            raise ValueError(
                f"{name} indices out of range [0, {expected_size}): "
                f"got min={indices.min()}, max={indices.max()}"
            )
        mask = np.zeros(expected_size, dtype=bool)
        mask[indices] = True
        return mask

    # Boolean array
    arr = np.asarray(value, dtype=bool)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.shape[0] != expected_size:
        raise ValueError(
            f"{name} size mismatch: expected {expected_size}, got {arr.shape[0]}"
        )
    return arr.astype(bool)


def _as_int_scalar(name: str, value: object, non_negative: bool = True) -> int:
    """Normalize input to non-negative integer scalar.

    Args:
        name: Parameter name for error messages.
        value: Input value.
        non_negative: If True, require value >= 0.

    Returns:
        Integer value.

    Raises:
        TypeError: If not convertible to integer.
        ValueError: If out of range.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be int, not bool")
    try:
        scalar = int(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be convertible to int: {e}")

    if non_negative and scalar < 0:
        raise ValueError(f"{name} must be >= 0, got {scalar}")
    return scalar


def update_coverage_map(
    prev_coverage_map: object,
    visible_lit_points: object,
    total_points: int | None = None,
) -> CoverageUpdateResult:
    """Update cumulative coverage map with newly visible-lit points.

    This function computes a new coverage map by taking the union of the previous
    coverage and newly visible (lit & visible) points. It tracks both the cumulative
    coverage and the newly covered count for reward calculation.

    Idempotency:
        If visible_lit_points is a subset of prev_coverage_map, newly_covered_count
        will be 0 (no points counted twice).

    Args:
        prev_coverage_map: Previous cumulative coverage. Can be:
            - Boolean array with shape (N,), where N is total canonical points
            - Index list/array of already-covered point indices
        visible_lit_points: Newly observed lit-visible points. Can be:
            - Boolean array with shape (N,)
            - Index list/array of newly visible point indices
        total_points: Optional total point count N_m. If provided, used to validate
                     against. If not provided, inferred from prev_coverage_map shape.

    Returns:
        CoverageUpdateResult with:
        - coverage_map: New cumulative coverage mask, shape (N,), dtype bool
        - newly_covered_count: Integer count of newly covered points
        - newly_covered_ratio: Ratio newly_covered_count / N_m (0.0 to 1.0)

    Raises:
        TypeError: If inputs have wrong type.
        ValueError: If shapes don't match or contain invalid indices.
    """

    # Determine point count
    if isinstance(prev_coverage_map, np.ndarray):
        if prev_coverage_map.ndim != 1:
            raise ValueError(
                f"prev_coverage_map must be 1-D array, got shape {prev_coverage_map.shape}"
            )
        inferred_n = int(prev_coverage_map.shape[0])
    elif isinstance(prev_coverage_map, (list, tuple)):
        if total_points is None:
            raise ValueError(
                "total_points required when prev_coverage_map is index list"
            )
        inferred_n = None
    else:
        raise TypeError(
            f"prev_coverage_map must be array or index list, got {type(prev_coverage_map).__name__}"
        )

    # Validate total_points
    if total_points is not None:
        total_points = _as_int_scalar("total_points", total_points, non_negative=True)
        if inferred_n is not None and inferred_n != total_points:
            raise ValueError(
                f"total_points mismatch: array shape {inferred_n} != provided {total_points}"
            )
        n_m = total_points
    else:
        if inferred_n is None:
            raise ValueError(
                "Cannot determine point count: provide total_points or use prev_coverage_map as array"
            )
        n_m = inferred_n

    if n_m == 0:
        return CoverageUpdateResult(
            coverage_map=np.zeros(0, dtype=bool),
            newly_covered_count=0,
            newly_covered_ratio=0.0,
        )

    # Normalize to boolean masks
    prev_mask = _as_bool_mask("prev_coverage_map", prev_coverage_map, n_m)
    visible_mask = _as_bool_mask("visible_lit_points", visible_lit_points, n_m)

    # Compute newly covered points (visible points not already covered)
    newly_covered = visible_mask & ~prev_mask

    # Compute new coverage map (union)
    new_coverage_map = prev_mask | visible_mask

    # Compute results
    newly_covered_count = int(np.sum(newly_covered))
    newly_covered_ratio = float(newly_covered_count) / float(n_m)

    return CoverageUpdateResult(
        coverage_map=new_coverage_map.astype(bool),
        newly_covered_count=newly_covered_count,
        newly_covered_ratio=newly_covered_ratio,
    )

def update_continuous_coverage(
    new_points: np.ndarray,
    canonical_tensor: Any,
    prev_coverage_map: np.ndarray,
    current_coverage: float,
    ground_truth_points_cloud_size: int,
    coverage_threshold: float,
    chamfer_distance_function: Any,
    device: Any,
) -> tuple[np.ndarray, float, float]:
    """
    Update coverage for continuous view selection using Chamfer distance.
    Returns:
        tuple containing:
        - updated_coverage_map (np.ndarray)
        - new_current_coverage (float)
        - coverage_gain (float)
    """
    import torch
    if new_points.shape[0] == 0:
        return prev_coverage_map.copy(), current_coverage, 0.0

    # Convert to tensors
    new_points_tensor = torch.tensor(
        new_points[np.newaxis, :, :].astype(np.float32)
    ).to(device)

    # Calculate distance from new points to ground truth using cached tensor
    _, dist_to_gt = chamfer_distance_function.apply(
        new_points_tensor, canonical_tensor
    )

    # dist_to_gt[i] = distance from canonical point i to nearest new point
    newly_covered_mask = (
        dist_to_gt.detach().cpu().numpy()[0] < coverage_threshold
    )

    # Merge into persistent coverage map
    updated_coverage_map = prev_coverage_map | newly_covered_mask

    new_current_coverage = float(np.sum(updated_coverage_map)) / max(
        ground_truth_points_cloud_size, 1
    )
    coverage_gain = new_current_coverage - current_coverage
    
    return updated_coverage_map, new_current_coverage, coverage_gain
