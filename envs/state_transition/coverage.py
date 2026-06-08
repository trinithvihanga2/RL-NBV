"""Coverage map update for state transition.

This module computes cumulative coverage over canonical model points
for continuous view selection using Chamfer distance.
"""

from __future__ import annotations

import numpy as np
from typing import Any

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
