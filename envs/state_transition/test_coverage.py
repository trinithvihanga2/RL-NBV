import numpy as np
from envs.state_transition.coverage import update_continuous_coverage

class MockChamferDistance:
    def __init__(self, distances):
        self.distances = distances

    def apply(self, new_points, canonical):
        # returns dummy_forward_distance, dist_to_gt
        import torch
        return None, torch.tensor([self.distances], dtype=torch.float32)

def test_update_continuous_coverage_empty():
    import torch
    empty_points = np.zeros((0, 3))
    prev_map = np.zeros(10, dtype=bool)
    
    updated, current_cov, gain = update_continuous_coverage(
        new_points=empty_points,
        canonical_tensor=torch.zeros((1, 10, 3)),
        prev_coverage_map=prev_map,
        current_coverage=0.0,
        ground_truth_points_cloud_size=10,
        coverage_threshold=0.05,
        chamfer_distance_function=None,
        device=torch.device("cpu")
    )
    assert np.all(updated == 0)
    assert current_cov == 0.0
    assert gain == 0.0

def test_update_continuous_coverage_gain():
    import torch
    new_points = np.ones((5, 3))
    prev_map = np.zeros(10, dtype=bool)
    
    # Let 3 points be within threshold, 7 points outside
    mock_distances = [0.01, 0.02, 0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    mock_cd = MockChamferDistance(mock_distances)
    
    updated, current_cov, gain = update_continuous_coverage(
        new_points=new_points,
        canonical_tensor=torch.zeros((1, 10, 3)),
        prev_coverage_map=prev_map,
        current_coverage=0.0,
        ground_truth_points_cloud_size=10,
        coverage_threshold=0.05,
        chamfer_distance_function=mock_cd,
        device=torch.device("cpu")
    )
    
    # 3 points are newly covered (< 0.05)
    assert np.sum(updated) == 3
    assert current_cov == 0.3
    assert gain == 0.3
    
    # Run again with same distances, should not gain more
    updated2, current_cov2, gain2 = update_continuous_coverage(
        new_points=new_points,
        canonical_tensor=torch.zeros((1, 10, 3)),
        prev_coverage_map=updated,
        current_coverage=current_cov,
        ground_truth_points_cloud_size=10,
        coverage_threshold=0.05,
        chamfer_distance_function=mock_cd,
        device=torch.device("cpu")
    )
    assert np.sum(updated2) == 3
    assert current_cov2 == 0.3
    assert gain2 == 0.0
