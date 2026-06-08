"""Reward calculation for state transition.

Implements reward policy based on continuous coverage gain, travel time, and fuel constraints.
"""

from __future__ import annotations

def calculate_continuous_reward(
    cover_add: float,
    current_coverage: float,
    step_cnt: int,
    travel_time: float,
    max_travel_time: float,
    delta_v: float,
    max_delta_v: float,
    collision_penalty: float,
    is_reward_with_cur_coverage: bool = False,
    is_ratio_reward: bool = False,
    time_cost_weight: float = 1.0,
    delta_v_weight: float = 1.0,
) -> float:
    """Calculate reward for continuous point cloud viewpoint selection."""
    if is_reward_with_cur_coverage:
        if step_cnt < 4:
            coverage_reward = cover_add * 10
        else:
            if cover_add <= 0:
                coverage_reward = cover_add * 10
            else:
                remain = 1.0 - (current_coverage - cover_add)
                coverage_reward = (cover_add / remain) * 5 + cover_add * 5
    elif is_ratio_reward:
        if cover_add <= 0:
            coverage_reward = cover_add * 10
        else:
            remain = 1.0 - (current_coverage - cover_add)
            coverage_reward = (cover_add / remain) * 10
    else:
        coverage_reward = cover_add * 10

    normalized_travel_time = travel_time * 10 / max_travel_time if max_travel_time > 0 else 0.0
    normalized_delta_v = delta_v * 10 / max_delta_v if max_delta_v > 0 else 0.0
    
    time_penalty = time_cost_weight * normalized_travel_time
    fuel_penalty = delta_v_weight * normalized_delta_v
    
    return float(coverage_reward - time_penalty - fuel_penalty - collision_penalty)
