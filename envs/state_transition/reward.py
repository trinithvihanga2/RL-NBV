"""Reward calculation for state transition.

Implements reward policy based on action, visit status, and coverage gain.

Level 1 reward policy:
- revisit/invalid selected view: -5
- newly selected valid view: Coverage(P_new^i, P_m) = newly_covered_ratio
"""

from __future__ import annotations

from typing import NamedTuple
from collections.abc import Mapping

import numpy as np


class RewardResult(NamedTuple):
    """Result of reward calculation.
    
    Attributes:
        reward: Float scalar reward value.
        breakdown: Optional dict with reward components for debugging/analysis.
    """
    
    reward: float
    breakdown: dict[str, float]


def _as_int_scalar(name: str, value: object) -> int:
    """Normalize input to integer scalar.
    
    Args:
        name: Parameter name for error messages.
        value: Input value.
    
    Returns:
        Integer value.
    
    Raises:
        TypeError: If not convertible to integer.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be int, not bool")
    try:
        scalar = int(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be convertible to int: {e}")
    return scalar


def _as_float_scalar(name: str, value: object) -> float:
    """Normalize input to float scalar.
    
    Args:
        name: Parameter name for error messages.
        value: Input value.
    
    Returns:
        Float value.
    
    Raises:
        TypeError: If not convertible to float.
        ValueError: If not finite.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not bool")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be convertible to float: {e}")
    
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {scalar}")
    
    return scalar


def _as_bool_array(name: str, value: object) -> np.ndarray:
    """Normalize input to boolean array.
    
    Args:
        name: Parameter name for error messages.
        value: Input value (array-like).
    
    Returns:
        Boolean numpy array.
    
    Raises:
        ValueError: If input cannot be normalized.
    """
    arr = np.asarray(value, dtype=bool)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D array, got shape {arr.shape}")
    return arr


def calculate_reward(
    action: int,
    views: object,
    newly_covered_ratio: float,
    travel_time: float | None = None,
    reward_config: Mapping[str, object] | None = None,
) -> RewardResult:
    """Calculate reward based on action and coverage gain.
    
    Level 1 reward policy:
    - If action revisits an already selected view: reward = -5
    - If action selects a valid new view: reward = newly_covered_ratio
    
    Args:
        action: Selected view/action index (int).
        views: Boolean array indicating which views have been visited.
               Shape (n_views,), where views[action] indicates if action was visited.
        newly_covered_ratio: Coverage ratio for newly covered points (0.0 to 1.0).
                            This is Coverage(P_new^i, P_m) from the reward policy.
        travel_time: Optional travel time (seconds). Accepted for future extension
                    but not used in current reward policy (no travel-time penalty yet).
        reward_config: Optional configuration dict with reward coefficients.
                      Supported keys:
                      - revisit_penalty (float, default -5.0): penalty for revisiting
                      - coverage_coeff (float, default 1.0): multiplier for coverage gain
                      - shaping_terms (dict, optional): future shaping terms
    
    Returns:
        RewardResult with:
        - reward: Float scalar reward value
        - breakdown: Dict with reward components for debugging:
            - is_revisit (bool)
            - revisit_penalty (float, if applicable)
            - coverage_reward (float, if applicable)
            - total (float)
    
    Raises:
        TypeError: If inputs have wrong type.
        ValueError: If inputs are out of valid range.
        IndexError: If action is out of range for views array.
    """
    
    # Validate action
    action = _as_int_scalar("action", action)
    
    # Validate views
    views_arr = _as_bool_array("views", views)
    if action < 0 or action >= views_arr.shape[0]:
        raise IndexError(
            f"action {action} out of range for views array with {views_arr.shape[0]} views"
        )
    
    # Validate newly_covered_ratio
    newly_covered_ratio = _as_float_scalar("newly_covered_ratio", newly_covered_ratio)
    if not (0.0 <= newly_covered_ratio <= 1.0):
        raise ValueError(
            f"newly_covered_ratio must be in [0.0, 1.0], got {newly_covered_ratio}"
        )
    
    # Validate travel_time if provided (not used in reward, but accepted for future use)
    if travel_time is not None:
        travel_time = _as_float_scalar("travel_time", travel_time)
        if travel_time < 0.0:
            raise ValueError(f"travel_time must be >= 0, got {travel_time}")
    
    # Extract config parameters
    if reward_config is None:
        reward_config = {}
    
    if not isinstance(reward_config, Mapping):
        raise TypeError("reward_config must be a mapping")
    
    revisit_penalty = _as_float_scalar(
        "reward_config['revisit_penalty']",
        reward_config.get("revisit_penalty", -5.0)
    )
    coverage_coeff = _as_float_scalar(
        "reward_config['coverage_coeff']",
        reward_config.get("coverage_coeff", 1.0)
    )
    
    # Check if action is a revisit
    is_revisit = bool(views_arr[action])
    
    # Calculate reward based on policy
    breakdown = {}
    
    if is_revisit:
        # Revisit penalty
        reward = revisit_penalty
        breakdown["is_revisit"] = True
        breakdown["revisit_penalty"] = revisit_penalty
        breakdown["coverage_reward"] = 0.0
    else:
        # Coverage-based reward for new view
        coverage_reward = coverage_coeff * newly_covered_ratio
        reward = coverage_reward
        breakdown["is_revisit"] = False
        breakdown["revisit_penalty"] = 0.0
        breakdown["coverage_reward"] = coverage_reward
    
    # Apply any optional shaping terms (future extension)
    if "shaping_terms" in reward_config:
        shaping_terms = reward_config["shaping_terms"]
        if isinstance(shaping_terms, Mapping):
            for term_name, term_value in shaping_terms.items():
                term_value = _as_float_scalar(f"shaping_terms['{term_name}']", term_value)
                reward += term_value
                breakdown[f"shaping_{term_name}"] = term_value
    
    breakdown["total"] = reward
    
    return RewardResult(reward=reward, breakdown=breakdown)

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
