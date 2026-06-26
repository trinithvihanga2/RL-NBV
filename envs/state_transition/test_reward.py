import pytest
from envs.state_transition.reward import calculate_continuous_reward

def test_calculate_continuous_reward_basic():
    reward = calculate_continuous_reward(
        cover_add=0.1,
        current_coverage=0.5,
        step_cnt=5,
        travel_time=1.0,
        max_travel_time=10.0,
        delta_v=1.0,
        max_delta_v=10.0,
        collision_penalty=0.0
    )
    # basic reward = cover_add * 10 = 1.0
    # normalized_travel_time = 1.0 * 10 / 10 = 1.0, time penalty = 1.0
    # normalized_delta_v = 1.0 * 10 / 10 = 1.0, fuel penalty = 1.0
    # reward = 1.0 - 1.0 - 1.0 = -1.0
    assert pytest.approx(reward) == -1.0

def test_calculate_continuous_reward_ratio():
    reward = calculate_continuous_reward(
        cover_add=0.1,
        current_coverage=0.9,
        step_cnt=5,
        travel_time=0.0,
        max_travel_time=10.0,
        delta_v=0.0,
        max_delta_v=10.0,
        collision_penalty=0.0,
        is_ratio_reward=True
    )
    # remain = 1.0 - (0.9 - 0.1) = 0.2
    # reward = (0.1 / 0.2) * 10 = 5.0
    assert pytest.approx(reward) == 5.0

def test_calculate_continuous_reward_with_cur_coverage():
    reward = calculate_continuous_reward(
        cover_add=0.1,
        current_coverage=0.9,
        step_cnt=5,
        travel_time=0.0,
        max_travel_time=10.0,
        delta_v=0.0,
        max_delta_v=10.0,
        collision_penalty=0.0,
        is_reward_with_cur_coverage=True
    )
    # remain = 1.0 - (0.9 - 0.1) = 0.2
    # reward = (0.1 / 0.2) * 5 + 0.1 * 5 = 2.5 + 0.5 = 3.0
    assert pytest.approx(reward) == 3.0

def test_calculate_continuous_reward_collision():
    reward = calculate_continuous_reward(
        cover_add=0.0,
        current_coverage=0.5,
        step_cnt=1,
        travel_time=0.0,
        max_travel_time=10.0,
        delta_v=0.0,
        max_delta_v=10.0,
        collision_penalty=25.0
    )
    assert pytest.approx(reward) == -25.0
