import numpy as np
import pytest
from electricity_market import ElectricityMarketEnv
from periodic import PeriodicFunction

class ConstantPeriodicFunction(PeriodicFunction):
    def __init__(self, period: int, value: float = 1):
        super().__init__(period)
        self._value = value
    def _from_timestep(self, t_mod):
        return t_mod
    def _evaluate(self, x):
        return self._value    
    
def _get_dummy_env(capacity=10, horizon=24, const_demand=1, const_price=1):
    return ElectricityMarketEnv(capacity=capacity, horizon=horizon,
                                demand_fn=ConstantPeriodicFunction(horizon, const_demand),
                                price_fn=ConstantPeriodicFunction(horizon, const_price),
                                render_mode="console")
    
# Initialization Validity Test
# Check that the environment initializes correctly with valid parameters (capacity, horizon, demand_fn, price_fn, render_mode, seed).
def test_env_initialization():
    env = _get_dummy_env(capacity=10, horizon=24)
    assert env.action_space.contains(0), "Action space should contain 0"
    assert env.observation_space.contains(np.zeros(shape=(3,), dtype=np.float32)), "Observation space should contain [0,0,0]"
    assert env.render_mode == "console", f"Render mode should be 'console', is {env.render_mode}"
    assert env._capacity == 10, f"Capacity should be 10, is {env._capacity}"
    assert env._horizon == 24, f"Horizon should be 24, is {env._horizon}"
    assert env._timestep == 0, f"Timestep should be 0, is {env._timestep}"
    assert isinstance(env._demand, PeriodicFunction), f"Demand should be a PeriodicFunction, is {type(env._demand)}"
    assert isinstance(env._price, PeriodicFunction), f"Price should be a PeriodicFunction, is {type(env._price)}"

# Reset Functionality Test
# Call reset and verify that the timestep is reset to 0, the state of charge is reset to 0, and the observation returned matches initial values.
def test_env_reset():
    env = _get_dummy_env()
    for _ in range(5):
        env.step(env.action_space.sample())
    obs, _ = env.reset()
    assert env._timestep == 0, f"Timestep should be 0, is {env._timestep}"
    assert env._state_of_charge == 0, f"State of charge should be 0, is {env._state_of_charge}"
    assert obs[0] == 0, f"State of charge in observation should be 0. is {obs[0]}"

# Valid Charging Action Test (Within Capacity)
# Set a state where SoC is below capacity, then issue a positive action (charging) that does not exceed (capacity - SoC).
# Check that SoC increases correctly and the reward is computed as -(charge + demand) × price.
def test_valid_charging_action():
    env = _get_dummy_env(capacity=10)
    env.reset()
    obs, reward, _, _, _ = env.step(5)
    assert env._state_of_charge == 5, f"State of charge should be 5, got {env._state_of_charge}"
    assert obs[0] == 5, f"State of charge in observation should be 5, got {obs[0]}"
    assert reward == -6, f"Reward should be -6, got {reward}"

# Overcharge Action Test (Exceeding Capacity)
# Set a state with some SoC, then apply a charging action that would exceed the battery capacity.
def test_overcharge_action():
    env = _get_dummy_env(capacity=10)
    env.reset()
    env.step(action=5)
    obs, reward, _, _, _ = env.step(action=10) # overcharging
    assert env._state_of_charge == 10, f"State of charge should be 10, got {env._state_of_charge}"
    assert obs[0] == 10, f"State of charge in observation should be 10, got {obs[0]}"
    assert reward == -6, f"Reward should be -6, got {reward}"


# Over Capacity Illegal Action Test
# Attempt to step with an action that would exceed the battery capacity and confirm that a ValueError is raised.
def test_out_of_bounds_action():
    env = _get_dummy_env(capacity=10)
    env.reset()
    with pytest.raises(ValueError):
        env.step(15)
    with pytest.raises(ValueError):
        env.step(-15)

# Valid Discharging Action Test (Sufficient SoC)
# Set a state with a given SoC and apply a discharging (negative) action that is fully supported by the current SoC.
# Verify that the discharge is as requested and reward is computed as (discharge - demand) × price.
# Reason: To confirm that discharging works correctly when enough energy is available.
def test_valid_discharging_action():
    env = _get_dummy_env(capacity=10)
    env.reset()
    env.step(action=5)
    obs, reward, _, _, _ = env.step(action=-3)
    assert env._state_of_charge == 2, f"State of charge should be 2, got {env._state_of_charge}"
    assert obs[0] == 2, f"State of charge in observation should be 2, got {obs[0]}"
    assert reward == (3 - 1) * 1, f"Reward should be 2, got {reward}"
    
# Overdischarge Action Test (Insufficient SoC)
# Set a state where the SoC is less than the absolute value of the discharging action. 
# Check that the actual discharge is limited to the current SoC and that the reward reflects this limitation.
def test_overdischarge_action():
    env = _get_dummy_env(capacity=10)
    env.reset()
    env.step(action=5)
    obs, reward, _, _, _ = env.step(action=-10) # overdischarging
    assert env._state_of_charge == 0, f"State of charge should be 0, got {env._state_of_charge}"
    assert obs[0] == 0, f"State of charge in observation should be 0, got {obs[0]}"
    assert reward == (5 - 1) * 1, f"Reward should be 4, got {reward}"

# Discharging Below Demand Test
# Simulate a discharging action where the amount discharged (limited by SoC) is less than the current demand.
# Verify that the reward is negative (since discharge - demand < 0) and computed correctly.
def test_discharging_below_demand():
    env = _get_dummy_env(capacity=10, const_demand=5)
    env.reset()
    env.step(action=10)
    obs, reward, _, _, _ = env.step(action=-2) # overdischarging
    assert env._state_of_charge == 8, f"State of charge should be 8, got {env._state_of_charge}"
    assert obs[0] == 8, f"State of charge in observation should be 8, got {obs[0]}"
    assert reward == (2 - 5) * 1, f"Reward should be -3, got {reward}"

# Charging Reward Computation Test
# For a charging action, verify that the reward is computed as -(charge + demand) × price, and that any charging amount is clipped to the available capacity space.
def test_charging_reward_computation():
    env = _get_dummy_env(capacity=10, const_demand=5)
    env.reset()
    obs, reward, _, _, _ = env.step(action=10)
    assert env._state_of_charge == 10, f"State of charge should be 10, got {env._state_of_charge}"
    assert obs[0] == 10, f"State of charge in observation should be 10, got {obs[0]}"
    assert reward == -(10 + 5) * 1, f"Reward should be -15, got {reward}"

# Episode Termination Test
# Run steps until the timestep equals the horizon and then try calling step again, 
# ensuring an error is raised or that the environment signals termination/truncation correctly.
def test_episode_termination():
    env = _get_dummy_env(horizon=3)
    env.reset()
    env.step(action=0)
    env.step(action=0)
    _, _, terminated, truncated, _ = env.step(action=0)
    assert not terminated, "Environment should not signal terminated"
    assert truncated, "Environment should signal truncated"
    with pytest.raises(ValueError):
        env.step(0)

# Observation Space Consistency Test
# After a step, check that the observation returned is a dictionary with keys "state_of_charge", "demand", and "price", 
# and that the values fall within their defined ranges.
def test_obs_space_consistency():
    env = _get_dummy_env()
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray), f"Observation should be a numpy array, got {type(obs)}"
    assert obs.shape == (3,), f"Observation shape should be (3,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Observation dtype should be np.float32, got {obs.dtype}"
    assert obs.all() >= 0, f"Observation values should be non-negative, got {obs}"

    obs, _, _, _, _ = env.step(0)
    assert obs.all() >= 0, f"Observation values should be non-negative, got {obs}"

# Periodic Function Integration Test
# For a series of timesteps, validate that the demand and price functions (instances of PeriodicFunction) return non-negative, 
# reasonable values (consistent with their design).
def test_periodic_function_integration():
    for seed in (int(seed) for seed in np.random.randint(1, 1000, size=10)):
        env = ElectricityMarketEnv(capacity=10, horizon=24, seed=seed)
        env.reset()
        for _ in range(24):
            obs, _, _, _, _ = env.step(0)
            assert obs[1] >= 0, f"Demand should be non-negative, got {obs[1]}"
            assert obs[2] >= 0, f"Price should be non-negative, got {obs[2]}"

def test_gymnasium_requirements():
    from gymnasium.utils.env_checker import check_env

    env = ElectricityMarketEnv(capacity=100)
    # If the environment don't follow the interface, an error will be thrown
    check_env(env, warn=True, skip_render_check=True)