import numpy as np
import gymnasium as gym
from gymnasium import spaces

from periodic import PeriodicFunction, PeriodicTwoGaussianSum

# Define the Environment: The environment models the electricity market. Implement the following components:
# • State Variables: Define the state space including SoC (State of Charge), Dt
# (electricity demand), and Pt (electricity price).
# • Dynamics: Model the stochastic evolution of the market price (Pt) and demand
# (Dt). The market price and demand should both be periodic functions with two
# ”peaks”. Both functions should be noisy, meaning that a random noise should
# be added to the function value at each timestep. An example of the demand
# function could be a combination of two normal distributions. For example: 
# $$ f(x) = 100 \cdot e^\frac{−(x−0.4)^2}{2 \cdot (0.05)^2} + 120 \cdot e^\frac{−(x−0.7)^2}{2 \cdot (0.1)^2}
# • Reward Function: Design a reward function to maximize profits while meeting
# household demand

class ElectricityMarketEnv(gym.Env):
    """
    #TODO
    """
    def __init__(self, capacity: float,
                 horizon: int = 100,
                 demand_fn: PeriodicFunction = None,
                 price_fn: PeriodicFunction = None,
                 render_mode: str = "console",
                 seed: int = None):
        self.render_mode = render_mode
        
        assert isinstance(capacity, (int, float)), "The capacity should be a number"
        assert capacity > 0, "The capacity should be greater than 0"
        assert isinstance(horizon, int), "The horizon should be an integer"
        assert horizon > 0, "The horizon should be greater than 0"
        assert render_mode in ["console"], "Only console render mode is supported"
        assert seed is None or isinstance(seed, int), "The seed should be an integer"
        assert demand_fn is None or isinstance(demand_fn, PeriodicFunction), "The demand function should inherit from PeriodicFunction"
        assert price_fn is None or isinstance(price_fn, PeriodicFunction), "The price function should inherit from PeriodicFunction"
        demand_fn = demand_fn or PeriodicTwoGaussianSum.from_seed(horizon, seed=seed)
        price_fn = price_fn or PeriodicTwoGaussianSum.from_seed(horizon, seed=seed+1 if seed else None)

        # Initialize the state of charge
        self._capacity = capacity
        self._timestep = 0
        self._state_of_charge = 0
        self._demand = None
        self._price = None
        self._horizon = horizon
        
        self.action_space = spaces.Box(low=-self._capacity, high=self._capacity, shape=(), dtype=float)
        # TODO: find a way to make the space in [0,1] instead of [0, inf)
        self.observation_space = spaces.Dict({
            "state_of_charge": spaces.Box(low=0, high=self._capacity, shape=(1,), dtype=np.float32),
            "demand": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "price": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })
    
    def _get_obs(self):
        return {
            "state_of_charge": self._state_of_charge,
            "demand": self._demand[self._timestep],
            "price": self._price[self._timestep]
        }

    def reset(self, seed=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed)
        # Initialize the agent at the right of the grid
        self._state_of_charge = 0
        self._timestep = 0
        self._demand = PeriodicTwoGaussianSum.from_seed(self._horizon, seed=seed)
        self._price = PeriodicTwoGaussianSum.from_seed(self._horizon, seed=seed+1 if seed else None)
        
        return self._get_obs(), {}  # empty info dict

    def step(self, action: float):
        if not -self._capacity <= action <= self._capacity:
            raise ValueError(
                f"Action must be between -{self._capacity} and {self._capacity}"
            )
        if self._timestep >= self._horizon:
            raise ValueError("Episode is terminated, please reset the environment")
        
        if action < 0: # discharge
            print("SoC", self._state_of_charge)
            print("action", action)
            discharge = max(self._state_of_charge, -action) # can not discharge more than SoC
            reward = (discharge - self._demand[self._timestep]) * self._price[self._timestep]
        
        else: # charge
            reward = 0
        
        # Update the state of charge
        self._state_of_charge = np.clip(self._state_of_charge + action, 0, self._capacity)

        # Update the timestep to return the next observation
        self._timestep += 1

        return (
            self._get_obs(),
            reward,
            False, # terminated
            self._timestep == self._horizon, # truncated
            {} # no info
        )

    def render(self):
        print(__name__)
        print(f"State of Charge: {self._state_of_charge}")
        print(f"Demand: {self._demand[self._timestep]}")
        print(f"Price: {self._price[self._timestep]}")

    def close(self):
        pass

if __name__ == "__main__":
    num_steps = 100
    env = ElectricityMarketEnv(capacity=100, horizon=num_steps)
    # If the environment don't follow the interface, an error will be thrown
    # check_env(env, warn=True, skip_render_check=True)
    obs, _ = env.reset()
    terminated, truncated = False, False
    total_reward = 0
    SoCs, demands, prices = [], [], []

    for idx in range(num_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        SoCs.append(next_obs["state_of_charge"])
        demands.append(next_obs["demand"])
        prices.append(next_obs["price"])
        total_reward += reward
        obs = next_obs
        if terminated or truncated:
            break
    