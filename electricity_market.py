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
    A reinforcement learning environment modeling an electricity market where an agent 
    manages a battery storage system to maximize profit while meeting household demand.

    ## State Space:
    - **State of Charge (SoC):** Battery charge in [0, capacity].
    - **Demand (Dt):** Periodic electricity demand with stochastic noise.
    - **Price (Pt):** Periodic market price with stochastic noise.

    ## Action Space:
    - A single continuous value in [-capacity, capacity], representing charge (positive) or discharge (negative).

    ## Reward Function:
    - **Discharge (action < 0):** First meets demand; surplus energy is sold to the grid at Pt.
    - **Charge (action > 0):** Battery is charged, incurring a cost based on (charge + demand) * Pt.

    ## Episode Termination:
    - Fixed horizon (number of timesteps).
    - An attempt to step beyond the horizon raises an error.

    ## Parameters:
    - `capacity` (float): Battery capacity.
    - `horizon` (int): Number of timesteps per episode.
    - `demand_fn`, `price_fn` (PeriodicFunction, optional): Functions modeling demand and price.
    - `render_mode` (str): One of ["console", "human", "debug", "none].
    - `seed` (int, optional): Random seed for reproducibility.

    ## Example:
        ```python
        env = ElectricityMarketEnv(capacity=100, horizon=100, seed=42)
        obs, _ = env.reset()
        action = env.action_space.sample()  # Sampled action
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()  # Display state
        ```
    """
    
    _render_modes = ["console", "human", "debug", "none"]
    
    def __init__(self, capacity: float,
                 horizon: int = 100,
                 demand_fn: PeriodicFunction = None,
                 price_fn: PeriodicFunction = None,
                 render_mode: str = "none",
                 seed: int = None):
        self.render_mode = render_mode
        
        assert isinstance(capacity, (int, float)), f"The capacity should be a number, got {type(capacity)}"
        assert capacity > 0, f"The capacity should be greater than 0, got {capacity}"
        assert isinstance(horizon, int), f"The horizon should be an integer, got {type(horizon)}"
        assert horizon > 0, f"The horizon should be greater than 0, got {horizon}"
        assert render_mode in self._render_modes, f"Only {', '.join(self._render_modes)} render mode/s are supported, got {render_mode}"
        assert seed is None or isinstance(seed, int), f"The seed should be an integer, got {type(seed)}"
        assert demand_fn is None or isinstance(demand_fn, PeriodicFunction), f"The demand function should inherit from PeriodicFunction, got {type(demand_fn)}"
        assert price_fn is None or isinstance(price_fn, PeriodicFunction), f"The price function should inherit from PeriodicFunction, got {type(price_fn)}"
        
        demand_fn = demand_fn or PeriodicTwoGaussianSum.from_seed(horizon, seed=seed)
        price_fn = price_fn or PeriodicTwoGaussianSum.from_seed(horizon, seed=seed+1 if seed else None)

        # Initialize the state of charge
        self._capacity = capacity
        self._timestep = 0
        self._state_of_charge = 0.
        self._demand = demand_fn
        self._price = price_fn
        self._horizon = horizon
        
        # TODO: Normalize to [-1, 1]
        self.action_space = spaces.Box(low=-self._capacity, high=self._capacity, shape=(), dtype=np.float32)
        # TODO: find a way to make the space in [0,1] instead of [0, inf), define P_max and D_max
        self.observation_space = spaces.Box(
            low=np.array([0., 0., 0.]), 
            high=np.array([self._capacity, np.inf, np.inf]),
            shape=(3,), dtype=np.float32
        )
        self.render()
    
    def _get_obs(self):
        return np.asarray([self._state_of_charge, self._demand[self._timestep], self._price[self._timestep]], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._state_of_charge = 0.
        self._timestep = 0
        self.render()
        return self._get_obs(), {}  # empty info dict

    def step(self, action: float):
        if not -self._capacity <= action <= self._capacity:
            raise ValueError(
                f"Action must be between -{self._capacity} and {self._capacity}"
            )
        if self._timestep >= self._horizon:
            raise ValueError("Episode is terminated, please reset the environment")
        
        demand = self._demand[self._timestep]
        price = self._price[self._timestep]
        
        if action < 0: # discharge
            discharge = min(self._state_of_charge, -action) # can not discharge more than SoC
            
            self._state_of_charge -= discharge
            
            # discharge - demand > 0 -> we have leftovers to sell
            # discharge - demand < 0 -> we need to buy extra units to satisfy demand
            reward = (discharge - demand) * price
       
        else: # charge
            charge = min(self._capacity - self._state_of_charge, action) # can not charge more than the capacity
            
            self._state_of_charge += charge
            
            reward = -(charge + demand) * price
        
        self.render()
        
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
        if self.render_mode == 'none':
            return
        if self.render_mode == 'human':
            self._render_human()
        elif self.render_mode == 'console':
            print(f"State of Charge: {self._state_of_charge}")
            print(f"Demand: {self._demand[self._timestep]}")
            print(f"Price: {self._price[self._timestep]}")

    def close(self):
        pass
    
    def _render_human(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        obs = self._get_obs()
        
        plt.bar(obs.keys(), obs.values())
        plt.show()
        

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
        SoC, demand, price = next_obs
        SoCs.append(SoC)
        demands.append(demand)
        prices.append(price)
        total_reward += reward
        obs = next_obs
        if terminated or truncated:
            break
    