import numpy as np
import gymnasium as gym
from gymnasium import spaces

from defaults import (demand_default_fn, price_default_fn, 
                      DEFAULT_BATTERY_CAPACITY, HOURS_A_YEAR, 
                      DEMAND_STD, PRICE_STD)
from utils import NormalNoiseWrapper

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
    - `demand_fn`, `price_fn` (callable, optional): Functions modeling demand and price.
    - `render_mode` (str): One of ["console", "human", "debug", "none"].
    - `seed` (int, optional): Random seed for reproducibility.
    - `noisy` (bool): Whether to add noise to demand and price functions.

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
    
    def __init__(self,
                 capacity: float = DEFAULT_BATTERY_CAPACITY,
                 horizon: int = HOURS_A_YEAR-1,
                 demand_fn: callable = demand_default_fn,
                 price_fn: callable = price_default_fn,
                 render_mode: str = "none",
                 noisy=True):
        super().__init__()
        self.render_mode = render_mode
        
        assert isinstance(capacity, (int, float)), f"The capacity should be a number, got {type(capacity)}"
        assert capacity > 0, f"The capacity should be greater than 0, got {capacity}"
        assert isinstance(horizon, int), f"The horizon should be an integer, got {type(horizon)}"
        assert horizon > 0, f"The horizon should be greater than 0, got {horizon}"
        assert callable(demand_fn), f"The demand function should be callable, got {type(demand_fn)}"
        assert callable(price_fn), f"The price function should be callable, got {type(price_fn)}"
        assert render_mode in self._render_modes, f"Only {', '.join(self._render_modes)} render mode/s are supported, got {render_mode}"
        
        self._noisy = noisy
        self._capacity = capacity
        self._timestep = 0
        self._state_of_charge = 0.
        self._horizon = horizon
        
        self._demand_fn = demand_fn #if not noisy else NormalNoiseWrapper(demand_fn, scale=DEMAND_STD)
        self._price_fn = price_fn #if not noisy else NormalNoiseWrapper(price_fn, scale=PRICE_STD)
        self.demand = self._demand_fn if not self._noisy else NormalNoiseWrapper(self._demand_fn, scale=DEMAND_STD)
        self.price = self._price_fn if not self._noisy else NormalNoiseWrapper(self._price_fn, scale=PRICE_STD)
        
        self._demand_from_grid = []
        
        self.action_space = spaces.Box(low=-self._capacity, high=self._capacity, shape=(), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0., 0., 0.], dtype=np.float32), 
            high=np.array([self._capacity, np.inf, np.inf], dtype=np.float32),
            shape=(3,), dtype=np.float32
        )
        self.render()
    
    def _get_obs(self):
        return np.asarray([self._state_of_charge, self.demand(self._timestep), self.price(self._timestep)], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._state_of_charge = 0.
        self._timestep = 0
        self._demand_from_grid = []
        self.render()
        return self._get_obs(), {}  # empty info dict

    def step(self, action: float):
        if not -self._capacity <= action <= self._capacity:
            raise ValueError(
                f"Action must be between -{self._capacity} and {self._capacity}"
            )
        if self._timestep >= self._horizon:
            raise ValueError("Episode is terminated, please reset the environment")
        
        # print(f"[Env] Time: {self._timestep}")
        # print(f"[Env] Action: {action}")
        demand = self.demand(self._timestep)
        # print(f"[Env] Demand: {demand}")
        price = self.price(self._timestep)
        
        if action < 0: # discharge
            discharge = min(self._state_of_charge, -action) # can not discharge more than SoC
            # print(f"[Env] Discharge: {discharge}")
            self._state_of_charge -= discharge
            
            # discharge - demand > 0 -> we have leftovers to sell
            # discharge - demand < 0 -> we need to buy extra units to satisfy demand
            self._demand_from_grid.append(max(0, demand - discharge))
            reward = (discharge - demand) * price
       
        else: # charge
            charge = min(self._capacity - self._state_of_charge, action) # can not charge more than the capacity
            # print(f"[Env] Charge: {charge}")
            
            self._state_of_charge += charge
            
            self._demand_from_grid.append(charge + demand)
            reward = -(charge + demand) * price
        # print(f"[Env] Demand from grid: {self._demand_from_grid[-1]}")
        
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
            print(f"Demand: {self.demand(self._timestep)}")
            print(f"Price: {self.price(self._timestep)}")

    def close(self):
        pass
    
    def _render_human(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(2, 6))
        soc, demand, price = self._get_obs()
        plt.bar([0], [soc], color='g', alpha=0.5)
        plt.ylim(0, self._capacity)
        plt.yticks(np.array([0, 0.25, 0.5, 0.75, 1])*self._capacity, labels=['0%', '25%', '50%', '75%', '100%'])
        plt.xticks([])
        plt.title(f"SoC: {soc:.2f}")
        plt.show()

if __name__ == "__main__":
    num_steps = 100
    env = ElectricityMarketEnv()
    print(env.action_space.shape)
    # # If the environment don't follow the interface, an error will be thrown
    # # check_env(env, warn=True, skip_render_check=True)
    # obs, _ = env.reset()
    # terminated, truncated = False, False
    # total_reward = 0
    # SoCs, demands, prices = [], [], []

    # for idx in range(num_steps):
    #     action = env.action_space.sample()
    #     next_obs, reward, terminated, truncated, _ = env.step(action)
    #     SoC, demand, price = next_obs
    #     SoCs.append(SoC)
    #     demands.append(demand)
    #     prices.append(price)
    #     total_reward += reward
    #     obs = next_obs
    #     if terminated or truncated:
    #         break
    