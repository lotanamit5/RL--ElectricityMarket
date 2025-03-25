# Electricity Market Environment

This project implements a reinforcement learning environment for modeling an electricity market where an agent manages a battery storage system to maximize profit while meeting household demand. The environment is built using the `gymnasium` library.

## Environment Description

### State Space
- **State of Charge (SoC):** Battery charge in [0, capacity].
- **Demand (Dt):** Periodic electricity demand with stochastic noise.
- **Price (Pt):** Periodic market price with stochastic noise.

### Action Space
- A single continuous value in [-capacity, capacity], representing charge (positive) or discharge (negative).

### Reward Function
- **Discharge (action < 0):** First meets demand; surplus energy is sold to the grid at Pt.
- **Charge (action > 0):** Battery is charged, incurring a cost based on (charge + demand) * Pt.

### Episode Termination
- Fixed horizon (number of timesteps).
- An attempt to step beyond the horizon raises an error.

### Parameters
- `capacity` (float): Battery capacity.
- `horizon` (int): Number of timesteps per episode.
- `demand_fn`, `price_fn` (callable, optional): Functions modeling demand and price.
- `render_mode` (str): One of ["console", "debug", "none"].
- `seed` (int, optional): Random seed for reproducibility.
- `noisy` (bool): Whether to add noise to demand and price functions.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/lotanamit5/RL--ElectricityMarket.git
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To use the environment, you can create an instance of `ElectricityMarketEnv`, or preferably, create a wrapped instance using `make_env`, and interact with it using the `step` and `reset` methods.

Example:
```python
from make_env import make_env
log_dir = './logs'
env = make_env(log_dir=log_dir)
obs, _ = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Replace with your action selection logic
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break