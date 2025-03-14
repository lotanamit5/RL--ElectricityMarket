import os
import pandas as pd
import gymnasium as gym

from datetime import datetime
from torch.distributions import Normal
from gymnasium.wrappers import RescaleAction
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv

from electricity_market import ElectricityMarketEnv

class NormalNoiseWrapper():
    def __init__(self, func: callable,
                 loc=0, scale=1):
        assert callable(func)
        self.func = func
        self.noise = Normal(loc=loc, scale=scale)
    
    def __call__(self, *args, **kwds):
        val =  self.func(*args) + self.noise.sample().item()
        if val < 0:
            val = 1e-3
        return val

class CustomMonitorWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """
    def _add_row(self, episode, step, action, soc, demand, price, reward, done):
        row = pd.DataFrame({'episode': episode,
                            'step': step,
                            'action': action,
                            'soc': soc,
                            'demand': demand,
                            'price': price,
                            'reward': reward,
                            'done': done}, index=[0])
        self.data = pd.concat([self.data, row], ignore_index=True)
        

    def __init__(self, env, log_dir='logs'):
        # Call the parent constructor, so we can access self.env later
        super().__init__(env)
        os.makedirs(log_dir, exist_ok=True)
        self.file_path = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv")
        self._step = 0
        self._episode = 0
        
        self.data = pd.DataFrame(columns=['episode', 'step', 'action', 'soc', 'demand', 'price', 'reward', 'done'])
        self.data.to_csv(self.file_path, mode='w', header=True)

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        obs, info = self.env.reset(**kwargs)
        self._step = 0
        self._episode += 1
        self.data.to_csv(self.file_path, mode='a', header=False, index=False)
        self.data = pd.DataFrame({'episode': self._episode,
                                  'step': self._step,
                                  'action': None,
                                  'soc': obs[0],
                                  'demand': obs[1],
                                  'price': obs[2],
                                  'reward': None,
                                  'done': False
                                }, index=[0])
        return obs, info

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) observation, reward, is this a final state (episode finished),
        is the max number of steps reached (episode finished artificially), additional informations
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step += 1
        self._add_row(self._episode, self._step, action, obs[0], obs[1], obs[2], reward, terminated or truncated)
        return obs, reward, terminated, truncated, info
    
    def close(self):
        self.data.to_csv(self.file_path, mode='a', header=False)
        self.env.close()

def make_env(log_dir='./logs', record=True):
    # Create log dir
    os.makedirs(log_dir, exist_ok=True)

    env = ElectricityMarketEnv(noisy=True)
    if record:
        env = CustomMonitorWrapper(env, log_dir=log_dir)
    env = Monitor(env, log_dir)
    env = RescaleAction(env, -1, 1)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    env = VecFrameStack(env, n_stack=4)

    return env
