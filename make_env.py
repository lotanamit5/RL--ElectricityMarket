import os
import pandas as pd
import gymnasium as gym

from gymnasium.wrappers import RescaleAction
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv

from electricity_market import ElectricityMarketEnv

class CustomMonitorWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """
    def __init__(self, env, log_dir='logs', filename='MyMonitor'):
        # Call the parent constructor, so we can access self.env later
        super().__init__(env)
        os.makedirs(log_dir, exist_ok=True)
        self.file_path = os.path.join(log_dir, f"{filename}.csv")
        self._columns_order = ['episode', 'step', 'action', 'soc', 'demand', 'price', 'reward']
        # self._step = 0
        self._episode = 0
        self._shift_index = 0
        
        self.actions = []
        self.socs = []
        self.demands = []
        self.prices = []
        self.rewards = []
        
        with open(self.file_path, 'w') as f:
            f.write(',' + ','.join(self._columns_order) + '\n')

    def _write_episode_logs(self):
        df = pd.DataFrame({
            'action': self.actions,
            'soc': self.socs,
            'demand': self.demands,
            'price': self.prices,
            'reward': self.rewards
        })
        df['step'] = df.index + 1
        df['episode'] = self._episode
        df.index += self._shift_index
        df = df[self._columns_order]
        df.to_csv(self.file_path, mode='a', header=False)
        self._shift_index += len(df)
        
    def reset(self, **kwargs):
        """
        Reset the environment
        """
        
        # Write episode logs to file
        self._write_episode_logs()
        
        # Reset episode logs
        self.actions = []
        self.socs = []
        self.demands = []
        self.prices = []
        self.rewards = []
        self._episode += 1
        
        # reset environment
        obs, info = self.env.reset(**kwargs)
        self.socs.append(obs[0])
        self.demands.append(obs[1])
        self.prices.append(obs[2])

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.actions.append(action)
        self.rewards.append(reward)
        if not (terminated or truncated):
            self.socs.append(obs[0])
            self.demands.append(obs[1])
            self.prices.append(obs[2])
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        self._write_episode_logs()
        self.env.close()

def make_env(log_dir='./logs',
             env_kwargs=None,
             my_monitor_kwargs=None
             ):
    # Create log dir
    os.makedirs(log_dir, exist_ok=True)
    env_kwargs = env_kwargs or {}
    my_monitor_kwargs = my_monitor_kwargs or {}

    env = ElectricityMarketEnv(**env_kwargs)
    env = CustomMonitorWrapper(env, **my_monitor_kwargs)
    env = RescaleAction(env, -1, 1)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    env = VecFrameStack(env, n_stack=4)

    return env
