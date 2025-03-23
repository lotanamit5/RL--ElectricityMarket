from typing import List
import pandas as pd
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy

from make_env import make_env
from electricity_market import ElectricityMarketEnv
from defaults import HOURS_A_YEAR
class Market:
    def __init__(self, ratio=0.1):
        self.ratio = ratio
        self.demand = 0
        self.grid_capacity = 2
        self.env = None
        self._prices = []
    
    def market_price(self, t):
        if t == 0:
            # print("[Market] Initial price 0.5")
            price = 0.5 # dummy value
        else:
            prev_grid_use = self.env._demand_from_grid[t-1]
            prev_demand = self.env.demand(t-1)
            total_buy_from_grid = self.ratio * prev_grid_use + (1 - self.ratio) * prev_demand
            price = total_buy_from_grid / self.grid_capacity
        
        if len(self._prices) == t:
            self._prices.append(float(price))
        return price
    
    def update_env(self, env):
        self.env = env

def run_naive_eval(
    model: BaseAlgorithm
):
    log_dir = f'./logs/{model.__class__.__name__}'
    env = make_env(eval=True, stack=type(model)!=SAC,
                   log_dir=log_dir,
                   my_monitor_kwargs={"log_dir": log_dir ,"filename": "base_env_eval"})
    evaluate_policy(model, env, deterministic=True)

def run_simulation(
    model: BaseAlgorithm,
    ratio: float,
):
    market = Market(ratio=ratio)
    log_dir = f'./logs/{model.__class__.__name__}/{ratio}'
    env = make_env(eval=True, stack=type(model)!=SAC,
                   log_dir=log_dir,
                   env_kwargs={"price_fn": market.market_price, "noisy": False},
                   my_monitor_kwargs={"log_dir": log_dir ,"filename": "true_market_simulation"})
    base_env = env
    while not isinstance(base_env, ElectricityMarketEnv):
        if hasattr(base_env, "env"):
            base_env = base_env.env
        elif hasattr(base_env, "envs"):
            base_env = base_env.envs[0]
    market.update_env(base_env)
    
    evaluate_policy(model, env, deterministic=True)

if __name__ == "__main__":
    a2c = A2C.load("models/a2c/best_model.zip")
    ppo = PPO.load("models/ppo/best_model.zip")
    sac = SAC.load("models/sac/best_model.zip")
    models = [a2c, ppo, sac]
    ratios = [0.1, 0.5, 0.9]
    for model in models:
        run_naive_eval(model)
        for ratio in ratios:
            print('*'*10 + f"Running simulation for {model.__class__.__name__} with ratio {ratio}" + '*'*10)
            run_simulation(model, ratio)
    
            