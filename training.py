from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from utils import make_env


def train_ppo(
    timesteps: int = 10_000,
    model_dir: str = "./models",
    log_dir: str = "./logs",
):
    env = make_env()
    eval_env = make_env()
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path=model_dir, log_path=log_dir,
                                 eval_freq=1000, n_eval_episodes=1,
                                 deterministic=True)
    model = PPO(policy="MlpPolicy", env=env)
    model.learn(total_timesteps=timesteps, callback=[eval_callback])
    model.save(f"{model_dir}/{model.__class__.__name__}")