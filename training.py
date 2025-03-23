import os
import numpy as np
from stable_baselines3 import PPO, SAC, A2C, DDPG, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from make_env import make_env
import argparse

def make_model(model_name, env):
    model_name = model_name.upper()
    if model_name == "PPO":
        return PPO(policy="MlpPolicy", env=env)
    elif model_name == "SAC":
        return SAC(policy="MlpPolicy", env=env)
    elif model_name == "A2C":
        return A2C(policy="MlpPolicy", env=env)
    elif model_name == "DDPG":
        action_noise = NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(1))
        return DDPG(policy="MlpPolicy", env=env, action_noise=action_noise)
    elif model_name == "TD3":
        action_noise = NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(1))
        return TD3(policy="MlpPolicy", env=env)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train(
    model_name: str,
    timesteps: int = 100_000,
    models_dir: str = "./models",
    logs_dir: str = "./logs",
):
    model_dir = os.path.join(models_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    log_dir = os.path.join(logs_dir, model_name)
    os.makedirs(logs_dir, exist_ok=True)
    
    env = make_env(log_dir=log_dir)
    eval_env = make_env(log_dir=log_dir)
    eval_callback = EvalCallback(eval_env, deterministic=True,
                                 best_model_save_path=model_dir, log_path=log_dir,
                                 eval_freq=1000, n_eval_episodes=1)
    
    model = make_model(model_name, env)
    model.learn(total_timesteps=timesteps, callback=[eval_callback])
    model.save(os.path.join(model_dir, "final_model"))
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reinforcement learning model.")
    parser.add_argument("--model", type=str, required=True, help="The name of the model to train (PPO, SAC, A2C, DDPG, TD3).")
    parser.add_argument("--timesteps", type=int, default=100_000, help="The number of timesteps to train the model.")
    parser.add_argument("--model_dir", type=str, default="./models", help="The directory to save the trained model.")
    parser.add_argument("--log_dir", type=str, default="./logs", help="The directory to save the training logs.")
    args = parser.parse_args()

    train(model_name=args.model, timesteps=args.timesteps, model_dir=args.model_dir, log_dir=args.log_dir)