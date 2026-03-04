"""
Training script for the Dragon spacecraft ISS docking simulator.

Uses Proximal Policy Optimisation (PPO) from Stable-Baselines3 to train an
agent to autonomously dock the Dragon spacecraft with the ISS.

Usage
-----
    python train.py                      # train with default settings
    python train.py --timesteps 1000000  # train for 1 M steps
    python train.py --save-path models/ppo_docking
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import IssDockingEnv


def make_env():
    """Factory function for a monitored docking environment."""
    env = IssDockingEnv()
    env = Monitor(env)
    return env


def train(timesteps: int, save_path: str, n_envs: int) -> None:
    """Train a PPO agent on the ISS docking environment.

    Args:
        timesteps: Total number of environment steps to train for.
        save_path: Path (without extension) where the model will be saved.
        n_envs: Number of parallel environments to use during training.
    """
    # Validate the environment against the Gymnasium API before training
    check_env(IssDockingEnv(), warn=True)

    vec_env = DummyVecEnv([make_env] * n_envs)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        learning_rate=3e-4,
        tensorboard_log="./logs/",
    )

    print(f"Training PPO for {timesteps:,} timesteps …")
    model.learn(total_timesteps=timesteps, progress_bar=True)

    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    model.save(save_path)
    print(f"Model saved to '{save_path}.zip'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on the ISS docking simulator."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500 000).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/ppo_docking",
        help="Path to save the trained model (default: models/ppo_docking).",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4).",
    )
    args = parser.parse_args()

    train(
        timesteps=args.timesteps,
        save_path=args.save_path,
        n_envs=args.n_envs,
    )


if __name__ == "__main__":
    main()
