"""
Evaluation script for the Dragon spacecraft ISS docking simulator.

Loads a trained PPO model and runs it in the docking environment, reporting
statistics about episode outcomes (success rate, mean reward, etc.).

Usage
-----
    python evaluate.py                            # evaluate default model
    python evaluate.py --model models/ppo_docking # specify model path
    python evaluate.py --episodes 20              # run 20 evaluation episodes
"""

import argparse

import numpy as np
from stable_baselines3 import PPO

from environment import IssDockingEnv


def evaluate(model_path: str, n_episodes: int) -> None:
    """Run evaluation episodes and print a summary.

    Args:
        model_path: Path to the saved model (with or without `.zip`).
        n_episodes: Number of episodes to evaluate.
    """
    env = IssDockingEnv()
    model = PPO.load(model_path, env=env)

    episode_rewards = []
    final_distances = []
    successes = 0

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        episode_rewards.append(total_reward)
        final_distances.append(info["distance"])

        if info["success"]:
            successes += 1

        print(
            f"Episode {episode + 1:3d}/{n_episodes}: "
            f"reward={total_reward:8.2f}  "
            f"distance={info['distance']:6.2f} m  "
            f"speed={info['speed']:.2f} m/s  "
            f"steps={info['steps']}"
        )

    print("\n--- Evaluation Summary ---")
    print(f"Episodes        : {n_episodes}")
    print(f"Success rate    : {successes / n_episodes * 100:.1f}%")
    print(f"Mean reward     : {np.mean(episode_rewards):.2f}")
    print(f"Mean final dist : {np.mean(final_distances):.2f} m")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO agent on the ISS docking simulator."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/ppo_docking",
        help="Path to the trained model file (default: models/ppo_docking).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes to run (default: 10).",
    )
    args = parser.parse_args()

    evaluate(model_path=args.model, n_episodes=args.episodes)


if __name__ == "__main__":
    main()
