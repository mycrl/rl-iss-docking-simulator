"""
Evaluation script for the SpaceX ISS Docking Simulator.

Loads a trained PPO model and runs it on the browser-based SpaceX ISS Docking
Simulator in deterministic mode, then prints per-episode and aggregate stats.
"""

import argparse
import logging

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments import EvalIssDockingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(
    model_path: str,
    n_episodes: int,
    launch_browser: bool,
    headless: bool,
) -> None:
    """Run the trained policy for `n_episodes` and print per-episode stats.

    Loads the environment normalization statistics and the trained PPO model,
    runs the policy deterministically, collects per-episode metrics and
    prints a short summary at the end.
    """
    env = EvalIssDockingEnv(launch_browser=launch_browser, headless=headless)
    stats_path = model_path + "_vec_normalize.pkl"
    vec_env = VecNormalize.load(stats_path, DummyVecEnv([lambda: env]))
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env)

    final_ranges: list[float] = []
    final_rates: list[float] = []
    episode_steps: list[int] = []
    episode_fuel_used: list[int] = []
    successes = 0

    for episode in range(n_episodes):
        obs = vec_env.reset()
        done = False
        info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True) # pyright: ignore[reportArgumentType]
            obs, _rewards, dones, infos = vec_env.step(action)
            done = dones[0]
            info = infos[0]

        final_ranges.append(float(info.get("range", 0.0)))
        final_rates.append(float(info.get("rate", 0.0)))
        episode_steps.append(int(info.get("steps", 0)))
        episode_fuel_used.append(int(info.get("fuel_used", 0)))
        if info.get("success", False):
            successes += 1

        print(
            f"Episode {episode + 1:3d}/{n_episodes}: "
            f"success={bool(info.get('success', False))!s:5s}  "
            f"range={info.get('range', 0.0):.2f} m  "
            f"rate={info.get('rate', 0.0):.3f} m/s  "
            f"steps={info.get('steps', 0):4d}  "
            f"fuel_used={info.get('fuel_used', 0):4d}"
        )

    vec_env.close()

    print("\n--- Evaluation Summary ---")
    print(f"Episodes     : {n_episodes}")
    print(f"Success rate : {successes / n_episodes * 100:.1f}%")
    print(f"Mean range   : {np.mean(final_ranges):.2f} m")
    print(f"Mean rate    : {np.mean(final_rates):.3f} m/s")
    print(f"Mean steps   : {np.mean(episode_steps):.1f}")
    print(f"Mean fuel    : {np.mean(episode_fuel_used):.1f}")


def main() -> None:
    """CLI entrypoint: parse evaluation options and run `evaluate`."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO agent on the SpaceX ISS Docking Simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="models/ppo_docking",
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes to run.",
    )
    parser.add_argument(
        "--launch-browser",
        action="store_true",
        help=(
            "Let Playwright launch a Chromium browser automatically."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser without a visible window.",
    )
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        n_episodes=args.episodes,
        launch_browser=args.launch_browser,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
