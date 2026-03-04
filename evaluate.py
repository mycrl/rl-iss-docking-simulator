"""
Evaluation script for the SpaceX ISS Docking Simulator.

Loads a trained SAC model and runs it on the browser-based SpaceX ISS Docking
Simulator in deterministic mode, then prints per-episode and aggregate stats.

Usage
-----
    # Auto-launch the browser (managed mode)
    python evaluate.py --launch-browser --model models/sac_docking --episodes 10

    # Connect to a manually-opened Chrome instance (CDP mode, default)
    python evaluate.py --model models/sac_docking --episodes 10

In CDP mode, make sure the simulator is open in Chrome with remote debugging
enabled (see ``docking/browser.py`` for instructions).
"""

import argparse
import logging

import numpy as np
from stable_baselines3 import SAC

from docking import IssDockingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(
    model_path: str,
    n_episodes: int,
    launch_browser: bool,
    headless: bool,
) -> None:
    """Run deterministic evaluation episodes and print a summary.

    Parameters
    ----------
    model_path:
        Path to the saved SAC model (with or without the ``.zip`` extension).
    n_episodes:
        Number of episodes to evaluate.
    launch_browser:
        If ``True``, Playwright launches a Chromium browser automatically.
        If ``False``, connect to an already-running Chrome via CDP.
    headless:
        Only used when ``launch_browser=True``.  Run the browser without a
        visible window when ``True``.
    """
    env = IssDockingEnv(launch_browser=launch_browser, headless=headless)
    model = SAC.load(model_path, env=env)

    episode_rewards: list[float] = []
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
        if info.get("success", False):
            successes += 1

        print(
            f"Episode {episode + 1:3d}/{n_episodes}: "
            f"reward={total_reward:8.2f}  "
            f"range={info.get('range', 0.0):.2f} m  "
            f"rate={info.get('rate', 0.0):.3f} m/s  "
            f"steps={info['steps']}"
        )

    env.close()

    print("\n--- Evaluation Summary ---")
    print(f"Episodes     : {n_episodes}")
    print(f"Success rate : {successes / n_episodes * 100:.1f}%")
    print(f"Mean reward  : {np.mean(episode_rewards):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SAC agent on the SpaceX ISS Docking Simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="models/sac_docking",
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
            "Let Playwright launch a Chromium browser automatically. "
            "When omitted, connect to an already-running Chrome via CDP."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser without a visible window (only used with --launch-browser).",
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
