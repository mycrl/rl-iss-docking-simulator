"""
Training script for the SpaceX ISS Docking Simulator.

Uses Deep Q-Network (DQN) from Stable-Baselines3 to train an agent on the
browser-based SpaceX ISS Docking Simulator (https://iss-sim.spacex.com/).

Only a single simulator instance can run at a time, so training is sequential
(no parallel environments).  A checkpoint callback saves the model periodically,
and the ``--resume`` flag lets you continue training from a previous run.

Usage
-----
    # Start fresh
    python train.py

    # Resume from the latest saved model and replay buffer
    python train.py --resume --model-path models/dqn_docking

    # Customise training length and checkpoint frequency
    python train.py --timesteps 1000000 --checkpoint-freq 20000

Before running, start Chrome with remote debugging enabled::

    google-chrome --remote-debugging-port=9222 https://iss-sim.spacex.com/

Then fill in the CSS selectors in ``docking/browser.py`` and run this script.
"""

import argparse
import logging
import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from docking import IssDockingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(
    model_path: str,
    timesteps: int,
    resume: bool,
    checkpoint_freq: int,
    checkpoint_dir: str,
) -> None:
    """Train a DQN agent on the ISS docking environment.

    Parameters
    ----------
    model_path:
        Path (without ``.zip`` extension) to save the final model.
    timesteps:
        Total number of environment steps to train for.
    resume:
        If ``True``, load an existing model (and replay buffer if present)
        and continue training from where it left off.
    checkpoint_freq:
        Save an intermediate checkpoint every this many steps.
    checkpoint_dir:
        Directory in which checkpoint files are stored.
    """
    env = Monitor(IssDockingEnv())

    # SB3's save_replay_buffer appends ".pkl" automatically; use the base path
    # everywhere and check for the ".pkl" file on disk when resuming.
    replay_buffer_base = model_path + "_replay_buffer"
    replay_buffer_file = replay_buffer_base + ".pkl"

    if resume and os.path.exists(model_path + ".zip"):
        logger.info("Resuming training from '%s.zip' …", model_path)
        model = DQN.load(model_path, env=env)
        if os.path.exists(replay_buffer_file):
            model.load_replay_buffer(replay_buffer_file)
            logger.info("Replay buffer loaded from '%s'", replay_buffer_file)
    else:
        logger.info("Starting fresh training …")
        model = DQN(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1_000,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            tensorboard_log="./logs/",
        )

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="dqn_docking",
        save_replay_buffer=True,
    )

    logger.info("Training DQN for %s timesteps …", f"{timesteps:,}")
    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=not resume,
        progress_bar=True,
    )

    dir_path = os.path.dirname(model_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    model.save(model_path)
    model.save_replay_buffer(replay_buffer_base)
    logger.info("Model saved to '%s.zip'", model_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on the SpaceX ISS Docking Simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        default="models/dqn_docking",
        help="Path to save (or load, when resuming) the model.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from an existing model.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10_000,
        help="Save a checkpoint every N environment steps.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory for periodic checkpoint files.",
    )
    args = parser.parse_args()

    train(
        model_path=args.model_path,
        timesteps=args.timesteps,
        resume=args.resume,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
