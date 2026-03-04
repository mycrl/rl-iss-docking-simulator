"""
Training script for the SpaceX ISS Docking Simulator.

Uses Soft Actor-Critic (SAC) from Stable-Baselines3 to train an agent on the
browser-based SpaceX ISS Docking Simulator (https://iss-sim.spacex.com/).

Only a single simulator instance can run at a time, so training is sequential
(no parallel environments).  A checkpoint callback saves the model periodically,
and the ``--resume`` flag lets you continue training from a previous run.

Usage
-----
    # Auto-launch the browser (managed mode)
    python train.py --launch-browser

    # Connect to a manually-opened Chrome instance (CDP mode, default)
    python train.py

    # Resume from the latest saved model and replay buffer
    python train.py --launch-browser --resume --model-path models/sac_docking

    # Customise training length and checkpoint frequency
    python train.py --launch-browser --timesteps 1000000 --checkpoint-freq 20000

In CDP mode, start Chrome with remote debugging enabled before running::

    google-chrome --remote-debugging-port=9222 https://iss-sim.spacex.com/

Then run this script.
"""

import argparse
import importlib.util
import logging
import os
from typing import Any

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from docking import IssDockingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_tensorboard_log_dir() -> str | None:
    """Return TensorBoard log directory when tensorboard is available."""
    if importlib.util.find_spec("tensorboard") is None:
        logger.warning(
            "tensorboard is not installed; disabling TensorBoard logging. "
            "Install it with: pip install tensorboard"
        )
        return None
    return "./logs/"


class SaveOnSuccessCallback(BaseCallback):
    """Save model/replay buffer as soon as a successful docking episode occurs."""

    def __init__(self, model_path: str, replay_buffer_base: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.model_path = model_path
        self.replay_buffer_base = replay_buffer_base
        self._save_count = 0

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        dones = self.locals.get("dones")

        if dones is None:
            return True

        for done, info in zip(dones, infos):
            if bool(done) and bool(info.get("success", False)):
                self.model.save(self.model_path)
                self.model.save_replay_buffer(self.replay_buffer_base)
                self._save_count += 1
                logger.info(
                    "Docking success detected; model auto-saved to '%s.zip' (count=%d)",
                    self.model_path,
                    self._save_count,
                )
        return True


def train(
    model_path: str,
    timesteps: int,
    resume: bool,
    checkpoint_freq: int,
    checkpoint_dir: str,
    launch_browser: bool,
    headless: bool,
    control_interval_steps: int,
    action_confirmation_steps: int,
    adaptive_control: bool,
) -> None:
    """Train an SAC agent on the ISS docking environment.

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
    launch_browser:
        If ``True``, Playwright launches a Chromium browser automatically.
        If ``False``, connect to an already-running Chrome via CDP.
    headless:
        Only used when ``launch_browser=True``.  Run the browser without a
        visible window when ``True``.
    control_interval_steps:
        Execute one real control input every N environment steps to avoid
        over-control under inertia.
    action_confirmation_steps:
        In non-high-risk states, require the same intended command for N
        consecutive policy steps before executing a thrust click.
    adaptive_control:
        Dynamically increase/decrease control authority based on simulator
        risk state (distance, attitude, angular rates, closing rate).
    """
    env = Monitor(
        IssDockingEnv(
            launch_browser=launch_browser,
            headless=headless,
            control_interval_steps=control_interval_steps,
            action_confirmation_steps=action_confirmation_steps,
            adaptive_control=adaptive_control,
        )
    )

    # SB3's save_replay_buffer appends ".pkl" automatically; use the base path
    # everywhere and check for the ".pkl" file on disk when resuming.
    replay_buffer_base = model_path + "_replay_buffer"
    replay_buffer_file = replay_buffer_base + ".pkl"

    if resume and os.path.exists(model_path + ".zip"):
        logger.info("Resuming training from '%s.zip' …", model_path)
        model = SAC.load(model_path, env=env)
        if os.path.exists(replay_buffer_file):
            model.load_replay_buffer(replay_buffer_file)
            logger.info("Replay buffer loaded from '%s'", replay_buffer_file)
    else:
        logger.info("Starting fresh training …")
        model = SAC(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            tensorboard_log=_get_tensorboard_log_dir(),
        )

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="sac_docking",
        save_replay_buffer=True,
    )
    success_save_callback = SaveOnSuccessCallback(
        model_path=model_path,
        replay_buffer_base=replay_buffer_base,
    )

    logger.info("Training SAC for %s timesteps …", f"{timesteps:,}")
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, success_save_callback],
        reset_num_timesteps=not resume,
    )

    dir_path = os.path.dirname(model_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    model.save(model_path)
    model.save_replay_buffer(replay_buffer_base)
    logger.info("Model saved to '%s.zip'", model_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an SAC agent on the SpaceX ISS Docking Simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        default="models/sac_docking",
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
    parser.add_argument(
        "--control-interval-steps",
        type=int,
        default=2,
        help="Execute one control input every N env steps (>=1). Larger values reduce control frequency.",
    )
    parser.add_argument(
        "--action-confirmation-steps",
        type=int,
        default=2,
        help="Require same intended command for N consecutive steps before executing (except high-risk states).",
    )
    parser.add_argument(
        "--adaptive-control",
        action="store_true",
        help="Enable adaptive control authority (default).",
    )
    parser.add_argument(
        "--no-adaptive-control",
        dest="adaptive_control",
        action="store_false",
        help="Disable adaptive control and keep fixed control interval.",
    )
    parser.set_defaults(adaptive_control=True)
    args = parser.parse_args()

    train(
        model_path=args.model_path,
        timesteps=args.timesteps,
        resume=args.resume,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_dir=args.checkpoint_dir,
        launch_browser=args.launch_browser,
        headless=args.headless,
        control_interval_steps=args.control_interval_steps,
        action_confirmation_steps=args.action_confirmation_steps,
        adaptive_control=args.adaptive_control,
    )


if __name__ == "__main__":
    main()
