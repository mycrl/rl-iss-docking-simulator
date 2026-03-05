"""
Training script for the SpaceX ISS Docking Simulator.

Uses Soft Actor-Critic (SAC) from Stable-Baselines3 to train an agent on the
browser-based SpaceX ISS Docking Simulator (https://iss-sim.spacex.com/).

Supports vectorized training with multiple simulator environments. By default,
each environment runs in its own browser process (true parallelism via
``SubprocVecEnv``). Optionally, you can use one shared browser with multiple
tabs. A checkpoint callback saves the model periodically, and the ``--resume``
flag lets you continue training from a previous run.

Usage
-----
    # Default: 5 environments in parallel (one browser per env)
    python train.py --headless

    # Force single-environment CDP mode with a manually-opened Chrome
    python train.py --num-envs 1

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
import json
import logging
import os
from pathlib import Path
from typing import Callable
from typing import Any

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import VecMonitor

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


class ExportActionEffectCallback(BaseCallback):
    """Periodically export action-effect summary for visualization."""

    def __init__(self, export_freq: int, export_dir: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.export_freq = max(1, int(export_freq))
        self.export_dir = Path(export_dir)
        self._last_export_step = -1

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_export_step) < self.export_freq:
            return True

        vec_env = self.model.get_env()
        if vec_env is None:
            return True

        try:
            summaries = vec_env.env_method("get_action_effect_summary")
        except Exception:
            return True

        if not summaries:
            return True

        summary = summaries[0]
        payload = {
            "timesteps": int(self.num_timesteps),
            **summary,
        }

        self.export_dir.mkdir(parents=True, exist_ok=True)
        step_file = self.export_dir / f"action_effects_step_{self.num_timesteps}.json"
        latest_file = self.export_dir / "action_effects_latest.json"
        step_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        latest_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        self._last_export_step = self.num_timesteps
        logger.info("Exported action-effect summary to '%s'", step_file)
        return True


def _make_env(
    launch_browser: bool,
    headless: bool,
    shared_browser_tabs: bool,
    expected_shared_tabs: int | None,
    control_interval_steps: int,
    action_confirmation_steps: int,
    adaptive_control: bool,
    effect_guidance: bool,
) -> Callable[[], IssDockingEnv]:
    """Return an env factory function for vectorized env creation."""

    def _init() -> IssDockingEnv:
        return IssDockingEnv(
            launch_browser=launch_browser,
            headless=headless,
            shared_browser_tabs=shared_browser_tabs,
            expected_shared_tabs=expected_shared_tabs,
            control_interval_steps=control_interval_steps,
            action_confirmation_steps=action_confirmation_steps,
            adaptive_control=adaptive_control,
            effect_guidance=effect_guidance,
        )

    return _init


def _build_vec_env(
    num_envs: int,
    launch_browser: bool,
    headless: bool,
    shared_browser_tabs: bool,
    expected_shared_tabs: int | None,
    use_subproc_envs: bool,
    control_interval_steps: int,
    action_confirmation_steps: int,
    adaptive_control: bool,
    effect_guidance: bool,
) -> VecEnv:
    """Create vectorized training environment(s)."""
    env_fns = [
        _make_env(
            launch_browser=launch_browser,
            headless=headless,
            shared_browser_tabs=shared_browser_tabs,
            expected_shared_tabs=expected_shared_tabs,
            control_interval_steps=control_interval_steps,
            action_confirmation_steps=action_confirmation_steps,
            adaptive_control=adaptive_control,
            effect_guidance=effect_guidance,
        )
        for _ in range(num_envs)
    ]

    if num_envs > 1 and use_subproc_envs:
        return VecMonitor(SubprocVecEnv(env_fns, start_method="spawn"))

    return VecMonitor(DummyVecEnv(env_fns))


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
    effect_guidance: bool,
    shared_browser_tabs: bool,
    num_envs: int,
    effect_export_freq: int,
    effect_export_dir: str,
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
        Multi-environment runs auto-enable managed mode.
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
    effect_guidance:
        Enable learned button→state effect guidance for high-risk corrections.
    shared_browser_tabs:
        If ``True`` and ``num_envs > 1``, use one shared browser with multiple
        tabs. If ``False`` (default), each environment uses its own browser.
    num_envs:
        Number of simulator environments to run in parallel.
    effect_export_freq:
        Export learned action-effect summary every N timesteps.
    effect_export_dir:
        Directory for exported action-effect summary JSON files.
    """
    num_envs = max(1, int(num_envs))
    if num_envs > 1 and not launch_browser:
        logger.warning(
            "num_envs=%d requires managed mode for parallel browser control. "
            "Auto-enabling managed mode (--launch-browser).",
            num_envs,
        )
        launch_browser = True

    shared_browser_tabs = bool(launch_browser and shared_browser_tabs and num_envs > 1)
    expected_shared_tabs = num_envs if shared_browser_tabs else None
    use_subproc_envs = bool(num_envs > 1 and not shared_browser_tabs)

    if num_envs > 1 and not headless:
        if shared_browser_tabs:
            logger.warning(
                "Running one visible browser with %d tabs. Consider --headless for lower overhead.",
                num_envs,
            )
        else:
            logger.warning(
                "Running %d visible browser windows in parallel. Consider --headless for lower overhead.",
                num_envs,
            )

    env = _build_vec_env(
        num_envs=num_envs,
        launch_browser=launch_browser,
        headless=headless,
        shared_browser_tabs=shared_browser_tabs,
        expected_shared_tabs=expected_shared_tabs,
        use_subproc_envs=use_subproc_envs,
        control_interval_steps=control_interval_steps,
        action_confirmation_steps=action_confirmation_steps,
        adaptive_control=adaptive_control,
        effect_guidance=effect_guidance,
    )

    # SB3's save_replay_buffer appends ".pkl" automatically; use the base path
    # everywhere and check for the ".pkl" file on disk when resuming.
    replay_buffer_base = model_path + "_replay_buffer"
    replay_buffer_file = replay_buffer_base + ".pkl"

    if resume and os.path.exists(model_path + ".zip"):
        logger.info("Resuming training from '%s.zip' …", model_path)
        model = SAC.load(model_path, env=env)
        if os.path.exists(replay_buffer_file):
            try:
                model.load_replay_buffer(replay_buffer_file)
                logger.info("Replay buffer loaded from '%s'", replay_buffer_file)
            except Exception as exc:
                logger.warning(
                    "Could not load replay buffer from '%s' (%s). "
                    "Continuing with empty replay buffer.",
                    replay_buffer_file,
                    exc,
                )
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
    checkpoint_freq_calls = max(1, checkpoint_freq // num_envs)
    if checkpoint_freq_calls != checkpoint_freq:
        logger.info(
            "Adjusted checkpoint callback frequency to %d calls for %d envs "
            "(target every ~%d timesteps).",
            checkpoint_freq_calls,
            num_envs,
            checkpoint_freq,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq_calls,
        save_path=checkpoint_dir,
        name_prefix="sac_docking",
        save_replay_buffer=True,
    )
    success_save_callback = SaveOnSuccessCallback(
        model_path=model_path,
        replay_buffer_base=replay_buffer_base,
    )
    effect_export_callback = ExportActionEffectCallback(
        export_freq=effect_export_freq,
        export_dir=effect_export_dir,
    )

    def _save_progress(reason: str) -> None:
        dir_path = os.path.dirname(model_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        model.save(model_path)
        model.save_replay_buffer(replay_buffer_base)
        logger.info("Progress saved (%s) to '%s.zip'", reason, model_path)

    logger.info(
        "Training SAC for %s timesteps with %d environment(s)%s …",
        f"{timesteps:,}",
        num_envs,
        " (one shared browser, multi-tab)"
        if shared_browser_tabs
        else " (one browser per env, subprocess parallel)",
    )
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=[checkpoint_callback, success_save_callback, effect_export_callback],
            reset_num_timesteps=not resume,
        )
        _save_progress("completed")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C). Saving progress …")
        _save_progress("interrupted")
    finally:
        env.close()


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
            "When omitted, connect to Chrome via CDP for --num-envs=1; "
            "for --num-envs>1 managed mode is auto-enabled."
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
        default=1,
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
    parser.add_argument(
        "--effect-guidance",
        action="store_true",
        help="Enable learned action-effect guidance in high-risk states (default).",
    )
    parser.add_argument(
        "--no-effect-guidance",
        dest="effect_guidance",
        action="store_false",
        help="Disable learned action-effect guidance.",
    )
    parser.add_argument(
        "--shared-browser-tabs",
        action="store_true",
        help=(
            "Use one shared browser with multiple tabs for --num-envs>1. "
            "Default behavior is one browser per env in subprocess parallel mode."
        ),
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=5,
        help=(
            "Number of simulator environments to run in parallel. "
            "When >1, managed mode is enabled automatically."
        ),
    )
    parser.add_argument(
        "--effect-export-freq",
        type=int,
        default=5000,
        help="Export learned action-effect summary every N timesteps.",
    )
    parser.add_argument(
        "--effect-export-dir",
        default="analysis/effects",
        help="Directory for periodic action-effect summary JSON exports.",
    )
    parser.set_defaults(adaptive_control=True)
    parser.set_defaults(effect_guidance=True)
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
        effect_guidance=args.effect_guidance,
        shared_browser_tabs=args.shared_browser_tabs,
        num_envs=args.num_envs,
        effect_export_freq=args.effect_export_freq,
        effect_export_dir=args.effect_export_dir,
    )


if __name__ == "__main__":
    main()
