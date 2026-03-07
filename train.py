"""
Training script for the SpaceX ISS Docking Simulator (Fast Version).

Uses Proximal Policy Optimization (PPO) from Stable-Baselines3 to train an agent.
To maximise performance, this script exclusively uses the purely Python-based
TrainIssDockingEnv, achieving thousands of steps per second.
"""

import argparse
import importlib.util
import logging
import os
import time
from typing import Callable, Any

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import VecNormalize

from environments import TrainIssDockingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _safe_save_model(model: BaseAlgorithm, path: str, reason: str, max_retries: int = 2) -> str | None:
    """Save model robustly against transient filesystem errors on Windows."""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    for attempt in range(max_retries + 1):
        try:
            model.save(path)
            return path
        except OSError as exc:
            if attempt >= max_retries:
                break
            delay_seconds = 0.2 * (attempt + 1)
            logger.warning(
                "Model save failed (%s) at '%s' (attempt %d/%d): %s. Retrying in %.1fs ...",
                reason,
                path,
                attempt + 1,
                max_retries + 1,
                exc,
                delay_seconds,
            )
            time.sleep(delay_seconds)

    fallback_path = f"{path}_fallback_{int(time.time())}"
    try:
        model.save(fallback_path)
        logger.warning(
            "Primary save failed (%s). Saved fallback model to '%s.zip' instead.",
            reason,
            fallback_path,
        )
        return fallback_path
    except OSError as exc:
        logger.error(
            "Model save permanently failed (%s) for '%s' and fallback '%s': %s",
            reason,
            path,
            fallback_path,
            exc,
        )
        return None


def _get_tensorboard_log_dir() -> str | None:
    if importlib.util.find_spec("tensorboard") is None:
        return None
    return "./logs/"


class SaveOnSuccessCallback(BaseCallback):
    """Callback that saves the model whenever an episode ends in success.

    The callback inspects the `infos` returned by the environment for a
    `success` flag (boolean). When a finished episode reports success, the
    current model is saved using a safe save helper to avoid transient
    filesystem errors on Windows.
    """
    def __init__(self, model_path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.model_path = model_path
        self._save_count = 0

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        dones = self.locals.get("dones")

        if dones is None:
            return True

        for done, info in zip(dones, infos):
            if bool(done) and bool(info.get("success", False)):
                saved_path = _safe_save_model(self.model, self.model_path, "success-autosave")
                if saved_path is not None:
                    self._save_count += 1
                    logger.info(
                        "Docking success detected; model auto-saved to '%s.zip' (count=%d)",
                        saved_path,
                        self._save_count,
                    )
        return True


class SafeCheckpointCallback(BaseCallback):
    """Periodic checkpoint callback with robust saving.

    Saves the model every `save_freq` calls to the given `save_path` with the
    provided `name_prefix`. Uses the `_safe_save_model` wrapper to retry on
    transient errors and log failures.
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.save_freq = max(1, int(save_freq))
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True

        checkpoint_path = os.path.join(
            self.save_path,
            f"{self.name_prefix}_{self.num_timesteps}_steps",
        )
        saved_path = _safe_save_model(self.model, checkpoint_path, "periodic-checkpoint")
        if saved_path is not None:
            logger.info("Checkpoint saved to '%s.zip'", saved_path)
        return True


def _make_env() -> Callable[[], gym.Env]:
    """Factory returning a callable that creates a single `TrainIssDockingEnv`.

    This is used by the vectorized environment constructors (Dummy/Subproc)
    which expect a no-argument callable that produces a fresh env instance.
    """
    def _init() -> gym.Env:
        return TrainIssDockingEnv()

    return _init


def _build_vec_env(
    num_envs: int,
    use_subproc_envs: bool,
    model_path: str = "models/ppo_docking",
) -> VecEnv:
    """Construct a vectorized, monitored and normalized environment.

    - Creates `num_envs` copies of the fast Python environment.
    - Chooses `SubprocVecEnv` when `use_subproc_envs` is True and multiple
        envs are requested; otherwise uses `DummyVecEnv`.
    - Wraps with `VecMonitor` to record episode statistics and with
        `VecNormalize` to normalize observations and rewards. If normalization
        stats exist on disk they will be loaded so training can resume steadily.
    """
    # Create factory callables for each parallel env. Each callable must
    # produce a fresh environment instance when called by the vector wrapper.
    env_fns = [_make_env() for _ in range(num_envs)]

    if num_envs > 1 and use_subproc_envs:
        env = VecMonitor(SubprocVecEnv(env_fns, start_method="spawn"))
    else:
        env = VecMonitor(DummyVecEnv(env_fns))
    
    vec_env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    stats_path = model_path + "_vec_normalize.pkl"
    # If normalization statistics exist from a previous run, load them so
    # observation/reward scaling remains consistent across resume sessions.
    # Note: VecNormalize.load requires passing the underlying env used for
    # runtime wrapping, so we reload into the current `env` instance.
    if os.path.exists(stats_path):
        logger.info(f"Loading environment normalization stats from {stats_path}")
        vec_env = VecNormalize.load(stats_path, env)
        # Ensure the wrapper stays in training mode while learning resumes.
        vec_env.training = True
        
    return vec_env


def train(
    model_path: str,
    timesteps: int,
    resume: bool,
    checkpoint_freq: int,
    checkpoint_dir: str,
    num_envs: int,
) -> None:
    """Train a PPO agent using the configured environment and callbacks.

    Args:
        model_path: Base path (without .zip) to save/load the policy.
        timesteps: Total number of timesteps to learn for.
        resume: If True, attempt to resume from an existing checkpoint.
        checkpoint_freq: How often (in environment steps) to save checkpoints.
        checkpoint_dir: Directory where checkpoints are written.
        num_envs: Number of parallel environments to use.
    """
    num_envs = max(1, int(num_envs))
    use_subproc_envs = bool(num_envs > 1)

    env = _build_vec_env(
        num_envs=num_envs,
        use_subproc_envs=use_subproc_envs,
        model_path=model_path if resume else "None",
    )

    if resume and os.path.exists(model_path + ".zip"):
        logger.info("Resuming training from '%s.zip' ...", model_path)
        model = PPO.load(model_path, env=env)
    else:
        logger.info("Starting fresh training ...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=2e-4,
            n_steps=2048,
            batch_size=512,
            gamma=0.995,
            gae_lambda=0.98,
            ent_coef=0.002,
            tensorboard_log=_get_tensorboard_log_dir(),
        )

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_freq_calls = max(1, checkpoint_freq // num_envs)
    # checkpoint_freq is expressed in environment *steps*. When using
    # vectorized envs, `model.learn` advances `num_envs` observations per
    # algorithm step. We divide by `num_envs` so that checkpoints are saved
    # approximately every `checkpoint_freq` environment steps in total.

    checkpoint_callback = SafeCheckpointCallback(
        save_freq=checkpoint_freq_calls,
        save_path=checkpoint_dir,
        name_prefix="ppo_docking",
    )
    success_save_callback = SaveOnSuccessCallback(model_path=model_path)

    def _save_progress(reason: str) -> None:
        saved_model_path = _safe_save_model(model, model_path, f"final-{reason}")
        if saved_model_path is not None:
            env.save(saved_model_path + "_vec_normalize.pkl") # type: ignore
            logger.info(
                "Progress saved (%s) to '%s.zip' and stats to '%s_vec_normalize.pkl'",
                reason,
                saved_model_path,
                saved_model_path,
            )
        else:
            logger.error("Could not save final model during '%s'.", reason)

    logger.info(
        "Training PPO entirely on CPU (Fast env) for %s total timesteps with %d environment(s)...",
        f"{timesteps:,}",
        num_envs,
    )
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=[checkpoint_callback, success_save_callback],
            reset_num_timesteps=not resume,
        )
        _save_progress("completed")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C). Saving progress ...")
        _save_progress("interrupted")
    finally:
        env.close()


def main() -> None:
    """CLI entrypoint: parse arguments and start training."""
    parser = argparse.ArgumentParser(
        description="Fast Train a PPO agent on the pure Python SpaceX ISS Docking Simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", default="models/ppo_docking", help="Path to save/load the model.")
    parser.add_argument("--timesteps", type=int, default=5_000_000, help="Total training timesteps.")
    parser.add_argument("--resume", action="store_true", help="Resume training.")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Checkpoint frequency in steps.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for checkpoints.")
    parser.add_argument("--num-envs", type=int, default=16, help="Number of vectorized fast envs.")
    args = parser.parse_args()

    train(
        model_path=args.model_path,
        timesteps=args.timesteps,
        resume=args.resume,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_dir=args.checkpoint_dir,
        num_envs=args.num_envs,
    )

if __name__ == "__main__":
    main()
