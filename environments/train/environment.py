"""
Pure Python simulation of the SpaceX ISS Docking Simulator.

This environment perfectly mirrors the state space, action space, 
and reward scale of the actual simulator (IssDockingEnv) but runs 
locally at thousands of steps per second without Playwright or a browser.
"""

import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .simulator import TrainDockingSimulator

logger = logging.getLogger(__name__)

class TrainIssDockingEnv(gym.Env):
    """
    Lightning-fast Gymnasium environment mirroring the SpaceX ISS Docking Simulator.
    Runs entirely in Python.
    """

    metadata = {"render_modes": []}

    INITIAL_FUEL: float = 800.0
    FUEL_PER_BUTTON: float = 1.0

    OBS_KEYS: list[str] = [
        "x", "y", "z",
        "roll", "roll_rate",
        "range",
        "yaw", "yaw_rate",
        "rate",
        "pitch", "pitch_rate",
        "fuel",
    ]
    OBS_HIGH: np.ndarray = np.array(
        [300.0, 300.0, 300.0, 180.0, 10.0, 500.0, 180.0, 10.0, 5.0, 180.0, 10.0, 1.0],
        dtype=np.float32,
    )

    SUCCESS_THRESHOLD: float = 0.2
    GOOD_POS_THRESHOLD: float = 0.2
    GOOD_ATTITUDE_THRESHOLD: float = 0.2
    GOOD_ANG_RATE_THRESHOLD: float = 0.25
    GOOD_RANGE_THRESHOLD: float = 2.0
    MAX_RANGE: float = 350.0   # metres
    MAX_ATTITUDE: float = 30.0   # degrees
    MIN_SAFE_RATE: float = 0.02  # m/s
    MAX_SAFE_RATE: float = 0.2   # m/s
    NEAR_DISTANCE: float = 5.0   # metres
    ATTITUDE_KEYS: tuple[str, ...] = ("roll", "yaw", "pitch")
    ACTION_MAP: dict[int, dict[int, str]] = {
        0: {1: "translate_forward", 2: "translate_backward"},
        1: {1: "translate_up", 2: "translate_down"},
        2: {1: "translate_right", 2: "translate_left"},
        3: {1: "roll_right", 2: "roll_left"},
        4: {1: "pitch_up", 2: "pitch_down"},
        5: {1: "yaw_right", 2: "yaw_left"},
    }
    TRANSLATION_QUICK_REPEAT_PENALTY: float = 0.12
    TRANSLATION_DIRECTION_FLIP_PENALTY: float = 0.16
    METRIC_REWARD_WEIGHTS: dict[str, float] = {
        "x": 0.9,
        "y": 0.9,
        "z": 0.9,
        "roll": 1.0,
        "pitch": 1.0,
        "yaw": 1.0,
        "roll_rate": 1.1,
        "pitch_rate": 1.1,
        "yaw_rate": 1.1,
        "rate": 1.4,
        "range": 1.2,
    }

    def __init__(
        self,
        step_delay: float = 0.5,
        max_steps: int | None = None,
        render_mode=None,
        **kwargs
    ) -> None:
        super().__init__()

        # step_delay is now used entirely as our local physics 'dt' (delta time)
        self.dt = step_delay
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3, 3, 3])
        self.observation_space = spaces.Box(
            low=-self.OBS_HIGH,
            high=self.OBS_HIGH,
            dtype=np.float32,
        )

        self._steps: int = 0
        self.fuel_used: int = 0
        self.fuel_remaining: float = self.INITIAL_FUEL
        self._prev_state = {}
        self._prev_action: np.ndarray = np.zeros(6, dtype=np.int8)
        self._sim = TrainDockingSimulator(dt=self.dt)
        self.state_vars: dict[str, float] = {}

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self._steps = 0
        self._prev_action.fill(0)
        self._sim.reset(self.np_random)
        self._sync_from_sim()

        obs = self._get_obs()
        self._prev_state = self._obs_to_dict(obs)
        return obs, {}

    def step(self, action: np.ndarray):
        step_idx = self._steps
        for dim, act_val_raw in enumerate(action):
            act_val = int(act_val_raw)
            if act_val in (1, 2):
                action_name = self.ACTION_MAP[dim][act_val]
                self._sim.click_action(action_name)

        self._sync_from_sim(drive=True)
        self._steps += 1

        button_presses = int(self._sim.button_presses)
        active_dims = set(self._sim.active_dims)
        quick_repeat_translation_dims = set(self._sim.quick_repeat_translation_dims)
        flip_translation_dims = set(self._sim.flip_translation_dims)

        obs = self._get_obs()
        state = self._obs_to_dict(obs)

        # =========================================================
        # REWARD COMPUTATION
        # =========================================================
        reward_components: dict[str, float] = {}

        # 1. Action/Fuel penalty (per control dimension)
        for dim in active_dims:
            self._add_reward_component(reward_components, f"fuel_dim_{dim}", -0.03)

        for dim in sorted(quick_repeat_translation_dims):
            self._add_reward_component(
                reward_components,
                f"translation_quick_repeat_dim{dim}",
                -self.TRANSLATION_QUICK_REPEAT_PENALTY,
            )

        for dim in sorted(flip_translation_dims):
            self._add_reward_component(
                reward_components,
                f"translation_flip_dim{dim}",
                -self.TRANSLATION_DIRECTION_FLIP_PENALTY,
            )

        progress_component_scores: dict[str, float] = {}
        noop_component_scores: dict[str, float] = {}

        # Local per-dimension credit: each action only affects its own mapped metrics.
        dim_to_metrics: dict[int, tuple[str, ...]] = {
            0: ("range", "rate"),
            1: ("z",),
            2: ("y", "x"),
            3: ("roll", "roll_rate"),
            4: ("pitch", "pitch_rate"),
            5: ("yaw", "yaw_rate"),
        }

        for dim, metrics in dim_to_metrics.items():
            act_val = int(action[dim])
            is_active = act_val in (1, 2)
            prev_same_dir = bool(self._prev_action[dim] == act_val and is_active)

            for metric in metrics:
                improvement = self._metric_improvement(metric, self._prev_state, state)
                weight = self.METRIC_REWARD_WEIGHTS.get(metric, 1.0)
                progress_score = float(np.clip(improvement * weight, -0.8, 0.8))
                progress_component_scores[f"dim{dim}_{metric}"] = progress_score

                if is_active:
                    # Translation has delayed response, so immediate reward for
                    # translation actions is intentionally softened.
                    local_progress = progress_score * (0.35 if dim < 3 else 1.0)
                    self._add_reward_component(
                        reward_components,
                        f"active_dim{dim}_{metric}",
                        local_progress,
                    )
                    # If metric already improving, repeating same direction is over-excited.
                    if improvement > 0.0 and prev_same_dir:
                        repeat_penalty = -0.08 if dim < 3 else -0.12
                        self._add_reward_component(
                            reward_components,
                            f"repeat_dim{dim}_{metric}",
                            repeat_penalty,
                        )
                    if improvement <= 0.0 and dim >= 3:
                        self._add_reward_component(
                            reward_components,
                            f"ineffective_dim{dim}_{metric}",
                            -0.06,
                        )
                else:
                    violation = self._metric_violation(metric, state)
                    observe_window = (
                        dim < 3
                        and 0 < (step_idx - int(self._sim.translation_last_command_step[dim]))
                        <= self._sim.TRANSLATION_OBSERVE_WINDOW_STEPS
                    )
                    if improvement > 0.0:
                        hold_reward = float(np.clip(improvement * 0.5, 0.0, 0.25))
                        noop_component_scores[f"dim{dim}_{metric}"] = hold_reward
                        self._add_reward_component(
                            reward_components,
                            f"hold_dim{dim}_{metric}",
                            hold_reward,
                        )
                        if observe_window:
                            observe_reward = float(np.clip(improvement * 0.8, 0.0, 0.2))
                            self._add_reward_component(
                                reward_components,
                                f"observe_dim{dim}_{metric}",
                                observe_reward,
                            )
                    elif violation > 0.0:
                        lazy_penalty = -float(np.clip((violation + (-improvement)) * 0.35, 0.0, 0.25))
                        noop_component_scores[f"dim{dim}_{metric}"] = lazy_penalty
                        self._add_reward_component(
                            reward_components,
                            f"lazy_dim{dim}_{metric}",
                            lazy_penalty,
                        )

        # 4. Safety Constraints & Violations (Extreme Penalties)
        current_range = state["range"]
        if current_range < self.NEAR_DISTANCE and state["rate"] > self.MAX_SAFE_RATE:
            self._add_reward_component(reward_components, "near_overspeed", -10.0)

        # Global approach-rate safety shaping: regardless of distance, heavily
        # discourage unsafe closing speed beyond configured limit.
        # Per requested rule: rate can be negative physically, but negative means backing away and is penalized.
        if state["rate"] < 0.0:
            reverse_speed = -state["rate"]
            self._add_reward_component(
                reward_components,
                "rate_reverse",
                -((reverse_speed) * 30.0) ** 2,
            )
        elif state["rate"] < self.MIN_SAFE_RATE:
            under_speed = self.MIN_SAFE_RATE - state["rate"]
            self._add_reward_component(
                reward_components,
                "rate_under",
                -((under_speed) * 20.0) ** 2,
            )
        elif state["rate"] > self.MAX_SAFE_RATE:
            overspeed = state["rate"] - self.MAX_SAFE_RATE
            self._add_reward_component(
                reward_components,
                "rate_overspeed",
                -((overspeed) * 30.0) ** 2,
            )

        if current_range > 15.0 and 0.0 <= state["rate"] < self.MIN_SAFE_RATE:
            self._add_reward_component(reward_components, "far_stagnation", -0.1)

        # c) Angular-rate target-band shaping (per-axis, no global coupling).
        axis_to_rate = {
            "roll": ("roll_rate", -0.02),
            "pitch": ("pitch_rate", 0.02),
            "yaw": ("yaw_rate", 0.02),
        }
        for axis, (rate_key, gain) in axis_to_rate.items():
            target_rate = float(np.clip(state[axis] * gain, -self.GOOD_ANG_RATE_THRESHOLD, self.GOOD_ANG_RATE_THRESHOLD))
            delta = abs(state[rate_key] - target_rate)
            self._add_reward_component(reward_components, f"angular_target_{axis}", -delta * 0.8)

        # Keep hard punishment only for clearly unsafe spin rates.
        for key in ("roll_rate", "yaw_rate", "pitch_rate"):
            rate_val = abs(state[key])
            if rate_val > self.GOOD_ANG_RATE_THRESHOLD:
                self._add_reward_component(
                    reward_components,
                    f"spin_overspeed_{key}",
                    -((rate_val - self.GOOD_ANG_RATE_THRESHOLD) * 12.0) ** 2,
                )

        # 5. Terminal Conditions
        terminated = False
        truncated = False
        success = False

        if self._is_docked(state):
            self._add_reward_component(reward_components, "terminal_success", 1000.0)
            terminated = True
            success = True
        elif self.fuel_remaining <= 0.0:
            self._add_reward_component(reward_components, "terminal_fuel_empty", -300.0)
            terminated = True
        elif state["rate"] > 0.8:
            self._add_reward_component(reward_components, "terminal_rate_overspeed", -500.0)
            terminated = True
        elif current_range > self.MAX_RANGE:
            self._add_reward_component(reward_components, "terminal_range_limit", -1000.0)
            terminated = True
        elif any(abs(state[k]) > self.MAX_ATTITUDE for k in self.ATTITUDE_KEYS):
            self._add_reward_component(reward_components, "terminal_attitude_limit", -1000.0)
            terminated = True
        elif self.max_steps is not None and self._steps >= self.max_steps:
            truncated = True

        reward = float(sum(reward_components.values()))

        self._prev_state = state
        for dim in range(6):
            v = int(action[dim])
            self._prev_action[dim] = v if v in (1, 2) else 0

        info = {
            "steps": self._steps,
            "fuel_used": int(self.fuel_used),
            "fuel_remaining": float(self.fuel_remaining),
            "success": success,
            "button_presses": int(button_presses),
            "translation_pending_pulses": int(len(self._sim.translation_pending)),
            "reward_components": reward_components,
            "progress_component_scores": progress_component_scores,
            "noop_component_scores": noop_component_scores,
            **state,
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        pass # No browser to close!

    def _sync_from_sim(self, drive: bool = False) -> None:
        self.state_vars = self._sim.read_state() if drive else self._sim.get_state_snapshot()
        self.fuel_used = self._sim.fuel_used
        self.fuel_remaining = self._sim.fuel_remaining

    def _get_obs(self) -> np.ndarray:
        obs = np.array(
            [self.state_vars[k] for k in self.OBS_KEYS if k != "fuel"] + [self.fuel_remaining / self.INITIAL_FUEL],
            dtype=np.float32,
        )
        return np.clip(obs, -self.OBS_HIGH, self.OBS_HIGH)

    def _obs_to_dict(self, obs: np.ndarray) -> dict[str, float]:
        return dict(zip(self.OBS_KEYS, obs.tolist()))

    def _metric_violation(self, key: str, state: dict[str, float]) -> float:
        if key in ("x", "y", "z"):
            return max(0.0, abs(state[key]) - self.GOOD_POS_THRESHOLD)
        if key in ("roll", "pitch", "yaw"):
            return max(0.0, abs(state[key]) - self.GOOD_ATTITUDE_THRESHOLD)
        if key in ("roll_rate", "pitch_rate", "yaw_rate"):
            return max(0.0, abs(state[key]) - self.GOOD_ANG_RATE_THRESHOLD)
        if key == "range":
            return max(0.0, state["range"] - self.GOOD_RANGE_THRESHOLD)
        if key == "rate":
            return max(0.0, self.MIN_SAFE_RATE - state["rate"]) + max(0.0, state["rate"] - self.MAX_SAFE_RATE)
        return 0.0

    def _metric_improvement(
        self,
        key: str,
        prev_state: dict[str, float],
        curr_state: dict[str, float],
    ) -> float:
        return self._metric_violation(key, prev_state) - self._metric_violation(key, curr_state)

    @staticmethod
    def _add_reward_component(components: dict[str, float], key: str, value: float) -> None:
        components[key] = components.get(key, 0.0) + float(value)

    @staticmethod
    def _is_docked(state: dict[str, float]) -> bool:
        return (
            abs(state["x"]) <= TrainIssDockingEnv.GOOD_POS_THRESHOLD
            and abs(state["y"]) <= TrainIssDockingEnv.GOOD_POS_THRESHOLD
            and abs(state["z"]) <= TrainIssDockingEnv.GOOD_POS_THRESHOLD
            and abs(state["roll"]) <= TrainIssDockingEnv.GOOD_ATTITUDE_THRESHOLD
            and abs(state["pitch"]) <= TrainIssDockingEnv.GOOD_ATTITUDE_THRESHOLD
            and abs(state["yaw"]) <= TrainIssDockingEnv.GOOD_ATTITUDE_THRESHOLD
            and abs(state["roll_rate"]) <= TrainIssDockingEnv.GOOD_ANG_RATE_THRESHOLD
            and abs(state["pitch_rate"]) <= TrainIssDockingEnv.GOOD_ANG_RATE_THRESHOLD
            and abs(state["yaw_rate"]) <= TrainIssDockingEnv.GOOD_ANG_RATE_THRESHOLD
            and TrainIssDockingEnv.MIN_SAFE_RATE <= state["rate"] <= TrainIssDockingEnv.MAX_SAFE_RATE
            and state["range"] < TrainIssDockingEnv.GOOD_RANGE_THRESHOLD
        )

