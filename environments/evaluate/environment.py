"""
Custom Gymnasium environment for the SpaceX ISS Docking Simulator.

Wraps the browser-controlled SpaceX ISS Docking Simulator
(https://iss-sim.spacex.com/) as a Gymnasium environment with continuous
observation space and MultiDiscrete action space.
"""

import logging
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .browser import SimulatorBrowser

logger = logging.getLogger(__name__)


class EvalIssDockingEnv(gym.Env):
    """
    Gymnasium environment that wraps the SpaceX ISS Docking Simulator.
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

    def __init__(
        self,
        launch_browser: bool = False,
        headless: bool = False,
        cdp_url: str = SimulatorBrowser.CDP_URL,
        shared_browser_tabs: bool = False,
        expected_shared_tabs: int | None = None,
        step_delay: float = 0.5,
        reset_wait: float = 3.0,
        max_steps: int | None = None,
        render_mode=None,
        **kwargs
    ) -> None:
        super().__init__()

        self.step_delay = step_delay
        self.reset_wait = reset_wait
        self.max_steps = max_steps
        self.render_mode = render_mode

        self._browser = SimulatorBrowser(
            launch=launch_browser,
            headless=headless,
            cdp_url=cdp_url,
            shared_launch=shared_browser_tabs,
            expected_shared_tabs=expected_shared_tabs,
        )
        self._browser.connect()

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

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self._browser.reset(wait=self.reset_wait)
        self._steps = 0
        self.fuel_used = 0
        self.fuel_remaining = self.INITIAL_FUEL
        
        obs = self._get_obs()
        self._prev_state = self._obs_to_dict(obs)
        return obs, {}

    def step(self, action: np.ndarray):
        button_presses = 0
        for dim, act_val in enumerate(action):
            act_val = int(act_val)
            if act_val in (1, 2):
                action_name = self.ACTION_MAP[dim][act_val]
                self._browser.click_action(action_name)
                button_presses += 1

        if button_presses > 0:
            fuel_spent = button_presses * self.FUEL_PER_BUTTON
            self.fuel_used += int(fuel_spent)
            self.fuel_remaining = max(0.0, self.fuel_remaining - fuel_spent)

        time.sleep(self.step_delay)
        self._steps += 1

        obs = self._get_obs()
        state = self._obs_to_dict(obs)
        # Evaluation-only environment: no reward shaping.
        reward = 0.0

        # Terminal Conditions
        current_range = state["range"]
        terminated = False
        truncated = False
        success = False

        if self._is_docked(state):
            terminated = True
            success = True
        elif self.fuel_remaining <= 0.0:
            terminated = True
        elif state["rate"] > 0.8:
            terminated = True
        elif current_range > self.MAX_RANGE:
            terminated = True
        elif any(abs(state[k]) > self.MAX_ATTITUDE for k in self.ATTITUDE_KEYS):
            terminated = True
        elif self.max_steps is not None and self._steps >= self.max_steps:
            truncated = True

        self._prev_state = state

        info = {
            "steps": self._steps,
            "fuel_used": int(self.fuel_used),
            "success": success,
            "button_presses": int(button_presses),
            "fuel_remaining": float(self.fuel_remaining),
            **state,
        }

        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        self._browser.disconnect()

    def _get_obs(self) -> np.ndarray:
        raw = self._browser.read_state()
                    
        obs = np.array(
            [raw[k] for k in self.OBS_KEYS if k != "fuel"] + [self.fuel_remaining / self.INITIAL_FUEL],
            dtype=np.float32,
        )
        return np.clip(obs, -self.OBS_HIGH, self.OBS_HIGH)

    def _obs_to_dict(self, obs: np.ndarray) -> dict[str, float]:
        return dict(zip(self.OBS_KEYS, obs.tolist()))

    @staticmethod
    def _is_docked(state: dict[str, float]) -> bool:
        return (
            abs(state["x"]) <= EvalIssDockingEnv.GOOD_POS_THRESHOLD
            and abs(state["y"]) <= EvalIssDockingEnv.GOOD_POS_THRESHOLD
            and abs(state["z"]) <= EvalIssDockingEnv.GOOD_POS_THRESHOLD
            and abs(state["roll"]) <= EvalIssDockingEnv.GOOD_ATTITUDE_THRESHOLD
            and abs(state["pitch"]) <= EvalIssDockingEnv.GOOD_ATTITUDE_THRESHOLD
            and abs(state["yaw"]) <= EvalIssDockingEnv.GOOD_ATTITUDE_THRESHOLD
            and abs(state["roll_rate"]) <= EvalIssDockingEnv.GOOD_ANG_RATE_THRESHOLD
            and abs(state["pitch_rate"]) <= EvalIssDockingEnv.GOOD_ANG_RATE_THRESHOLD
            and abs(state["yaw_rate"]) <= EvalIssDockingEnv.GOOD_ANG_RATE_THRESHOLD
            and 0.0 <= state["rate"] <= EvalIssDockingEnv.MAX_SAFE_RATE
            and state["range"] < EvalIssDockingEnv.GOOD_RANGE_THRESHOLD
        )

