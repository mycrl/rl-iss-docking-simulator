"""
Custom Gymnasium environment for the SpaceX ISS Docking Simulator.

Wraps the browser-controlled SpaceX ISS Docking Simulator
(https://iss-sim.spacex.com/) as a Gymnasium environment with a continuous
observation space and a discrete action space.

Observation (8-D continuous)
-----------------------------
x, y, z    Position offsets from the docking axis (metres).
roll        Roll attitude error (degrees).
range       Distance to the ISS docking port (metres).
yaw         Yaw attitude error (degrees).
rate        Approach rate — negative means closing in (m/s).
pitch       Pitch attitude error (degrees).

Action (Discrete, 12)
-----------------------
Each action corresponds to a single RCS button press on the simulator
interface.  The resulting motion is continuous because momentum accumulates.

Index  Name
-----  ----
0      translate_forward
1      translate_backward
2      translate_up
3      translate_down
4      translate_left
5      translate_right
6      roll_left
7      roll_right
8      pitch_up
9      pitch_down
10     yaw_left
11     yaw_right

Docking success
---------------
All of x, y, z, roll, range, yaw, rate, pitch must be within
``SUCCESS_THRESHOLD`` (0.2) of zero simultaneously.
"""

import logging
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .browser import SimulatorBrowser

logger = logging.getLogger(__name__)


class IssDockingEnv(gym.Env):
    """
    Gymnasium environment that wraps the SpaceX ISS Docking Simulator.

    Parameters
    ----------
    cdp_url:
        Chrome DevTools Protocol endpoint URL.  Pass a custom value if Chrome
        is not listening on the default port (9222).
    step_delay:
        Seconds to wait after each button press before reading the new state.
        Longer delays give the simulator physics more time to settle.
    reset_wait:
        Extra seconds to wait after reloading the page for the simulator
        initialisation animation to complete.
    max_steps:
        Maximum number of steps per episode before truncation.
    render_mode:
        Unused; kept for Gymnasium API compatibility.
    """

    metadata = {"render_modes": []}

    # Ordered list of action names (index → button name).
    ACTIONS: list[str] = [
        "translate_forward",   # 0
        "translate_backward",  # 1
        "translate_up",        # 2
        "translate_down",      # 3
        "translate_left",      # 4
        "translate_right",     # 5
        "roll_left",           # 6
        "roll_right",          # 7
        "pitch_up",            # 8
        "pitch_down",          # 9
        "yaw_left",            # 10
        "yaw_right",           # 11
    ]

    # Observation upper bounds used for clipping and normalisation.
    # Order mirrors OBS_KEYS: x, y, z, roll, range, yaw, rate, pitch
    OBS_KEYS: list[str] = ["x", "y", "z", "roll", "range", "yaw", "rate", "pitch"]
    OBS_HIGH: np.ndarray = np.array(
        [300.0, 300.0, 300.0, 180.0, 500.0, 180.0, 5.0, 180.0],
        dtype=np.float32,
    )

    # Docking success: all readings must be strictly below this value.
    SUCCESS_THRESHOLD: float = 0.2

    # Collision guard: approach rate (m/s, negative = closing) must not
    # exceed this magnitude when within NEAR_DISTANCE of the port.
    MAX_SAFE_RATE: float = 0.2   # m/s
    NEAR_DISTANCE: float = 5.0   # metres

    # Reward constants.
    REWARD_SUCCESS: float = 100.0
    REWARD_COLLISION: float = -50.0
    REWARD_STEP: float = -0.01   # per-step time penalty

    def __init__(
        self,
        cdp_url: str = SimulatorBrowser.CDP_URL,
        step_delay: float = 0.5,
        reset_wait: float = 5.0,
        max_steps: int = 3000,
        render_mode=None,
    ) -> None:
        super().__init__()

        self.step_delay = step_delay
        self.reset_wait = reset_wait
        self.max_steps = max_steps
        self.render_mode = render_mode

        self._browser = SimulatorBrowser(cdp_url=cdp_url)
        self._browser.connect()

        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(
            low=-self.OBS_HIGH,
            high=self.OBS_HIGH,
            dtype=np.float32,
        )

        self._steps: int = 0
        self._prev_error: float = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Reload the simulator page and return the initial observation."""
        super().reset(seed=seed)
        self._browser.reset(wait=self.reset_wait)
        self._steps = 0
        obs = self._get_obs()
        self._prev_error = self._total_error(obs)
        return obs, {}

    def step(self, action: int):
        """Apply *action* (button press), wait, then return the new state.

        Parameters
        ----------
        action:
            Integer index into :attr:`ACTIONS`.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        action_name = self.ACTIONS[int(action)]
        self._browser.click_action(action_name)
        time.sleep(self.step_delay)
        self._steps += 1

        obs = self._get_obs()
        state = self._obs_to_dict(obs)
        error = self._total_error(obs)

        terminated = False
        truncated = False
        success = False

        # Shaped reward: progress + per-step penalty
        reward = self.REWARD_STEP + (self._prev_error - error)
        self._prev_error = error

        if self._is_docked(state):
            reward += self.REWARD_SUCCESS
            terminated = True
            success = True
        elif (
            state["range"] < self.NEAR_DISTANCE
            and state["rate"] < -self.MAX_SAFE_RATE
        ):
            # Approaching too fast within 5 m — collision risk
            reward += self.REWARD_COLLISION
            terminated = True
        elif self._steps >= self.max_steps:
            truncated = True

        info = {
            "steps": self._steps,
            "total_error": float(error),
            "success": success,
            **state,
        }
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        """Disconnect from the browser and release resources."""
        self._browser.disconnect()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Read simulator state and return a clipped float32 observation."""
        raw = self._browser.read_state()
        obs = np.array([raw[k] for k in self.OBS_KEYS], dtype=np.float32)
        return np.clip(obs, -self.OBS_HIGH, self.OBS_HIGH)

    def _obs_to_dict(self, obs: np.ndarray) -> dict[str, float]:
        """Convert an observation vector to a named dictionary."""
        return dict(zip(self.OBS_KEYS, obs.tolist()))

    def _total_error(self, obs: np.ndarray) -> float:
        """Normalised sum of absolute observation values (scale: [0, N]).

        Each component is divided by its maximum bound before summing, so
        all components contribute equally regardless of unit.
        """
        return float(np.sum(np.abs(obs) / self.OBS_HIGH))

    @staticmethod
    def _is_docked(state: dict[str, float]) -> bool:
        """Return ``True`` when every reading is within the success threshold."""
        return all(abs(v) < IssDockingEnv.SUCCESS_THRESHOLD for v in state.values())
