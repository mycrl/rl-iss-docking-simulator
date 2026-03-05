"""
Custom Gymnasium environment for the SpaceX ISS Docking Simulator.

Wraps the browser-controlled SpaceX ISS Docking Simulator
(https://iss-sim.spacex.com/) as a Gymnasium environment with continuous
observation and action spaces.

Observation (11-D continuous)
------------------------------
x, y, z         Position offsets from the docking axis (metres).
roll            Roll attitude error (degrees).
roll_rate       Roll angular rate (°/s).
range           Distance to the ISS docking port (metres).
yaw             Yaw attitude error (degrees).
yaw_rate        Yaw angular rate (°/s).
rate            Approach rate — negative means closing in (m/s).
pitch           Pitch attitude error (degrees).
pitch_rate      Pitch angular rate (°/s).

Action (Continuous, 6-D)
------------------------
Action vector in ``[-1, 1]^6``:

``[tx, ty, tz, roll, pitch, yaw]``

At each step, the component with the largest absolute value is selected as the
dominant command axis. If its magnitude exceeds a deadzone threshold, a single
corresponding simulator button click is issued in the sign direction.

Docking success
---------------
All of x, y, z, roll, roll_rate, range, yaw, yaw_rate, rate, pitch, pitch_rate
must be within ``SUCCESS_THRESHOLD`` (0.2) of zero simultaneously.
This mirrors the real simulator's criterion: every displayed reading must be
within ±0.2 (inclusive) before a successful dock is registered.

Out-of-range failure
--------------------
If ``range`` exceeds ``MAX_RANGE`` (350 m) the episode ends immediately with a
failure penalty.  This prevents the spacecraft from drifting uncontrollably far
from the ISS; the agent must keep range within bounds while closing in.

Attitude failure
----------------
If the absolute value of roll, yaw, or pitch exceeds ``MAX_ATTITUDE`` (80°)
the spacecraft is considered to have lost attitude control and the episode ends
immediately with ``REWARD_ATTITUDE_FAILURE``.

Reward shaping
--------------
Each step the agent receives:

* **Asymmetric range reward** — `+range_decrease × RANGE_REWARD_SCALE` when
  closing in, or `+range_decrease × RANGE_PENALTY_SCALE` when drifting away
  (``RANGE_PENALTY_SCALE > RANGE_REWARD_SCALE`` to strongly discourage retreat).
* **Position trend shaping** — encourage ``|x|+|y|+|z|`` to keep decreasing;
    worsening this sum receives a stronger penalty than the reward for improving.
* ``-n_warnings × WARNING_PENALTY`` — a small penalty for every non-range
  observation that still exceeds the 0.2 success threshold.
* ``REWARD_STEP`` — a fixed time penalty to encourage speed.
* **Angular-rate damping** — additional penalty proportional to
    ``|roll_rate| + |yaw_rate| + |pitch_rate|`` to actively suppress spin.
* **Control smoothness penalties** — extra penalties when thrusters are fired
    too often, when commands switch frequently, or when immediately reversing
    direction on the same axis.  These penalties are down-weighted in
    high-risk states so the policy can react faster when needed.
* Terminal bonuses/penalties on docking (`+100`), out-of-range (`−50`),
  attitude failure (`−50`), or collision (`−50`).
"""

import logging
import time
from typing import Any

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
    launch_browser:
        When ``True``, Playwright launches a Chromium browser automatically
        and navigates to the simulator URL.  When ``False`` (default),
        Playwright connects to an already-running Chrome instance via CDP.
    headless:
        Only used when ``launch_browser=True``.  Set to ``True`` to run the
        browser without a visible window (headless mode).
    cdp_url:
        Chrome DevTools Protocol endpoint URL used in CDP mode.  Pass a custom
        value if Chrome is not listening on the default port (9222).
        Ignored when ``launch_browser=True``.
    shared_browser_tabs:
        Only used when ``launch_browser=True``. If ``True``, environments in
        the same process reuse a single browser and open one tab per env.
    expected_shared_tabs:
        Expected number of managed tabs that should complete startup before
        training begins (used only when shared_browser_tabs is enabled).
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

    # Ordered list of action names (index → simulator control button).
    BUTTON_ACTIONS: list[str] = [
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

    # Continuous action components: [tx, ty, tz, roll, pitch, yaw].
    ACTION_DIM: int = 6
    ACTION_DEADZONE: float = 0.2

    # Axis index -> (positive-button-index, negative-button-index)
    AXIS_TO_BUTTON_INDEX: dict[int, tuple[int, int]] = {
        0: (0, 1),    # tx: forward/backward
        1: (2, 3),    # ty: up/down
        2: (4, 5),    # tz: left/right
        3: (7, 6),    # roll: + => roll_right, - => roll_left
        4: (8, 9),    # pitch: + => pitch_up, - => pitch_down
        5: (11, 10),  # yaw: + => yaw_right, - => yaw_left
    }
    OPPOSITE_ACTION_PAIRS: dict[int, int] = {
        0: 1, 1: 0,
        2: 3, 3: 2,
        4: 5, 5: 4,
        6: 7, 7: 6,
        8: 9, 9: 8,
        10: 11, 11: 10,
    }

    # Observation upper bounds used for clipping and normalisation.
    # Order mirrors OBS_KEYS:
    #   x, y, z, roll, roll_rate, range, yaw, yaw_rate, rate, pitch, pitch_rate
    OBS_KEYS: list[str] = [
        "x", "y", "z",
        "roll", "roll_rate",
        "range",
        "yaw", "yaw_rate",
        "rate",
        "pitch", "pitch_rate",
    ]
    OBS_HIGH: np.ndarray = np.array(
        [300.0, 300.0, 300.0, 180.0, 10.0, 500.0, 180.0, 10.0, 5.0, 180.0, 10.0],
        dtype=np.float32,
    )

    # Docking success: all readings must be within ±SUCCESS_THRESHOLD (inclusive).
    SUCCESS_THRESHOLD: float = 0.2

    # Out-of-range abort: if range exceeds this the episode ends immediately
    # as a failure (the spacecraft has drifted too far to recover).
    MAX_RANGE: float = 350.0   # metres

    # Attitude limit: if roll, yaw, or pitch exceeds this magnitude (degrees)
    # the spacecraft is considered to have lost control — episode fails immediately.
    MAX_ATTITUDE: float = 30.0   # degrees

    # Collision guard: approach rate (m/s, negative = closing) must not
    # exceed this magnitude when within NEAR_DISTANCE of the port.
    MAX_SAFE_RATE: float = 0.2   # m/s
    NEAR_DISTANCE: float = 5.0   # metres

    # Reward constants.
    REWARD_SUCCESS: float = 100.0
    REWARD_COLLISION: float = -50.0
    REWARD_OUT_OF_RANGE: float = -50.0    # penalty for drifting beyond MAX_RANGE
    REWARD_ATTITUDE_FAILURE: float = -50.0  # penalty for attitude limit breach
    REWARD_STEP: float = -0.01              # per-step time penalty

    # Reward shaping scales.
    # Points awarded per metre the spacecraft closes on the ISS.
    RANGE_REWARD_SCALE: float = 1.0
    # Penalty multiplier per metre the spacecraft moves away from the ISS.
    # Larger than RANGE_REWARD_SCALE to strongly discourage backing away.
    RANGE_PENALTY_SCALE: float = 2.0
    # Position-trend shaping for |x|+|y|+|z|.
    POSITION_REWARD_SCALE: float = 0.8
    POSITION_PENALTY_SCALE: float = 1.6
    # Penalty applied for each non-range observation that exceeds the 0.2
    # success threshold (i.e. each "warning" reading that is still too large).
    WARNING_PENALTY: float = 0.05
    # Penalty scale for angular-rate damping reward term.
    ANGULAR_RATE_DAMPING_PENALTY: float = 0.12
    ANGULAR_RATE_DAMPING_QUADRATIC_PENALTY: float = 0.4
    # Explicit danger-zone threshold for angular rates (deg/s).
    ANGULAR_RATE_DANGER_THRESHOLD: float = 0.5
    ANGULAR_RATE_DANGER_PENALTY: float = 0.08
    # Hard red-zone threshold: terminate episode immediately when exceeded.
    ANGULAR_RATE_REDZONE_THRESHOLD: float = 0.8
    REWARD_ANGULAR_RATE_REDZONE: float = -50.0
    # Approach-rate danger zone threshold (m/s, absolute value).
    RATE_DANGER_THRESHOLD: float = 0.2
    RATE_DANGER_PENALTY: float = 0.2

    # Speed-vs-distance conflict handling: avoid reducing range by excessive
    # closing speed. A dynamic safe closing rate is derived from range.
    AGGRESSIVE_CLOSING_PENALTY: float = 6.0
    MOVING_AWAY_RATE_PENALTY: float = 0.8

    # Global divergence penalty: whenever any metric moves farther from zero,
    # apply a penalty that grows with divergence magnitude.
    DIVERGENCE_LINEAR_PENALTY: float = 3.2
    DIVERGENCE_QUADRATIC_PENALTY: float = 8.0

    # Explicit progress reward: if total error decreases vs previous step,
    # reward the improvement even when still far from the target state.
    STEP_IMPROVEMENT_REWARD_SCALE: float = 1.5
    # Absolute distance-to-target penalty (always on): farther from target
    # incurs heavier penalties regardless of short-term improvement.
    ABSOLUTE_ERROR_LINEAR_PENALTY: float = 0.35
    ABSOLUTE_ERROR_QUADRATIC_PENALTY: float = 0.08
    # Improvement reward can offset only part of absolute-error penalty to
    # avoid conflict with "farther means harsher penalty" objective.
    IMPROVEMENT_REWARD_MAX_RATIO: float = 0.35

    # Control smoothness penalties.
    ACTUATION_PENALTY: float = 0.005
    ACTION_SWITCH_PENALTY: float = 0.01
    OPPOSITE_ACTION_PENALTY: float = 0.03
    # Small reward for restraint when no thruster is fired.
    OBSERVATION_PATIENCE_REWARD: float = 0.003

    # Action-bound shaping: only the executed action's mapped metric is scored.
    ACTION_TARGET_KEY_BY_BUTTON: dict[int, str] = {
        0: "range",
        1: "range",
        2: "y",
        3: "y",
        4: "z",
        5: "z",
        6: "roll",
        7: "roll",
        8: "pitch",
        9: "pitch",
        10: "yaw",
        11: "yaw",
    }
    ACTION_BINDING_PROGRESS_EPS: float = 1e-4
    ACTION_IMPROVEMENT_REWARD_SCALE: float = 3.0
    ACTION_WORSENING_LINEAR_PENALTY: float = 4.0
    ACTION_WORSENING_QUADRATIC_PENALTY: float = 16.0
    ACTION_WORSENING_HIGH_RISK_MULTIPLIER: float = 2.0
    ACTION_WORSENING_ATTITUDE_DANGER_MULTIPLIER: float = 1.8
    ACTION_WORSENING_RATE_DANGER_MULTIPLIER: float = 2.2
    ACTION_WORSENING_RATE_REDZONE_MULTIPLIER: float = 2.5

    # Trend shaping for x/y/z/range and roll/yaw/pitch.
    TREND_PROGRESS_EPS: float = 1e-4
    TREND_STAGNATION_GROWTH: float = 0.15
    POSITION_TREND_IMPROVEMENT_REWARD: float = 2.2
    POSITION_STAGNATION_PENALTY: float = 0.015
    POSITION_WORSENING_LINEAR_PENALTY: float = 7.0
    POSITION_WORSENING_QUADRATIC_PENALTY: float = 26.0
    POSITION_HOLD_REWARD: float = 0.001
    ATTITUDE_TREND_IMPROVEMENT_REWARD: float = 2.6
    ATTITUDE_STAGNATION_PENALTY: float = 0.02
    ATTITUDE_WORSENING_LINEAR_PENALTY: float = 8.5
    ATTITUDE_WORSENING_QUADRATIC_PENALTY: float = 28.0
    ATTITUDE_HOLD_REWARD: float = 0.0015

    # Rate target shaping.
    RATE_TARGET_ABS: float = 0.1
    RATE_TARGET_BAND: float = 0.02
    RATE_TARGET_LIMIT_ABS: float = 0.2
    RATE_ZERO_EPS: float = 0.01
    RATE_ZERO_PENALTY: float = 0.15
    RATE_TARGET_HOLD_REWARD: float = 0.015
    RATE_TARGET_APPROACH_REWARD_SCALE: float = 0.8
    RATE_TARGET_AWAY_PENALTY_SCALE: float = 1.0
    RATE_OVER_LIMIT_LINEAR_PENALTY: float = 4.0
    RATE_OVER_LIMIT_QUADRATIC_PENALTY: float = 14.0

    # Angular-rate redline shaping: >0.3 deg/s should be penalized heavily.
    ANGULAR_RATE_SOFT_REDLINE: float = 0.3
    ANGULAR_RATE_DECREASE_REWARD_SCALE: float = 0.7
    ANGULAR_RATE_INCREASE_PENALTY_SCALE: float = 1.2
    ANGULAR_RATE_REDLINE_BASE_PENALTY: float = 0.5
    ANGULAR_RATE_REDLINE_LINEAR_PENALTY: float = 5.0
    ANGULAR_RATE_REDLINE_QUADRATIC_PENALTY: float = 20.0

    # Keys tracked by stagnation counters.
    STAGNATION_KEYS: tuple[str, ...] = (
        "x", "y", "z", "roll", "pitch", "yaw", "range"
    )

    # Risk-weight bounds applied to smoothness penalties.
    # Low risk -> stronger smoothing; high risk -> more control authority.
    MIN_RISK_WEIGHT: float = 0.35
    MAX_RISK_WEIGHT: float = 1.0

    # Attitude axes monitored for the ±MAX_ATTITUDE hard limit.
    ATTITUDE_KEYS: tuple[str, ...] = ("roll", "yaw", "pitch")

    # Control cadence: execute one actual button press every N env steps.
    # Intermediate steps observe dynamics without applying new impulses.
    DEFAULT_CONTROL_INTERVAL_STEPS: int = 2
    DEFAULT_ACTION_CONFIRMATION_STEPS: int = 2

    # Adaptive-control thresholds.
    HIGH_AUTH_RANGE: float = 10.0
    MEDIUM_AUTH_RANGE: float = 50.0
    HIGH_AUTH_ATTITUDE: float = 12.0
    HIGH_AUTH_ANGULAR_RATE: float = ANGULAR_RATE_DANGER_THRESHOLD
    HIGH_AUTH_CLOSING_RATE: float = 0.15

    # Online action-effect estimation.
    # Tracks average delta(observation) per executed button.
    MIN_EFFECT_SAMPLES_FOR_CONFIDENCE: int = 20
    EFFECT_GUIDANCE_MARGIN: float = 0.01

    def __init__(
        self,
        launch_browser: bool = False,
        headless: bool = False,
        cdp_url: str = SimulatorBrowser.CDP_URL,
        shared_browser_tabs: bool = False,
        expected_shared_tabs: int | None = None,
        step_delay: float = 0.5,
        reset_wait: float = 5.0,
        control_interval_steps: int = DEFAULT_CONTROL_INTERVAL_STEPS,
        action_confirmation_steps: int = DEFAULT_ACTION_CONFIRMATION_STEPS,
        adaptive_control: bool = True,
        effect_guidance: bool = True,
        max_steps: int = 3000,
        render_mode=None,
    ) -> None:
        super().__init__()

        self.step_delay = step_delay
        self.reset_wait = reset_wait
        self.control_interval_steps = max(1, int(control_interval_steps))
        self.action_confirmation_steps = max(1, int(action_confirmation_steps))
        self.adaptive_control = adaptive_control
        self.effect_guidance = effect_guidance
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

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.ACTION_DIM,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-self.OBS_HIGH,
            high=self.OBS_HIGH,
            dtype=np.float32,
        )

        self._steps: int = 0
        self._prev_error: float = 0.0
        self._prev_range: float = 0.0
        self._prev_position_error: float = 0.0
        self._prev_abs_norm_obs: np.ndarray = np.zeros_like(self.OBS_HIGH)
        self._last_executed_action: int | None = None
        self._last_state: dict[str, float] | None = None
        self._steps_since_action: int = 0
        self._pending_button_index: int | None = None
        self._pending_button_count: int = 0
        self._action_effect_sum: np.ndarray = np.zeros(
            (len(self.BUTTON_ACTIONS), len(self.OBS_KEYS)),
            dtype=np.float32,
        )
        self._action_effect_count: np.ndarray = np.zeros(
            len(self.BUTTON_ACTIONS),
            dtype=np.int32,
        )
        self._obs_key_to_index: dict[str, int] = {
            key: idx for idx, key in enumerate(self.OBS_KEYS)
        }
        self._stagnation_counters: dict[str, int] = {
            key: 0 for key in self.STAGNATION_KEYS
        }
        # Pre-compute the observation keys used for warning detection (all
        # keys except "range", which has its own approach-reward logic).
        self._warning_keys: list[str] = [k for k in self.OBS_KEYS if k != "range"]

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Reload the simulator page and return the initial observation."""
        super().reset(seed=seed)
        self._browser.reset(wait=self.reset_wait)
        self._steps = 0
        obs = self._get_obs()
        state = self._obs_to_dict(obs)
        self._prev_error = self._total_error(obs)
        self._prev_range = state["range"]
        self._prev_position_error = self._position_error(state)
        self._prev_abs_norm_obs = np.abs(obs) / self.OBS_HIGH
        self._last_executed_action = None
        self._last_state = state
        self._steps_since_action = self._control_interval_for_state(state) - 1
        self._pending_button_index = None
        self._pending_button_count = 0
        for key in self._stagnation_counters:
            self._stagnation_counters[key] = 0
        return obs, {}

    def step(self, action: np.ndarray):
        """Apply *action* (button press), wait, then return the new state.

        Parameters
        ----------
        action:
            Continuous 6-D control vector in ``[-1, 1]``.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        action_vector = np.asarray(action, dtype=np.float32).reshape(-1)
        button_index, action_magnitude = self._map_continuous_action(action_vector)
        current_state_for_control = self._last_state or {
            key: 0.0 for key in self.OBS_KEYS
        }
        guided_override = False
        high_authority_state = self._is_high_authority_state(current_state_for_control)
        dynamic_control_interval = self._control_interval_for_state(current_state_for_control)
        required_confirmation_steps = 1 if high_authority_state else self.action_confirmation_steps

        if button_index is None:
            self._pending_button_index = None
            self._pending_button_count = 0
        elif button_index == self._pending_button_index:
            self._pending_button_count += 1
        else:
            self._pending_button_index = button_index
            self._pending_button_count = 1

        interval_ready = self._steps_since_action >= (dynamic_control_interval - 1)
        confirmation_ready = (
            button_index is not None
            and self._pending_button_index == button_index
            and self._pending_button_count >= required_confirmation_steps
        )
        executed_action = interval_ready and confirmation_ready

        if executed_action and button_index is not None:
            self._browser.click_action(self.BUTTON_ACTIONS[button_index])
            self._steps_since_action = 0
            self._pending_button_index = None
            self._pending_button_count = 0
        else:
            self._steps_since_action += 1
        time.sleep(self.step_delay)
        self._steps += 1

        obs = self._get_obs()
        state = self._obs_to_dict(obs)
        prev_state_for_effect = current_state_for_control
        error = self._total_error(obs)

        terminated = False
        truncated = False
        success = False

        reward, reward_terms = self._compute_reward_terms(
            obs=obs,
            state=state,
            prev_state_for_effect=prev_state_for_effect,
            executed_action=executed_action,
            button_index=button_index,
            error=error,
        )

        current_range = float(reward_terms["current_range"])
        current_position_error = float(reward_terms["current_position_error"])
        current_abs_norm_obs = reward_terms["current_abs_norm_obs"]
        overspeed = float(reward_terms["overspeed"])
        moving_away_speed = float(reward_terms["moving_away_speed"])

        angular_rate_redzone = bool(reward_terms["angular_rate_redzone"])
        rate_danger = bool(reward_terms["rate_danger"])

        risk_weight = self._risk_weight_for_state(state)

        if executed_action and button_index is not None:
            reward -= self.ACTUATION_PENALTY * risk_weight
            if self._last_executed_action is not None:
                if button_index != self._last_executed_action:
                    reward -= self.ACTION_SWITCH_PENALTY * risk_weight
                if self._is_opposite_action(button_index, self._last_executed_action):
                    reward -= self.OPPOSITE_ACTION_PENALTY * risk_weight
            self._last_executed_action = button_index

            prev_state_vec = np.array(
                [prev_state_for_effect[k] for k in self.OBS_KEYS],
                dtype=np.float32,
            )
            curr_state_vec = np.array([state[k] for k in self.OBS_KEYS], dtype=np.float32)
            state_delta = curr_state_vec - prev_state_vec
            self._action_effect_sum[button_index] += state_delta
            self._action_effect_count[button_index] += 1
        else:
            reward += self.OBSERVATION_PATIENCE_REWARD

        self._prev_error = error
        self._prev_range = current_range
        self._prev_position_error = current_position_error
        self._prev_abs_norm_obs = current_abs_norm_obs
        self._last_state = state

        # --- Termination checks (ordered from most severe to least) ----------

        if any(
            abs(state[k]) > self.MAX_ATTITUDE for k in self.ATTITUDE_KEYS
        ):
            # Attitude has exceeded ±80° — spacecraft lost control.
            reward += self.REWARD_ATTITUDE_FAILURE
            terminated = True
        elif angular_rate_redzone:
            # Angular rate exceeded hard red-zone threshold.
            reward += self.REWARD_ANGULAR_RATE_REDZONE
            terminated = True
        elif current_range > self.MAX_RANGE:
            # Spacecraft has drifted beyond 350 m — unrecoverable failure.
            reward += self.REWARD_OUT_OF_RANGE
            terminated = True
        elif self._is_docked(state):
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
            "executed_action": executed_action,
            "action_magnitude": float(action_magnitude),
            "button_index": -1 if button_index is None else int(button_index),
            "pending_button_count": int(self._pending_button_count),
            "required_confirmation_steps": int(required_confirmation_steps),
            "control_interval_steps": dynamic_control_interval,
            "risk_weight": float(risk_weight),
            "position_reward": float(reward_terms["position_reward"]),
            "absolute_error_penalty": float(reward_terms["absolute_error_penalty"]),
            "improvement_reward": float(reward_terms["improvement_reward"]),
            "raw_improvement_reward": float(reward_terms["raw_improvement_reward"]),
            "improvement_reward_cap": float(reward_terms["improvement_reward_cap"]),
            "error_improvement": float(reward_terms["error_improvement"]),
            "safe_closing_rate": float(reward_terms["safe_closing_rate"]),
            "closing_speed": float(reward_terms["closing_speed"]),
            "overspeed": float(overspeed),
            "aggressive_closing_penalty": float(reward_terms["aggressive_closing_penalty"]),
            "moving_away_penalty": float(reward_terms["moving_away_penalty"]),
            "divergence_penalty": float(reward_terms["divergence_penalty"]),
            "divergence_sum": float(reward_terms["divergence_sum"]),
            "action_state_worsening_penalty": float(reward_terms["action_state_worsening_penalty"]),
            "action_state_worsening_key": str(reward_terms["action_state_worsening_key"]),
            "action_state_worsening_norm_increase": float(reward_terms["action_state_worsening_norm_increase"]),
            "action_state_worsening_multiplier": float(reward_terms["action_state_worsening_multiplier"]),
            "action_state_improvement_reward": float(reward_terms["action_state_improvement_reward"]),
            "stagnation_penalty": float(reward_terms["stagnation_penalty"]),
            "target_hold_reward": float(reward_terms["target_hold_reward"]),
            "max_stagnation_steps": int(reward_terms["max_stagnation_steps"]),
            "attitude_reward": float(reward_terms["attitude_reward"]),
            "position_worsening_penalty": float(reward_terms["position_worsening_penalty"]),
            "attitude_worsening_penalty": float(reward_terms["attitude_worsening_penalty"]),
            "rate_target_reward": float(reward_terms["rate_target_reward"]),
            "rate_target_away_penalty": float(reward_terms["rate_target_away_penalty"]),
            "rate_zero_penalty": float(reward_terms["rate_zero_penalty"]),
            "rate_over_limit_penalty": float(reward_terms["rate_over_limit_penalty"]),
            "angular_rate_penalty": float(reward_terms["angular_rate_penalty"]),
            "angular_rate_quadratic_penalty": float(reward_terms["angular_rate_quadratic_penalty"]),
            "angular_rate_trend_reward": float(reward_terms["angular_rate_trend_reward"]),
            "angular_rate_redline_penalty": float(reward_terms["angular_rate_redline_penalty"]),
            "angular_rate_danger_count": int(reward_terms["angular_rate_danger_count"]),
            "angular_rate_redzone": bool(angular_rate_redzone),
            "rate_danger": bool(rate_danger),
            "range_reward": float(reward_terms["range_reward"]),
            "warning_count": int(reward_terms["warning_count"]),
            "moving_away_speed": float(moving_away_speed),
            "action_effect_samples": (
                int(self._action_effect_count[button_index])
                if button_index is not None
                else 0
            ),
            "action_effect_confident": (
                bool(
                    button_index is not None
                    and self._action_effect_count[button_index]
                    >= self.MIN_EFFECT_SAMPLES_FOR_CONFIDENCE
                )
            ),
            "guided_button_override": bool(guided_override),
            **state,
        }
        if button_index is not None and self._action_effect_count[button_index] > 0:
            info["estimated_action_effect"] = self._get_action_effect_vector(button_index).tolist()
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        """Disconnect from the browser and release resources."""
        self._browser.disconnect()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_action_state_binding_terms(
        self,
        state: dict[str, float],
        prev_state_for_effect: dict[str, float],
        executed_action: bool,
        button_index: int | None,
    ) -> tuple[float, float, str, float, float]:
        """Return action-bound improvement reward and worsening penalty."""
        if not (executed_action and button_index is not None):
            return 0.0, 0.0, "", 0.0, 1.0

        monitored_key = self.ACTION_TARGET_KEY_BY_BUTTON.get(button_index)
        if monitored_key is None:
            return 0.0, 0.0, "", 0.0, 1.0

        key_idx = self._obs_key_to_index[monitored_key]
        key_scale = float(self.OBS_HIGH[key_idx])
        prev_abs = abs(float(prev_state_for_effect[monitored_key]))
        curr_abs = abs(float(state[monitored_key]))
        signed_progress = (prev_abs - curr_abs) / max(key_scale, 1e-6)
        if abs(signed_progress) <= self.ACTION_BINDING_PROGRESS_EPS:
            return 0.0, 0.0, monitored_key, 0.0, 1.0

        effective_change = abs(signed_progress)

        risk_multiplier = 1.0
        if self._is_high_authority_state(state):
            risk_multiplier *= self.ACTION_WORSENING_HIGH_RISK_MULTIPLIER

        rotational_rate_key = {
            "roll": "roll_rate",
            "pitch": "pitch_rate",
            "yaw": "yaw_rate",
        }.get(monitored_key)

        if monitored_key in self.ATTITUDE_KEYS:
            if abs(float(state[monitored_key])) >= self.HIGH_AUTH_ATTITUDE:
                risk_multiplier *= self.ACTION_WORSENING_ATTITUDE_DANGER_MULTIPLIER

            if rotational_rate_key is not None:
                monitored_rate = abs(float(state[rotational_rate_key]))
                if monitored_rate >= self.ANGULAR_RATE_DANGER_THRESHOLD:
                    risk_multiplier *= self.ACTION_WORSENING_RATE_DANGER_MULTIPLIER
                if monitored_rate >= self.ANGULAR_RATE_REDZONE_THRESHOLD:
                    risk_multiplier *= self.ACTION_WORSENING_RATE_REDZONE_MULTIPLIER

        if signed_progress > 0.0:
            improvement_reward = (
                self.ACTION_IMPROVEMENT_REWARD_SCALE
                * signed_progress
                * risk_multiplier
            )
            return improvement_reward, 0.0, monitored_key, effective_change, risk_multiplier

        worsening_penalty = (
            self.ACTION_WORSENING_LINEAR_PENALTY * effective_change
            + self.ACTION_WORSENING_QUADRATIC_PENALTY * (effective_change ** 2)
        ) * risk_multiplier
        return 0.0, worsening_penalty, monitored_key, effective_change, risk_multiplier

    def _compute_reward_terms(
        self,
        obs: np.ndarray,
        state: dict[str, float],
        prev_state_for_effect: dict[str, float],
        executed_action: bool,
        button_index: int | None,
        error: float,
    ) -> tuple[float, dict[str, Any]]:
        """Compute reward and structured reward terms for diagnostics."""
        current_range = float(state["range"])
        current_position_error = self._position_error(state)
        current_abs_norm_obs = np.abs(obs) / self.OBS_HIGH

        closing_speed = max(-state["rate"], 0.0)
        moving_away_speed = max(state["rate"], 0.0)
        safe_closing_rate = min(
            self.MAX_SAFE_RATE,
            0.03 + 0.02 * np.sqrt(max(current_range, 0.0)),
        )
        overspeed = max(closing_speed - safe_closing_rate, 0.0)

        position_reward = 0.0
        range_reward = 0.0
        attitude_reward = 0.0
        position_worsening_penalty = 0.0
        attitude_worsening_penalty = 0.0
        stagnation_penalty = 0.0
        target_hold_reward = 0.0
        max_stagnation_steps = 0

        for key in ("x", "y", "z", "range"):
            curr_abs = abs(float(state[key]))
            prev_abs = abs(float(prev_state_for_effect[key]))
            key_scale = float(self.OBS_HIGH[self._obs_key_to_index[key]])
            delta_norm = (curr_abs - prev_abs) / max(key_scale, 1e-6)

            if curr_abs <= self.SUCCESS_THRESHOLD:
                self._stagnation_counters[key] = 0
                target_hold_reward += self.POSITION_HOLD_REWARD
                continue

            if delta_norm < -self.TREND_PROGRESS_EPS:
                self._stagnation_counters[key] = 0
                progress_reward = self.POSITION_TREND_IMPROVEMENT_REWARD * (-delta_norm)
                if key == "range":
                    range_reward += progress_reward
                else:
                    position_reward += progress_reward
                continue

            self._stagnation_counters[key] += 1
            stagnation_steps = self._stagnation_counters[key]
            max_stagnation_steps = max(max_stagnation_steps, stagnation_steps)
            growth_factor = 1.0 + self.TREND_STAGNATION_GROWTH * max(stagnation_steps - 1, 0)

            if delta_norm > self.TREND_PROGRESS_EPS:
                worsening_penalty = (
                    self.POSITION_WORSENING_LINEAR_PENALTY * delta_norm
                    + self.POSITION_WORSENING_QUADRATIC_PENALTY * (delta_norm ** 2)
                )
                position_worsening_penalty += worsening_penalty * growth_factor
            else:
                stagnation_penalty += self.POSITION_STAGNATION_PENALTY * growth_factor

        for key in ("roll", "yaw", "pitch"):
            curr_abs = abs(float(state[key]))
            prev_abs = abs(float(prev_state_for_effect[key]))
            key_scale = float(self.OBS_HIGH[self._obs_key_to_index[key]])
            delta_norm = (curr_abs - prev_abs) / max(key_scale, 1e-6)

            if curr_abs <= self.SUCCESS_THRESHOLD:
                self._stagnation_counters[key] = 0
                target_hold_reward += self.ATTITUDE_HOLD_REWARD
                continue

            if delta_norm < -self.TREND_PROGRESS_EPS:
                self._stagnation_counters[key] = 0
                attitude_reward += self.ATTITUDE_TREND_IMPROVEMENT_REWARD * (-delta_norm)
                continue

            self._stagnation_counters[key] += 1
            stagnation_steps = self._stagnation_counters[key]
            max_stagnation_steps = max(max_stagnation_steps, stagnation_steps)
            growth_factor = 1.0 + self.TREND_STAGNATION_GROWTH * max(stagnation_steps - 1, 0)

            if delta_norm > self.TREND_PROGRESS_EPS:
                worsening_penalty = (
                    self.ATTITUDE_WORSENING_LINEAR_PENALTY * delta_norm
                    + self.ATTITUDE_WORSENING_QUADRATIC_PENALTY * (delta_norm ** 2)
                )
                attitude_worsening_penalty += worsening_penalty * growth_factor
            else:
                stagnation_penalty += self.ATTITUDE_STAGNATION_PENALTY * growth_factor

        rate_abs = abs(float(state["rate"]))
        prev_rate_abs = abs(float(prev_state_for_effect["rate"]))
        rate_error = abs(rate_abs - self.RATE_TARGET_ABS)
        prev_rate_error = abs(prev_rate_abs - self.RATE_TARGET_ABS)

        rate_target_reward = 0.0
        rate_target_away_penalty = 0.0
        if rate_error <= self.RATE_TARGET_BAND:
            rate_target_reward += self.RATE_TARGET_HOLD_REWARD
        elif rate_error + self.TREND_PROGRESS_EPS < prev_rate_error:
            rate_target_reward += (
                self.RATE_TARGET_APPROACH_REWARD_SCALE
                * (prev_rate_error - rate_error)
            )
        elif rate_error > prev_rate_error + self.TREND_PROGRESS_EPS:
            rate_target_away_penalty += (
                self.RATE_TARGET_AWAY_PENALTY_SCALE
                * (rate_error - prev_rate_error)
            )

        rate_zero_penalty = self.RATE_ZERO_PENALTY if rate_abs <= self.RATE_ZERO_EPS else 0.0
        rate_over_limit_penalty = 0.0
        if rate_abs > self.RATE_TARGET_LIMIT_ABS:
            excess = rate_abs - self.RATE_TARGET_LIMIT_ABS
            rate_over_limit_penalty = (
                self.RATE_OVER_LIMIT_LINEAR_PENALTY * excess
                + self.RATE_OVER_LIMIT_QUADRATIC_PENALTY * (excess ** 2)
            )

        angular_rate_trend_reward = 0.0
        angular_rate_increase_penalty = 0.0
        angular_rate_redline_penalty = 0.0
        angular_rate_quadratic_penalty = 0.0
        angular_rate_danger_count = 0
        redline_denom = max(self.ANGULAR_RATE_SOFT_REDLINE, 1e-6)

        for rate_key in ("roll_rate", "yaw_rate", "pitch_rate"):
            curr_abs = abs(float(state[rate_key]))
            prev_abs = abs(float(prev_state_for_effect[rate_key]))
            delta_norm = (curr_abs - prev_abs) / redline_denom

            if delta_norm < -self.TREND_PROGRESS_EPS:
                angular_rate_trend_reward += self.ANGULAR_RATE_DECREASE_REWARD_SCALE * (-delta_norm)
            elif delta_norm > self.TREND_PROGRESS_EPS:
                angular_rate_increase_penalty += self.ANGULAR_RATE_INCREASE_PENALTY_SCALE * delta_norm

            if curr_abs > self.ANGULAR_RATE_SOFT_REDLINE:
                angular_rate_danger_count += 1
                excess_norm = (curr_abs - self.ANGULAR_RATE_SOFT_REDLINE) / redline_denom
                quadratic_term = self.ANGULAR_RATE_REDLINE_QUADRATIC_PENALTY * (excess_norm ** 2)
                angular_rate_redline_penalty += (
                    self.ANGULAR_RATE_REDLINE_BASE_PENALTY
                    + self.ANGULAR_RATE_REDLINE_LINEAR_PENALTY * excess_norm
                    + quadratic_term
                )
                angular_rate_quadratic_penalty += quadratic_term

        angular_rate_penalty = angular_rate_increase_penalty + angular_rate_redline_penalty

        (
            action_state_improvement_reward,
            action_state_worsening_penalty,
            action_state_key,
            action_state_change_norm,
            action_state_worsening_multiplier,
        ) = self._compute_action_state_binding_terms(
            state=state,
            prev_state_for_effect=prev_state_for_effect,
            executed_action=executed_action,
            button_index=button_index,
        )
        action_state_worsening_norm_increase = (
            action_state_change_norm if action_state_worsening_penalty > 0.0 else 0.0
        )

        warning_count = sum(
            1 for k in self._warning_keys if abs(state[k]) > self.SUCCESS_THRESHOLD
        )

        reward = self.REWARD_STEP
        reward += position_reward + range_reward + attitude_reward
        reward += rate_target_reward + angular_rate_trend_reward
        reward += target_hold_reward + action_state_improvement_reward
        reward -= position_worsening_penalty
        reward -= attitude_worsening_penalty
        reward -= stagnation_penalty
        reward -= rate_zero_penalty
        reward -= rate_target_away_penalty
        reward -= rate_over_limit_penalty
        reward -= angular_rate_penalty
        reward -= action_state_worsening_penalty

        absolute_error_penalty = 0.0
        error_improvement = 0.0
        raw_improvement_reward = 0.0
        improvement_reward_cap = 0.0
        improvement_reward = 0.0
        aggressive_closing_penalty = 0.0
        moving_away_penalty = 0.0
        divergence_penalty = 0.0
        divergence_sum = 0.0

        angular_rate_redzone = any(
            abs(state[key]) > self.ANGULAR_RATE_REDZONE_THRESHOLD
            for key in ("roll_rate", "yaw_rate", "pitch_rate")
        )
        rate_danger = rate_abs > self.RATE_DANGER_THRESHOLD

        return reward, {
            "current_range": float(current_range),
            "current_position_error": float(current_position_error),
            "current_abs_norm_obs": current_abs_norm_obs,
            "range_reward": float(range_reward),
            "warning_count": int(warning_count),
            "safe_closing_rate": float(safe_closing_rate),
            "closing_speed": float(closing_speed),
            "moving_away_speed": float(moving_away_speed),
            "overspeed": float(overspeed),
            "aggressive_closing_penalty": float(aggressive_closing_penalty),
            "moving_away_penalty": float(moving_away_penalty),
            "action_state_worsening_penalty": float(action_state_worsening_penalty),
            "action_state_worsening_key": action_state_key,
            "action_state_worsening_norm_increase": float(action_state_worsening_norm_increase),
            "action_state_worsening_multiplier": float(action_state_worsening_multiplier),
            "action_state_improvement_reward": float(action_state_improvement_reward),
            "stagnation_penalty": float(stagnation_penalty),
            "target_hold_reward": float(target_hold_reward),
            "max_stagnation_steps": int(max_stagnation_steps),
            "absolute_error_penalty": float(absolute_error_penalty),
            "error_improvement": float(error_improvement),
            "raw_improvement_reward": float(raw_improvement_reward),
            "improvement_reward_cap": float(improvement_reward_cap),
            "improvement_reward": float(improvement_reward),
            "position_reward": float(position_reward),
            "attitude_reward": float(attitude_reward),
            "position_worsening_penalty": float(position_worsening_penalty),
            "attitude_worsening_penalty": float(attitude_worsening_penalty),
            "rate_target_reward": float(rate_target_reward),
            "rate_target_away_penalty": float(rate_target_away_penalty),
            "rate_zero_penalty": float(rate_zero_penalty),
            "rate_over_limit_penalty": float(rate_over_limit_penalty),
            "divergence_penalty": float(divergence_penalty),
            "divergence_sum": float(divergence_sum),
            "angular_rate_penalty": float(angular_rate_penalty),
            "angular_rate_quadratic_penalty": float(angular_rate_quadratic_penalty),
            "angular_rate_trend_reward": float(angular_rate_trend_reward),
            "angular_rate_redline_penalty": float(angular_rate_redline_penalty),
            "angular_rate_danger_count": int(angular_rate_danger_count),
            "angular_rate_redzone": bool(angular_rate_redzone),
            "rate_danger": bool(rate_danger),
        }

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
    def _position_error(state: dict[str, float]) -> float:
        """Return |x|+|y|+|z| position error (metres)."""
        return abs(state["x"]) + abs(state["y"]) + abs(state["z"])

    @staticmethod
    def _is_opposite_action(current_action: int, previous_action: int) -> bool:
        """Return ``True`` when actions are opposite commands on the same axis."""
        return IssDockingEnv.OPPOSITE_ACTION_PAIRS.get(current_action) == previous_action

    def _map_continuous_action(self, action: np.ndarray) -> tuple[int | None, float]:
        """Map a continuous action vector to one simulator button index.

        Uses dominant-axis selection with a deadzone; returns ``(None, m)``
        when no button should be clicked.
        """
        if action.shape[0] != self.ACTION_DIM:
            raise ValueError(
                f"Expected action with shape ({self.ACTION_DIM},), got {action.shape}"
            )

        clipped = np.clip(action, -1.0, 1.0)
        dominant_axis = int(np.argmax(np.abs(clipped)))
        magnitude = float(abs(clipped[dominant_axis]))

        if magnitude < self.ACTION_DEADZONE:
            return None, magnitude

        positive_index, negative_index = self.AXIS_TO_BUTTON_INDEX[dominant_axis]
        button_index = positive_index if clipped[dominant_axis] >= 0 else negative_index
        return button_index, magnitude

    def _control_interval_for_state(self, state: dict[str, float]) -> int:
        """Return adaptive action interval based on current risk level."""
        if not self.adaptive_control:
            return self.control_interval_steps

        if self._is_high_authority_state(state):
            return 1

        if state["range"] <= self.MEDIUM_AUTH_RANGE:
            return max(1, self.control_interval_steps)

        return max(1, self.control_interval_steps + 1)

    def _is_high_authority_state(self, state: dict[str, float]) -> bool:
        """Return ``True`` when policy should be allowed high-frequency control."""
        if state["range"] <= self.HIGH_AUTH_RANGE:
            return True

        if any(abs(state[k]) >= self.HIGH_AUTH_ATTITUDE for k in self.ATTITUDE_KEYS):
            return True

        if any(
            abs(state[k]) >= self.HIGH_AUTH_ANGULAR_RATE
            for k in ("roll_rate", "yaw_rate", "pitch_rate")
        ):
            return True

        return abs(state["rate"]) >= self.HIGH_AUTH_CLOSING_RATE

    def _risk_weight_for_state(self, state: dict[str, float]) -> float:
        """Return smoothing-penalty multiplier from risk (high risk => lower)."""
        if not self.adaptive_control:
            return self.MAX_RISK_WEIGHT

        if self._is_high_authority_state(state):
            return self.MIN_RISK_WEIGHT

        if state["range"] <= self.MEDIUM_AUTH_RANGE:
            return 0.65

        return self.MAX_RISK_WEIGHT

    def _get_action_effect_vector(self, button_index: int) -> np.ndarray:
        """Return mean state delta vector for a button index."""
        samples = int(self._action_effect_count[button_index])
        if samples <= 0:
            return np.zeros(len(self.OBS_KEYS), dtype=np.float32)
        return self._action_effect_sum[button_index] / samples

    def _is_effect_confident(self, button_index: int) -> bool:
        """Return True when enough samples exist for a button's effect estimate."""
        return (
            int(self._action_effect_count[button_index])
            >= self.MIN_EFFECT_SAMPLES_FOR_CONFIDENCE
        )

    def _guided_button_override(
        self,
        base_button_index: int,
        state: dict[str, float],
    ) -> int:
        """Optionally override base button using learned effect map.

        In high-risk states, if the estimated effect of the chosen button would
        worsen the most critical metric and the opposite button is confidently
        estimated to improve it, switch to the opposite button.
        """
        if not self.effect_guidance:
            return base_button_index

        if not self._is_high_authority_state(state):
            return base_button_index

        opposite_button = self.OPPOSITE_ACTION_PAIRS.get(base_button_index)
        if opposite_button is None:
            return base_button_index

        if not (
            self._is_effect_confident(base_button_index)
            and self._is_effect_confident(opposite_button)
        ):
            return base_button_index

        state_norm = np.abs(
            np.array([state[k] for k in self.OBS_KEYS], dtype=np.float32)
        ) / self.OBS_HIGH
        critical_idx = int(np.argmax(state_norm))

        critical_key = self.OBS_KEYS[critical_idx]
        current_critical_value = float(state[critical_key])
        base_effect = self._get_action_effect_vector(base_button_index)
        opposite_effect = self._get_action_effect_vector(opposite_button)

        base_predicted_abs = (
            abs(current_critical_value + float(base_effect[critical_idx]))
            / float(self.OBS_HIGH[critical_idx])
        )
        opposite_predicted_abs = (
            abs(current_critical_value + float(opposite_effect[critical_idx]))
            / float(self.OBS_HIGH[critical_idx])
        )

        # If opposite action is predicted to bring the critical metric closer
        # to zero by a meaningful margin, use the opposite button.
        if base_predicted_abs > (opposite_predicted_abs + self.EFFECT_GUIDANCE_MARGIN):
            return opposite_button

        return base_button_index

    def get_action_effect_map(self) -> dict[str, dict[str, float]]:
        """Return learned button→state influence map as nested dictionaries."""
        effect_map: dict[str, dict[str, float]] = {}
        for idx, button_name in enumerate(self.BUTTON_ACTIONS):
            mean_delta = self._get_action_effect_vector(idx)
            effect_map[button_name] = {
                key: float(value) for key, value in zip(self.OBS_KEYS, mean_delta.tolist())
            }
        return effect_map

    def get_action_effect_counts(self) -> dict[str, int]:
        """Return sample counts collected for each button's effect estimate."""
        return {
            button_name: int(self._action_effect_count[idx])
            for idx, button_name in enumerate(self.BUTTON_ACTIONS)
        }

    def get_action_effect_summary(self) -> dict[str, Any]:
        """Return action-effect data bundle for export/visualization."""
        return {
            "obs_keys": list(self.OBS_KEYS),
            "button_actions": list(self.BUTTON_ACTIONS),
            "min_samples_for_confidence": int(self.MIN_EFFECT_SAMPLES_FOR_CONFIDENCE),
            "effects": self.get_action_effect_map(),
            "counts": self.get_action_effect_counts(),
        }

    @staticmethod
    def _is_docked(state: dict[str, float]) -> bool:
        """Return ``True`` when every reading is within the success threshold."""
        return all(abs(v) <= IssDockingEnv.SUCCESS_THRESHOLD for v in state.values())
