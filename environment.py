"""
Dragon Spacecraft ISS Docking Simulator Environment.

A custom Gymnasium environment that simulates the autonomous docking of a
SpaceX Dragon spacecraft with the International Space Station (ISS).

The Dragon starts some distance away along the docking axis and must navigate
to the ISS docking port using its Draco thruster system. The simulation uses
simplified Newtonian mechanics in 3D space.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class IssDockingEnv(gym.Env):
    """
    Gymnasium environment for Dragon spacecraft ISS docking.

    Observation space (6D continuous):
        - x, y: lateral offset from docking axis (metres)
        - z: distance along docking approach axis (metres, positive = away)
        - vx, vy: lateral velocity (m/s)
        - vz: approach velocity (m/s, negative = closing)

    Action space (3D continuous, [-1, 1]):
        - ax: thrust fraction along x-axis
        - ay: thrust fraction along y-axis
        - az: thrust fraction along z-axis (negative closes the gap)

    The episode ends when:
        - Successful docking: Dragon is within `dock_radius` metres and
          approach speed is below `max_dock_speed` m/s.
        - Crash: Dragon reaches the port with excessive speed.
        - Out of bounds: Dragon drifts beyond `max_distance` metres.
        - Timeout: `max_steps` steps elapsed without resolution.
    """

    metadata = {"render_modes": []}

    # Physical constants
    DRAGON_MASS = 12_000.0          # kg
    MAX_THRUST = 400.0              # N per axis (Draco thruster)
    DT = 0.5                        # seconds per simulation step

    # Episode parameters
    INITIAL_DISTANCE = 100.0        # m along approach axis at episode start
    INITIAL_OFFSET_RANGE = 5.0      # m random lateral offset
    INITIAL_SPEED_RANGE = 0.2       # m/s random initial velocity
    MAX_DISTANCE = 300.0            # m – out-of-bounds threshold
    MAX_LATERAL = 50.0              # m – max tolerated lateral offset
    MAX_SPEED = 10.0                # m/s – observation clamp

    # Docking success criteria
    DOCK_RADIUS = 0.5               # m – must be within this distance
    MAX_DOCK_SPEED = 0.3            # m/s – maximum safe docking speed

    # Reward shaping constants
    REWARD_SUCCESS = 100.0
    REWARD_CRASH = -50.0
    REWARD_OUT_OF_BOUNDS = -50.0
    REWARD_TIME_PENALTY = -0.01     # per step
    REWARD_LATERAL_SCALE = -0.005   # per metre of lateral offset per step
    REWARD_PROGRESS_SCALE = 0.1     # reward per metre of distance reduced

    def __init__(self, max_steps: int = 2000, render_mode=None):
        super().__init__()

        self.max_steps = max_steps
        self.render_mode = render_mode

        obs_high = np.array(
            [self.MAX_LATERAL, self.MAX_LATERAL, self.MAX_DISTANCE,
             self.MAX_SPEED, self.MAX_SPEED, self.MAX_SPEED],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        # Normalised thrust in each axis
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # State variables – initialised in reset()
        self._pos = np.zeros(3, dtype=np.float64)   # x, y, z (metres)
        self._vel = np.zeros(3, dtype=np.float64)   # vx, vy, vz (m/s)
        self._steps = 0
        self._prev_distance = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        rng = self.np_random

        # Dragon starts ~INITIAL_DISTANCE metres along the approach (z) axis
        # with small random lateral offsets and velocities.
        x = rng.uniform(-self.INITIAL_OFFSET_RANGE, self.INITIAL_OFFSET_RANGE)
        y = rng.uniform(-self.INITIAL_OFFSET_RANGE, self.INITIAL_OFFSET_RANGE)
        z = rng.uniform(
            self.INITIAL_DISTANCE * 0.8,
            self.INITIAL_DISTANCE * 1.2,
        )

        vx = rng.uniform(-self.INITIAL_SPEED_RANGE, self.INITIAL_SPEED_RANGE)
        vy = rng.uniform(-self.INITIAL_SPEED_RANGE, self.INITIAL_SPEED_RANGE)
        vz = rng.uniform(-self.INITIAL_SPEED_RANGE, self.INITIAL_SPEED_RANGE)

        self._pos = np.array([x, y, z], dtype=np.float64)
        self._vel = np.array([vx, vy, vz], dtype=np.float64)
        self._steps = 0
        self._prev_distance = float(np.linalg.norm(self._pos))

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        # Compute acceleration: F = m * a  →  a = F / m
        thrust = action * self.MAX_THRUST
        accel = thrust / self.DRAGON_MASS

        # Save position before integration to detect port crossing
        prev_pos = self._pos.copy()
        prev_z = prev_pos[2]

        # Integrate: semi-implicit Euler
        self._vel += accel * self.DT
        self._pos += self._vel * self.DT
        self._steps += 1

        speed = float(np.linalg.norm(self._vel))

        terminated = False
        truncated = False

        # --- Detect docking port crossing (overshoot guard) ---
        # If the spacecraft flew past z=0 in one step (prev_z > 0, z ≤ 0),
        # evaluate the docking condition at the interpolated crossing point.
        denom = prev_z - self._pos[2]
        if prev_z > 0.0 and self._pos[2] <= 0.0 and denom > 1e-12:
            # Linear interpolation: fraction of DT when z == 0
            t_frac = prev_z / denom
            cross_x = prev_pos[0] + t_frac * (self._pos[0] - prev_pos[0])
            cross_y = prev_pos[1] + t_frac * (self._pos[1] - prev_pos[1])
            # Compare squared distance to avoid unnecessary sqrt
            cross_lateral_sq = cross_x**2 + cross_y**2
            if cross_lateral_sq <= self.DOCK_RADIUS**2:
                # The spacecraft flew through the port face
                cross_lateral = float(np.sqrt(cross_lateral_sq))
                distance = cross_lateral
                lateral = cross_lateral
            else:
                distance = float(np.linalg.norm(self._pos))
                lateral = float(np.linalg.norm(self._pos[:2]))
        else:
            distance = float(np.linalg.norm(self._pos))
            lateral = float(np.linalg.norm(self._pos[:2]))

        reward = self.REWARD_TIME_PENALTY

        # Lateral offset penalty (encourages staying on the docking axis)
        reward += self.REWARD_LATERAL_SCALE * lateral

        # Progress reward: reward closing in on the target
        progress = self._prev_distance - distance
        reward += self.REWARD_PROGRESS_SCALE * progress
        self._prev_distance = distance

        # Check terminal conditions
        success = False
        if distance <= self.DOCK_RADIUS:
            if speed <= self.MAX_DOCK_SPEED:
                # Successful docking
                reward += self.REWARD_SUCCESS
                terminated = True
                success = True
            else:
                # Collision – too fast
                reward += self.REWARD_CRASH
                terminated = True

        elif (
            distance > self.MAX_DISTANCE
            or lateral > self.MAX_LATERAL
        ):
            reward += self.REWARD_OUT_OF_BOUNDS
            terminated = True

        elif self._steps >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {
            "distance": distance,
            "lateral_offset": lateral,
            "speed": speed,
            "steps": self._steps,
            "success": success,
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Return the current observation vector (float32)."""
        obs = np.concatenate([self._pos, self._vel]).astype(np.float32)
        return np.clip(
            obs,
            self.observation_space.low,
            self.observation_space.high,
        )
