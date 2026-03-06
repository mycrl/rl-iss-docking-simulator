"""Local spacecraft dynamics used by the training environment."""

import math
from typing import Any

import numpy as np


class TrainDockingSimulator:
    """Stateful physics simulator for Dragon docking dynamics."""

    INITIAL_FUEL: float = 800.0
    FUEL_PER_BUTTON: float = 1.0

    HARD_START_PROB: float = 0.35
    TRANSLATION_EFFECT_DELAY_STEPS: int = 3
    TRANSLATION_OBSERVE_WINDOW_STEPS: int = 4
    TRANSLATION_FIRST_PULSE_SCALE: float = 0.6
    TRANSLATION_SECOND_PULSE_SCALE: float = 0.85
    TRANSLATION_REVERSE_SCALE: float = 0.5

    TRANSLATION_PULSE: float = 0.06
    ROTATION_PULSE: float = 0.1

    ACTION_TO_DIM: dict[str, int] = {
        "noop": -1,
        "translate_forward": 0,
        "translate_backward": 0,
        "translate_up": 1,
        "translate_down": 1,
        "translate_right": 2,
        "translate_left": 2,
        "roll_right": 3,
        "roll_left": 3,
        "pitch_up": 4,
        "pitch_down": 4,
        "yaw_right": 5,
        "yaw_left": 5,
    }

    def __init__(self, dt: float) -> None:
        self.dt = float(dt)
        self.state_vars: dict[str, float] = {}
        self.fuel_used: int = 0
        self.fuel_remaining: float = self.INITIAL_FUEL

        self.translation_pending: list[tuple[int, np.ndarray, int]] = []
        self.translation_last_command_step = np.full(3, -10_000, dtype=np.int32)
        self.translation_last_command_value = np.zeros(3, dtype=np.int8)
        self.translation_command_streak = np.zeros(3, dtype=np.int16)

        self.active_dims: set[int] = set()
        self.quick_repeat_translation_dims: set[int] = set()
        self.flip_translation_dims: set[int] = set()
        self.button_presses: int = 0
        self._step_idx: int = 0
        self._step_open: bool = False

    def reset(self, np_random: Any) -> None:
        self.fuel_used = 0
        self.fuel_remaining = self.INITIAL_FUEL

        self.translation_pending.clear()
        self.translation_last_command_step.fill(-10_000)
        self.translation_last_command_value.fill(0)
        self.translation_command_streak.fill(0)

        self._clear_step_flags()
        self._step_idx = 0
        self._step_open = False

        hard_start = bool(np_random.random() < self.HARD_START_PROB)

        if hard_start:
            x = np_random.uniform(90.0, 260.0)
            y = np_random.uniform(-90.0, 90.0)
            z = np_random.uniform(-90.0, 90.0)
            vx = np_random.uniform(-0.12, 0.08)
            vy = np_random.uniform(-0.08, 0.08)
            vz = np_random.uniform(-0.08, 0.08)

            roll = np_random.uniform(-25.0, 25.0)
            pitch = np_random.uniform(-25.0, 25.0)
            yaw = np_random.uniform(-25.0, 25.0)
            roll_rate = np_random.uniform(-0.7, 0.7)
            pitch_rate = np_random.uniform(-0.7, 0.7)
            yaw_rate = np_random.uniform(-0.7, 0.7)
        else:
            x = np_random.uniform(170.0, 230.0)
            y = np_random.uniform(-25.0, 25.0)
            z = np_random.uniform(-25.0, 25.0)
            vx = np_random.uniform(-0.05, 0.03)
            vy = np_random.uniform(-0.03, 0.03)
            vz = np_random.uniform(-0.03, 0.03)

            roll = np_random.uniform(-18.0, 18.0)
            pitch = np_random.uniform(-18.0, 18.0)
            yaw = np_random.uniform(-18.0, 18.0)
            roll_rate = np_random.uniform(-0.3, 0.3)
            pitch_rate = np_random.uniform(-0.3, 0.3)
            yaw_rate = np_random.uniform(-0.3, 0.3)

        self.state_vars = {
            "x": float(x),
            "vx": float(vx),
            "y": float(y),
            "vy": float(vy),
            "z": float(z),
            "vz": float(vz),
            "roll": float(roll),
            "roll_rate": float(roll_rate),
            "pitch": float(pitch),
            "pitch_rate": float(pitch_rate),
            "yaw": float(yaw),
            "yaw_rate": float(yaw_rate),
        }

        self.state_vars["range"] = math.sqrt(x**2 + y**2 + z**2)
        if self.state_vars["range"] > 1e-6:
            radial_velocity = (x * vx + y * vy + z * vz) / self.state_vars["range"]
        else:
            radial_velocity = 0.0
        self.state_vars["rate"] = float(radial_velocity)

    def set_observable_state(
        self,
        state: dict[str, float],
        fuel_remaining: float | None = None,
    ) -> None:
        """Set simulator state from browser-observable fields.

        Browser state does not expose full linear velocity components. We
        reconstruct a radial-consistent velocity from ``rate`` and
        position direction so replay tests can start from a fixed browser
        snapshot.
        """
        x = float(state["x"])
        y = float(state["y"])
        z = float(state["z"])
        r = math.sqrt(x**2 + y**2 + z**2)
        rate = float(state["rate"])

        if r > 1e-8:
            ux, uy, uz = x / r, y / r, z / r
            vx = rate * ux
            vy = rate * uy
            vz = rate * uz
        else:
            vx = 0.0
            vy = 0.0
            vz = 0.0

        self.state_vars = {
            "x": x,
            "vx": vx,
            "y": y,
            "vy": vy,
            "z": z,
            "vz": vz,
            "roll": float(state["roll"]),
            "roll_rate": float(state["roll_rate"]),
            "pitch": float(state["pitch"]),
            "pitch_rate": float(state["pitch_rate"]),
            "yaw": float(state["yaw"]),
            "yaw_rate": float(state["yaw_rate"]),
            "range": float(state["range"]),
            "rate": rate,
        }

        self.translation_pending.clear()
        self.translation_last_command_step.fill(-10_000)
        self.translation_last_command_value.fill(0)
        self.translation_command_streak.fill(0)
        self._clear_step_flags()
        self._step_idx = 0
        self._step_open = False

        if fuel_remaining is not None:
            self.fuel_remaining = max(0.0, float(fuel_remaining))
            self.fuel_used = int(round(self.INITIAL_FUEL - self.fuel_remaining))
        else:
            self.fuel_remaining = self.INITIAL_FUEL
            self.fuel_used = 0

    def click_action(self, action_name: str) -> None:
        if action_name not in self.ACTION_TO_DIM:
            raise ValueError(f"Unsupported action '{action_name}'.")
        if action_name == "noop":
            if not self._step_open:
                self._clear_step_flags()
                self._step_open = True
            return

        if not self._step_open:
            self._clear_step_flags()
            self._step_open = True

        dim = self.ACTION_TO_DIM[action_name]
        self.active_dims.add(dim)
        self.button_presses += 1

        if action_name in ("translate_forward", "translate_backward"):
            self._apply_translation(axis_idx=0, act_val=1 if action_name.endswith("forward") else 2)
        elif action_name in ("translate_up", "translate_down"):
            self._apply_translation(axis_idx=1, act_val=1 if action_name.endswith("up") else 2)
        elif action_name in ("translate_right", "translate_left"):
            self._apply_translation(axis_idx=2, act_val=1 if action_name.endswith("right") else 2)
        elif action_name == "roll_right":
            self.state_vars["roll_rate"] += self.ROTATION_PULSE
        elif action_name == "roll_left":
            self.state_vars["roll_rate"] -= self.ROTATION_PULSE
        elif action_name == "pitch_up":
            self.state_vars["pitch_rate"] -= self.ROTATION_PULSE
        elif action_name == "pitch_down":
            self.state_vars["pitch_rate"] += self.ROTATION_PULSE
        elif action_name == "yaw_right":
            self.state_vars["yaw_rate"] += self.ROTATION_PULSE
        elif action_name == "yaw_left":
            self.state_vars["yaw_rate"] -= self.ROTATION_PULSE

        fuel_spent = self.FUEL_PER_BUTTON
        self.fuel_used += int(fuel_spent)
        self.fuel_remaining = max(0.0, self.fuel_remaining - fuel_spent)

    def integrate(self) -> None:
        pending_next: list[tuple[int, np.ndarray, int]] = []
        for wait_steps, delta_v, axis_idx in self.translation_pending:
            if wait_steps <= 0:
                self.state_vars["vx"] += float(delta_v[0])
                self.state_vars["vy"] += float(delta_v[1])
                self.state_vars["vz"] += float(delta_v[2])
            else:
                pending_next.append((wait_steps - 1, delta_v, axis_idx))
        self.translation_pending = pending_next

        old_range = self.state_vars["range"]

        self.state_vars["x"] += self.state_vars["vx"] * self.dt
        self.state_vars["y"] += self.state_vars["vy"] * self.dt
        self.state_vars["z"] += self.state_vars["vz"] * self.dt

        self.state_vars["roll"] += self.state_vars["roll_rate"] * self.dt
        self.state_vars["pitch"] -= self.state_vars["pitch_rate"] * self.dt
        self.state_vars["yaw"] -= self.state_vars["yaw_rate"] * self.dt

        new_range = math.sqrt(
            self.state_vars["x"]**2 + self.state_vars["y"]**2 + self.state_vars["z"]**2
        )
        self.state_vars["range"] = new_range
        self.state_vars["rate"] = (new_range - old_range) / self.dt

    def read_state(self) -> dict[str, float]:
        # read_state is the driver: each call advances one simulator step.
        if not self._step_open:
            self._clear_step_flags()
        self.integrate()
        self._step_idx += 1
        self._step_open = False
        return dict(self.state_vars)

    def get_state_snapshot(self) -> dict[str, float]:
        """Return state without advancing simulation time."""
        return dict(self.state_vars)

    def _apply_translation(self, axis_idx: int, act_val: int) -> None:
        since_last_cmd = self._step_idx - int(self.translation_last_command_step[axis_idx])
        prev_cmd = int(self.translation_last_command_value[axis_idx])

        if 0 <= since_last_cmd <= self.TRANSLATION_OBSERVE_WINDOW_STEPS:
            self.quick_repeat_translation_dims.add(axis_idx)

        recent_flip = (
            prev_cmd in (1, 2)
            and prev_cmd != act_val
            and 0 <= since_last_cmd <= self.TRANSLATION_OBSERVE_WINDOW_STEPS
        )
        if recent_flip:
            self.flip_translation_dims.add(axis_idx)

        if prev_cmd == act_val:
            self.translation_command_streak[axis_idx] += 1
        else:
            self.translation_command_streak[axis_idx] = 1

        self.translation_last_command_step[axis_idx] = self._step_idx
        self.translation_last_command_value[axis_idx] = act_val

        pulse_scale = 1.0
        if self.translation_command_streak[axis_idx] == 1:
            pulse_scale *= self.TRANSLATION_FIRST_PULSE_SCALE
        elif self.translation_command_streak[axis_idx] == 2:
            pulse_scale *= self.TRANSLATION_SECOND_PULSE_SCALE
        if recent_flip:
            pulse_scale *= self.TRANSLATION_REVERSE_SCALE

        direction = 1 if act_val == 1 else -1
        if axis_idx == 0:
            body_vec = np.array([-direction, 0.0, 0.0], dtype=np.float32)
        elif axis_idx == 1:
            body_vec = np.array([0.0, 0.0, direction], dtype=np.float32)
        else:
            body_vec = np.array([0.0, direction, 0.0], dtype=np.float32)

        world_vec = self._body_to_world(
            body_vec,
            roll_deg=self.state_vars["roll"],
            pitch_deg=self.state_vars["pitch"],
            yaw_deg=self.state_vars["yaw"],
        )
        delta_v = world_vec * (self.TRANSLATION_PULSE * pulse_scale)
        self.translation_pending.append(
            (
                self.TRANSLATION_EFFECT_DELAY_STEPS,
                delta_v.astype(np.float32),
                axis_idx,
            )
        )

    def _clear_step_flags(self) -> None:
        self.active_dims.clear()
        self.quick_repeat_translation_dims.clear()
        self.flip_translation_dims.clear()
        self.button_presses = 0

    @staticmethod
    def _body_to_world(
        body_vec: np.ndarray,
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
    ) -> np.ndarray:
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)

        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        rx = np.array(
            [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
            dtype=np.float32,
        )
        ry = np.array(
            [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
            dtype=np.float32,
        )
        rz = np.array(
            [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        return (rz @ ry @ rx) @ body_vec
