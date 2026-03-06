import unittest

import numpy as np

from environments.evaluate.environment import EvalIssDockingEnv
from environments.train.simulator import TrainDockingSimulator


ACTION_MAP = {
    0: {1: "translate_forward", 2: "translate_backward"},
    1: {1: "translate_up", 2: "translate_down"},
    2: {1: "translate_right", 2: "translate_left"},
    3: {1: "roll_right", 2: "roll_left"},
    4: {1: "pitch_up", 2: "pitch_down"},
    5: {1: "yaw_right", 2: "yaw_left"},
}

OBS_KEYS = [
    "x",
    "y",
    "z",
    "roll",
    "roll_rate",
    "range",
    "yaw",
    "yaw_rate",
    "rate",
    "pitch",
    "pitch_rate",
    "fuel",
]

TOLERANCES = {
    "x": 20.0,
    "y": 20.0,
    "z": 20.0,
    "roll": 12.0,
    "pitch": 12.0,
    "yaw": 12.0,
    "roll_rate": 1.2,
    "pitch_rate": 1.2,
    "yaw_rate": 1.2,
    "range": 25.0,
    "rate": 0.4,
    "fuel": 0.05,
}


class TestSimulatorMatchesBrowserTrace(unittest.TestCase):
    def test_random_single_action_matches_browser(self) -> None:
        dt = 1.0
        compare_steps = 6
        rng = np.random.default_rng(20260307)

        sim = TrainDockingSimulator(dt=dt)
        env = None
        try:
            env = EvalIssDockingEnv(
                launch_browser=True,
                headless=True,
                step_delay=dt,
                reset_wait=3.0,
                max_steps=compare_steps + 2,
            )
            obs, _ = env.reset(seed=42)
        except Exception as exc:
            self.skipTest(f"Live browser comparison unavailable: {exc}")

        initial_state = dict(zip(OBS_KEYS, obs.tolist()))
        sim.set_observable_state(
            {
                "x": float(initial_state["x"]),
                "y": float(initial_state["y"]),
                "z": float(initial_state["z"]),
                "roll": float(initial_state["roll"]),
                "roll_rate": float(initial_state["roll_rate"]),
                "range": float(initial_state["range"]),
                "yaw": float(initial_state["yaw"]),
                "yaw_rate": float(initial_state["yaw_rate"]),
                "rate": float(initial_state["rate"]),
                "pitch": float(initial_state["pitch"]),
                "pitch_rate": float(initial_state["pitch_rate"]),
            },
            fuel_remaining=float(initial_state["fuel"]) * sim.INITIAL_FUEL,
        )

        try:
            for step in range(compare_steps):
                action = np.zeros(6, dtype=np.int64)
                dim = int(rng.integers(0, 6))
                act_val = int(rng.choice([1, 2]))
                action[dim] = act_val

                sim.click_action(ACTION_MAP[dim][act_val])
                _, _, terminated, truncated, info = env.step(action)
                sim_state = sim.read_state()

                sim_obs = dict(sim_state)
                sim_obs["fuel"] = sim.fuel_remaining / sim.INITIAL_FUEL

                browser_state = {
                    "x": float(info["x"]),
                    "y": float(info["y"]),
                    "z": float(info["z"]),
                    "roll": float(info["roll"]),
                    "roll_rate": float(info["roll_rate"]),
                    "range": float(info["range"]),
                    "yaw": float(info["yaw"]),
                    "yaw_rate": float(info["yaw_rate"]),
                    "rate": float(info["rate"]),
                    "pitch": float(info["pitch"]),
                    "pitch_rate": float(info["pitch_rate"]),
                    "fuel": float(info["fuel_remaining"]) / sim.INITIAL_FUEL,
                }

                for key in OBS_KEYS:
                    self.assertAlmostEqual(
                        float(sim_obs[key]),
                        float(browser_state[key]),
                        delta=float(TOLERANCES[key]),
                        msg=f"step={step} key={key} action_dim={dim} action_val={act_val}",
                    )

                if terminated or truncated:
                    break
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    unittest.main()
