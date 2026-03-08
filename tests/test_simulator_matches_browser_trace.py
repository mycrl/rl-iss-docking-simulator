import unittest

import numpy as np

from environments.evaluate.browser import SimulatorBrowser
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
]


class TestSimulatorMatchesBrowserTrace(unittest.TestCase):
    def test_random_single_action_matches_browser(self) -> None:
        dt = 1.0
        compare_steps = 6
        rng = np.random.default_rng(20260307)

        sim = TrainDockingSimulator(dt=dt)
        browser = None

        try:
            browser = SimulatorBrowser(
                launch=True,
                headless=True,
            )

            browser.connect()
        except Exception as exc:
            self.skipTest(f"Live browser comparison unavailable: {exc}")

        initial_state = browser.read_state()
        sim.set_observable_state(initial_state)

        try:
            for step in range(compare_steps):
                action = np.zeros(6, dtype=np.int64)
                dim = int(rng.integers(0, 6))
                act_val = int(rng.choice([1, 2]))
                action[dim] = act_val

                sim.click_action(ACTION_MAP[dim][act_val])
                browser.click_action(ACTION_MAP[dim][act_val])
                
                sim_state = sim.read_state()
                browser_state = browser.read_state()

                for key in OBS_KEYS:
                    self.assertAlmostEqual(
                        float(sim_state[key]),
                        float(browser_state[key]),
                        delta=0.0,
                        msg=f"step={step} key={key} action_dim={dim} action_val={act_val}",
                    )
        finally:
            if browser is not None:
                browser.disconnect()


if __name__ == "__main__":
    unittest.main()
