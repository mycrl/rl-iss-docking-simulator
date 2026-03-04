# rl-iss-docking-simulator

A Dragon spacecraft ISS docking autonomous driving system built with a custom
[Gymnasium](https://gymnasium.farama.org/) environment and trained with
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) DQN.

The agent connects to the real [SpaceX ISS Docking Simulator](https://iss-sim.spacex.com/)
running in a Chrome browser, reads state data from the page DOM, and clicks
the control buttons to manoeuvre the Dragon spacecraft to a successful soft dock.

## Overview

The SpaceX ISS Docking Simulator presents a browser-based interface that
familiarises users with the controls used by NASA astronauts.  Successful
docking requires all six error readings (position offsets x/y/z and attitude
errors roll/pitch/yaw), the approach rate, and the range to all fall below 0.2.

### Environment

| Property | Value |
|---|---|
| Observation space | 8-D continuous — x, y, z (m), roll (°), range (m), yaw (°), rate (m/s), pitch (°) |
| Action space | Discrete (12) — one action per RCS button press |
| Step delay | 0.5 s (wait for physics to settle after each button press) |
| Max episode length | 3 000 steps |

**Actions**

| Index | Name |
|---|---|
| 0 | translate\_forward |
| 1 | translate\_backward |
| 2 | translate\_up |
| 3 | translate\_down |
| 4 | translate\_left |
| 5 | translate\_right |
| 6 | roll\_left |
| 7 | roll\_right |
| 8 | pitch\_up |
| 9 | pitch\_down |
| 10 | yaw\_left |
| 11 | yaw\_right |

**Episode termination conditions**

- ✅ **Success** — all readings (x, y, z, roll, range, yaw, rate, pitch) < 0.2
- 💥 **Collision** — approach rate < −0.2 m/s when within 5 m of the ISS
- ⏱ **Timeout** — 3 000 steps elapsed

**Reward shaping**

- `+100` for a successful docking
- `−50` for a collision
- `+ (previous_error − current_error)` progress reward each step
- `−0.01` per-step time penalty

### Algorithm

Deep Q-Network (**DQN**) from Stable-Baselines3 with an MLP policy.  Only one
simulator instance runs at a time, so training is strictly sequential.

## Project Structure

```
.
├── docking/
│   ├── __init__.py      # Package — exports IssDockingEnv
│   ├── browser.py       # Browser automation layer (CDP via Playwright)
│   └── environment.py   # Custom Gymnasium environment
├── train.py             # DQN training with checkpointing / resume
├── evaluate.py          # Deterministic model evaluation
├── environment.py       # Compatibility shim (imports from docking/)
├── requirements.txt     # Python dependencies
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
playwright install chromium
```

## Setup

### 1 — Configure CSS selectors

Open Chrome, navigate to the simulator, and press **F12** to open DevTools.
Use the Elements inspector to find the CSS selectors for each control button
and state readout, then fill them in `docking/browser.py`:

```python
BUTTON_SELECTORS = {
    "translate_forward":  "#translate-forward-button",  # example
    # … fill in all 12 entries …
}

STATE_SELECTORS = {
    "x":    "#x-error-number",  # example
    # … fill in all 8 entries …
}
```

### 2 — Launch Chrome with remote debugging

```bash
google-chrome --remote-debugging-port=9222 https://iss-sim.spacex.com/
```

On macOS:

```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
    --remote-debugging-port=9222 https://iss-sim.spacex.com/
```

## Usage

### Train

```bash
python train.py
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--timesteps` | 500 000 | Total training timesteps |
| `--model-path` | `models/dqn_docking` | Where to save the model |
| `--resume` | *(flag)* | Continue from an existing model |
| `--checkpoint-freq` | 10 000 | Steps between checkpoint saves |
| `--checkpoint-dir` | `checkpoints` | Directory for checkpoint files |

Example — resume a previous run:

```bash
python train.py --resume --model-path models/dqn_docking --timesteps 1000000
```

### Evaluate

```bash
python evaluate.py --model models/dqn_docking --episodes 10
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--model` | `models/dqn_docking` | Path to trained model |
| `--episodes` | 10 | Number of evaluation episodes |

## License

See [LICENSE](LICENSE).
