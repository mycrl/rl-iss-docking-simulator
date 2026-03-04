# rl-iss-docking-simulator

A Dragon spacecraft ISS docking autonomous driving system built with a custom
[Gymnasium](https://gymnasium.farama.org/) environment and trained with
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) SAC.

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
| Observation space | 11-D continuous — x, y, z (m), roll (°), roll rate (°/s), range (m), yaw (°), yaw rate (°/s), rate (m/s), pitch (°), pitch rate (°/s) |
| Action space | Continuous Box(6) — `[tx, ty, tz, roll, pitch, yaw]` in `[-1,1]` |
| Step delay | 0.5 s (wait for physics to settle after each button press) |
| Max episode length | 3 000 steps |

**Actions**

The policy outputs a continuous 6-D command vector. The environment maps the
dominant component to one simulator thruster button click (with deadzone and
adaptive control authority to reduce over-control in high-inertia dynamics).

**Episode termination conditions**

- ✅ **Success** — all 11 readings are within ±0.2
- 🧭 **Attitude limit** — `|roll|` or `|yaw|` or `|pitch|` > 30°
- 🔴 **Angular-rate red zone** — `|roll_rate|` or `|yaw_rate|` or `|pitch_rate|` > 0.8 °/s
- 🚫 **Out of range** — range > 350 m
- 💥 **Collision** — approach rate < −0.2 m/s when within 5 m of the ISS
- ⏱ **Timeout** — 3 000 steps elapsed

**Reward shaping**

- `+100` for a successful docking
- `−50` for terminal failures (collision / out-of-range / attitude / red-zone)
- Asymmetric range shaping (closing-in rewarded, drifting-away penalized more)
- Position trend shaping for `|x|+|y|+|z|` (worsening penalized more than improvement rewarded)
- Angular-rate damping penalty and danger-zone penalties (`|rate| > 0.2`, angular-rate danger > 0.5)
- Control smoothness penalties + small observation patience reward
- `−0.01` per-step time penalty

### Algorithm

Soft Actor-Critic (**SAC**) from Stable-Baselines3 with an MLP policy.  Only
one simulator instance runs at a time, so training is strictly sequential.

## Project Structure

```
.
├── docking/
│   ├── __init__.py      # Package — exports IssDockingEnv
│   ├── browser.py       # Browser automation layer (CDP via Playwright)
│   └── environment.py   # Custom Gymnasium environment
├── train.py             # SAC training with checkpointing / resume
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

### 1 — Choose a browser mode

There are two ways to connect to the simulator:

**Option A — Managed mode (Playwright launches the browser)**

Pass `--launch-browser` to any script.  Playwright starts a Chromium browser,
navigates to the simulator automatically, and closes it on exit.  No manual
Chrome setup is needed:

```bash
python train.py --launch-browser
```

In managed mode, the first page load asks for a one-time confirmation in terminal
(`y` + Enter). Later episode resets are fully automatic.

**Option B — CDP mode (connect to a manually-opened Chrome)**

Start Chrome with remote debugging enabled and navigate to the simulator:

```bash
google-chrome --remote-debugging-port=9222 https://iss-sim.spacex.com/
```

On macOS:

```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
    --remote-debugging-port=9222 https://iss-sim.spacex.com/
```

Then run the scripts without `--launch-browser` (the default):

```bash
python train.py
```

On Windows (PowerShell), launch Chrome with remote debugging like this:

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 https://iss-sim.spacex.com/
```

## Usage

### Train

```bash
# Managed mode — browser launched automatically
python train.py --launch-browser

# CDP mode — connect to a manually-opened Chrome (default)
python train.py
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--launch-browser` | *(flag)* | Let Playwright launch Chromium automatically |
| `--headless` | *(flag)* | Run browser without a visible window (with `--launch-browser`) |
| `--timesteps` | 500 000 | Total training timesteps |
| `--model-path` | `models/sac_docking` | Where to save the model |
| `--resume` | *(flag)* | Continue from an existing model |
| `--checkpoint-freq` | 10 000 | Steps between checkpoint saves |
| `--checkpoint-dir` | `checkpoints` | Directory for checkpoint files |
| `--control-interval-steps` | 2 | Base interval between real control clicks |
| `--action-confirmation-steps` | 2 | Consecutive same-intent steps required before click (non-high-risk) |
| `--adaptive-control` / `--no-adaptive-control` | enabled | Enable/disable risk-adaptive control authority |

Example — resume a previous run in managed mode:

```bash
python train.py --launch-browser --resume --model-path models/sac_docking --timesteps 1000000
```

### Evaluate

```bash
# Managed mode
python evaluate.py --launch-browser --model models/sac_docking --episodes 10

# CDP mode
python evaluate.py --model models/sac_docking --episodes 10
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--launch-browser` | *(flag)* | Let Playwright launch Chromium automatically |
| `--headless` | *(flag)* | Run browser without a visible window (with `--launch-browser`) |
| `--model` | `models/sac_docking` | Path to trained model |
| `--episodes` | 10 | Number of evaluation episodes |

## License

See [LICENSE](LICENSE).
