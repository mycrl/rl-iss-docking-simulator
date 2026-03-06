# Autonomous Dragon Spaceship

A Dragon spacecraft ISS docking autonomous driving system built with a custom
[Gymnasium](https://gymnasium.farama.org/) environment and trained with
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) PPO.

The agent trains using a pure-Python simulation of the SpaceX ISS Docking
Simulator, and when evaluated, connects to the real
[SpaceX ISS Docking Simulator](https://iss-sim.spacex.com/) running in a
Chrome browser to demonstrate the precise docking manoeuvre visually.

## Overview

The browser-based simulator is too slow for large-scale RL training due to
real-time constraints and rendering overhead. This repo uses two environments:

1. `TrainIssDockingEnv` (training): local physics simulation in Python.
2. `EvalIssDockingEnv` (evaluation): browser-backed environment using Playwright.

Both environments share the same observation/action interface.

## Environment Interface

| Property | Value |
|---|---|
| Observation space | 12-D continuous: `x,y,z,roll,roll_rate,range,yaw,yaw_rate,rate,pitch,pitch_rate,fuel` |
| Action space | `MultiDiscrete([3, 3, 3, 3, 3, 3])` |
| Action semantics | per dimension: `0=noop`, `1=positive`, `2=negative` |
| Fuel model | `INITIAL_FUEL=800`, `FUEL_PER_BUTTON=1` |
| Observation normalization | `VecNormalize` during train/eval scripts |

**Actions**

The policy outputs an array of 6 discrete values, one for each degree of freedom
(Translation: X, Y, Z / Rotation: Roll, Pitch, Yaw). For each dimension:
- `0`: NO_OP
- `1`: Positive control (e.g. forward, up, right, roll_right, pitch_up, yaw_right)
- `2`: Negative control (e.g. backward, down, left, roll_left, pitch_down, yaw_left)

This allows simultaneous multi-axis commands in one step.

## Termination Rules

Common thresholds used by both environments:

- Success (`_is_docked`):
	- `|x|,|y|,|z| <= 0.2`
	- `|roll|,|pitch|,|yaw| <= 0.2`
	- `|roll_rate|,|pitch_rate|,|yaw_rate| <= 0.25`
	- `0.0 <= rate <= 0.2`
	- `range < 2.0`
- Failure:
	- fuel exhausted
	- `rate > 0.8`
	- `range > 350`
	- any of `|roll|,|pitch|,|yaw| > 30`
- Truncation:
	- if `max_steps` is set and reached

## Reward (Training Environment)

`EvalIssDockingEnv` is evaluation-only and always returns reward `0.0`.

`TrainIssDockingEnv` uses dense shaping in `environments/train/environment.py`.
Final reward is the sum of these components:

- Action/fuel usage:
	- `-0.03` per active control dimension in the current step.
- Translation behavior penalties:
	- quick repeat penalty: `-0.12` (`translation_quick_repeat_dim*`)
	- direction flip penalty: `-0.16` (`translation_flip_dim*`)
- Local progress credit (metric-based):
	- per dimension, only mapped metrics receive credit.
	- progress score per metric: `clip(improvement * weight, -0.8, 0.8)`.
	- translation active metrics are softened by `0.35`; rotation keeps `1.0`.
	- repeated same-direction active command while already improving is penalized:
		- translation: `-0.08`
		- rotation: `-0.12`
	- ineffective active rotation metric (`improvement <= 0`) gets `-0.06`.
- Noop/hold behavior:
	- positive hold reward: `clip(improvement * 0.5, 0.0, 0.25)`
	- translation observe-window bonus (after recent translation command):
		`clip(improvement * 0.8, 0.0, 0.2)`
	- lazy penalty on violation without improvement:
		`-clip((violation + (-improvement)) * 0.35, 0.0, 0.25)`
- Safety shaping:
	- near overspeed (`range < 5` and `rate > 0.2`): `-10`
	- reverse rate (`rate < 0`): `-(((-rate) * 30)^2)`
	- under-speed (`0 <= rate < 0.02`): `-(((0.02-rate) * 20)^2)`
	- over-speed (`rate > 0.2`): `-(((rate-0.2) * 30)^2)`
	- far stagnation (`range > 15` and `0 <= rate < 0.02`): `-0.1`
	- angular target band:
		- per axis target rate from attitude:
			- roll target = `clip(roll * -0.02, -0.25, 0.25)`
			- pitch target = `clip(pitch * 0.02, -0.25, 0.25)`
			- yaw target = `clip(yaw * 0.02, -0.25, 0.25)`
		- penalty per axis: `-abs(rate - target) * 0.8`
	- spin overspeed (`|*_rate| > 0.25`): `-(((abs(rate)-0.25) * 12)^2)`
- Terminal rewards/penalties:
	- success: `+1000`
	- fuel empty: `-300`
	- terminal overspeed (`rate > 0.8`): `-500`
	- out of range (`range > 350`): `-1000`
	- attitude limit (`|roll|` or `|pitch|` or `|yaw| > 30`): `-1000`

Metric weights used in progress scoring:

- `x,y,z: 0.9`
- `roll,pitch,yaw: 1.0`
- `roll_rate,pitch_rate,yaw_rate: 1.1`
- `range: 1.2`
- `rate: 1.4`

### Reward Quick Reference

Definitions:

- $\mathrm{clip}(x,a,b)=\min(\max(x,a),b)$
- $\mathbb{I}(\cdot)$ is the indicator function, equal to $1$ when true and $0$ otherwise
- $A$ is the set of active control dimensions in the current step, and $|A|$ its size
- $\Delta_{d,m}$ is the improvement of metric $m$ on dimension $d$
- $w_m$ is the metric weight

Core Terms:

$$
R_{\text{step}} = R_{\text{action}} + R_{\text{progress}} + R_{\text{noop}} + R_{\text{safety}} + R_{\text{terminal}}
$$

$$
R_{\text{action}} = -0.03\,|A| - 0.12\,N_{\text{quick\_repeat}} - 0.16\,N_{\text{direction\_flip}}
$$

$$
R_{\text{progress}} = \sum_{d}\sum_{m\in\text{mapped}(d)} \mathbb{I}(\text{active}_d)\, s_d\,\mathrm{clip}(\Delta_{d,m}w_m,-0.8,0.8) + R_{\text{repeat}} + R_{\text{ineff}}
$$

Here, $s_d=0.35$ for translation dimensions and $s_d=1.0$ for rotation dimensions.
$R_{\text{repeat}}$ is the repeated-command penalty (translation: $-0.08$, rotation: $-0.12$),
and $R_{\text{ineff}}$ is the ineffective-rotation penalty ($-0.06$).

$$
R_{\text{noop}} = \sum_{d}\sum_{m\in\text{mapped}(d)} \mathbb{I}(\neg\text{active}_d)\left[
\mathrm{clip}(0.5\Delta_{d,m},0,0.25)
 + \mathbb{I}(\text{observe\_window}_d)\mathrm{clip}(0.8\Delta_{d,m},0,0.2)
 - \mathrm{clip}(0.35(\text{violation}_{d,m}-\Delta_{d,m}),0,0.25)
\right]
$$

$$
\begin{aligned}
R_{\text{safety}} =
&-10\,\mathbb{I}(\text{range}<5\land\text{rate}>0.2)
-((-\text{rate})\cdot 30)^2\,\mathbb{I}(\text{rate}<0) \\
&-(\,(0.02-\text{rate})\cdot 20\,)^2\,\mathbb{I}(0\le \text{rate}<0.02)
-(\,(\text{rate}-0.2)\cdot 30\,)^2\,\mathbb{I}(\text{rate}>0.2) \\
&-0.1\,\mathbb{I}(\text{range}>15\land 0\le\text{rate}<0.02)
 -\sum_{a\in\{roll,pitch,yaw\}} 0.8\,|\dot a - r_a^{*}| \\
&-\sum_{a\in\{roll,pitch,yaw\}} (\,(|\dot a|-0.25)\cdot 12\,)^2\,\mathbb{I}(|\dot a|>0.25)
\end{aligned}
$$

$$
\begin{aligned}
r^{*}_{roll} &= \mathrm{clip}(-0.02\\,roll,-0.25,0.25),\\
r^{*}_{pitch} &= \mathrm{clip}(0.02\\,pitch,-0.25,0.25),\\
r^{*}_{yaw} &= \mathrm{clip}(0.02\\,yaw,-0.25,0.25)
\end{aligned}
$$

$$
R_{\text{terminal}} = 1000\,\mathbb{I}(\text{success})
-300\,\mathbb{I}(\text{fuel\_empty})
-500\,\mathbb{I}(\text{rate}>0.8)
-1000\,\mathbb{I}(\text{range}>350\ \lor\ \exists a\in\{roll,pitch,yaw\}:|a|>30)
$$

### Algorithm

Proximal Policy Optimization (**PPO**) from Stable-Baselines3 with an MLP policy.
Training uses vectorized multi-environment sampling configurable via `--num-envs` (default `16`). 
State input is standardized dynamically using `VecNormalize` to guarantee numerical stability.

## Project Structure

```text
.
├── environments/
│   ├── __init__.py
│   ├── train/
│   │   ├── __init__.py
│   │   ├── environment.py    # TrainIssDockingEnv (reward shaping)
│   │   └── simulator.py      # Local dynamics engine
│   └── evaluate/
│       ├── __init__.py
│       ├── environment.py    # EvalIssDockingEnv (browser-backed, reward=0)
│       └── browser.py        # Playwright automation
├── train.py                  # Fast offline PPO training on TrainIssDockingEnv
├── evaluate.py               # Visual evaluation on the real browser via Playwright
├── tests/
│   └── test_simulator_matches_browser_trace.py
├── requirements.txt          # Python dependencies
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
playwright install chromium
```

## Usage

### Train (Fast offline Mode)

Thanks to the pure Python environment, training is now extremely fast and does not require a browser. 

```bash
# Default: Train with 16 parallel vector environments utilizing all CPU cores
python train.py

# Customizing training specs:
python train.py --num-envs 8 --timesteps 5000000 --checkpoint-freq 100000
```

When training, two things are saved periodically: the policy weights (`.zip`) and the environment normalization statistics (`_vec_normalize.pkl`). Both are required to evaluate or resume training.

Example – **resume a previous run**:

```bash
python train.py --resume --model-path models/ppo_docking --timesteps 1000000
```

### Stop training safely (without losing progress)

You can stop training any time with `Ctrl+C`.
The script automatically intercepts `KeyboardInterrupt`, saves the current model and statistics, and exits cleanly.

### Evaluate (Real Browser Mode)

Inference evaluation loads the model and connects to the **real SpaceX simulator in your browser**. It normalizes inputs perfectly using the training statistics (`_vec_normalize.pkl`).

```bash
# Launch Playwright browser locally and run 10 episodes visually
python evaluate.py --launch-browser --model models/ppo_docking --episodes 10
```

## Tests

The current parity test is browser-first and checks simulator correctness by
direct comparison to browser rollouts:

```bash
python -m unittest tests/test_simulator_matches_browser_trace.py -v
```

If browser and simulator diverge beyond tolerance, the test fails.

## License

See [LICENSE](LICENSE).
