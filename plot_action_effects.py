"""Visualize learned button→state effects as heatmaps.

Usage
-----
python plot_action_effects.py --input analysis/effects/action_effects_latest.json
python plot_action_effects.py --input analysis/effects/action_effects_latest.json --output analysis/effects/action_effects.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_effect_matrix(summary: dict) -> tuple[np.ndarray, list[str], list[str]]:
    obs_keys: list[str] = summary["obs_keys"]
    buttons: list[str] = summary["button_actions"]
    effects: dict[str, dict[str, float]] = summary["effects"]

    matrix = np.zeros((len(buttons), len(obs_keys)), dtype=np.float32)
    for row, button in enumerate(buttons):
        for col, key in enumerate(obs_keys):
            matrix[row, col] = float(effects[button][key])
    return matrix, buttons, obs_keys


def build_count_vector(summary: dict, buttons: list[str]) -> np.ndarray:
    counts: dict[str, int] = summary["counts"]
    return np.array([int(counts.get(button, 0)) for button in buttons], dtype=np.int32)


def plot_summary(summary: dict, output: Path) -> None:
    matrix, buttons, obs_keys = build_effect_matrix(summary)
    counts = build_count_vector(summary, buttons)
    timesteps = int(summary.get("timesteps", 0))

    max_abs = float(np.max(np.abs(matrix))) if matrix.size > 0 else 1.0
    vmax = max(max_abs, 1e-6)

    fig, (ax_heat, ax_bar) = plt.subplots(
        2,
        1,
        figsize=(16, 11),
        gridspec_kw={"height_ratios": [4, 1.4]},
        constrained_layout=True,
    )

    heat = ax_heat.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_heat.set_title(f"Learned Action Effects (timesteps={timesteps:,})")
    ax_heat.set_xlabel("State Keys")
    ax_heat.set_ylabel("Buttons")
    ax_heat.set_xticks(np.arange(len(obs_keys)))
    ax_heat.set_xticklabels(obs_keys, rotation=45, ha="right")
    ax_heat.set_yticks(np.arange(len(buttons)))
    ax_heat.set_yticklabels(buttons)
    cbar = fig.colorbar(heat, ax=ax_heat, shrink=0.9)
    cbar.set_label("Mean Δstate per button click")

    ax_bar.bar(np.arange(len(buttons)), counts, color="#4C78A8")
    ax_bar.set_title("Samples per Button")
    ax_bar.set_ylabel("Count")
    ax_bar.set_xticks(np.arange(len(buttons)))
    ax_bar.set_xticklabels(buttons, rotation=45, ha="right")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize learned button→state effects from exported JSON summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="analysis/effects/action_effects_latest.json",
        help="Path to action-effect summary JSON exported during training.",
    )
    parser.add_argument(
        "--output",
        default="analysis/effects/action_effects.png",
        help="Path to save the generated visualization image.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input summary not found: {input_path}")

    summary = load_summary(input_path)
    plot_summary(summary, output_path)
    print(f"Saved visualization: {output_path}")


if __name__ == "__main__":
    main()
