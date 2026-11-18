"""Reconstruct map composition from existing v4 events; no model or simulator runs.

    python -m harvest.coordination --input /path/to/blockA --output outputs/coordination

Exploratory analysis added after the initial results. State t is the state after
t joint steps; action-log frame t observes that state before the next step.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {1: "red", 2: "green", 3: "blue"}


def reconstruct_counts(initial, events, horizon):
    """Include state 0 and the final state; events use resulting-step indices."""
    initial = np.asarray(initial, dtype=int)
    if initial.shape != (3,) or (initial < 0).any() or horizon < 0:
        raise ValueError("Invalid initial counts or horizon")
    deltas = np.zeros((horizon + 1, 3), dtype=int)
    for frame, source, target in events:
        if not 1 <= frame <= horizon or source not in COLORS or target not in COLORS:
            raise ValueError("Event outside the frame or colour range")
        deltas[frame, source - 1] -= 1
        deltas[frame, target - 1] += 1
    counts = initial + deltas.cumsum(axis=0)
    if (counts < 0).any() or not np.all(counts.sum(axis=1) == initial.sum()):
        raise ValueError("Reconstruction violates patch conservation")
    return counts


def initial_counts():
    """Read the installed variant's executable map, not its contradictory docstring."""
    spec = importlib.util.find_spec("meltingpot")
    source = Path(spec.origin).parent / "configs/substrates/allelopathic_harvest__open.py"
    tree = ast.parse(source.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OPEN_ASCII_MAP" for t in n.targets)
    )
    grid = ast.literal_eval(node.value)
    return [grid.count(str(c)) for c in COLORS], source


def run(block, output):
    initial, map_source = initial_counts()
    assert initial == [116, 116, 116]
    trajectories, summaries = [], []
    validated = 0
    for folder in sorted(block.iterdir()):
        if not (folder / "manifest.json").exists():
            continue
        info = json.loads((folder / "manifest.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        horizon = summary["frames"]
        key = {
            "episode_id": folder.name,
            "model": info["protocol"]["model_key"],
            "reasoning": info["cell"]["reasoning"],
            "knowledge": info["cell"]["knowledge"],
            "seed": info["seed"],
        }
        events, overwrite = [], Counter()
        for line in (folder / "events.jsonl").open():
            event = json.loads(line)
            if event["name"] != "replanting":
                continue
            f = event["fields"]
            source, target, player = (int(f[k]) for k in ("source_berry", "target_berry", "player_index"))
            events.append((int(event["frame"]), source, target))
            own = 1 if player <= 8 else 2
            overwrite["blue" if source == 3 else "own" if source == own else "other"] += 1
        counts = reconstruct_counts(initial, events, horizon)
        maxima = counts.max(axis=1)
        unique = (counts == maxima[:, None]).sum(axis=1) == 1
        majorities = [COLORS[int(row.argmax()) + 1] if ok else None for row, ok in zip(counts, unique)]
        observed = {}
        attempts = own_attempts = 0
        for line in (folder / "actions.jsonl").open():
            action = json.loads(line)
            frame = action["frame"]
            assert action["majority_colour"] == majorities[frame], (folder.name, frame)
            observed[frame] = action["majority_colour"]
            if action["action"] in (8, 9, 10):
                attempts += 1
                own_attempts += action["action"] == {"red": 8, "green": 9}[action["taste"]]
        assert set(observed) == set(range(horizon))
        validated += len(observed)
        shares = counts / sum(initial)
        concentration = shares.max(axis=1)
        for frame, (row, value) in enumerate(zip(counts, concentration)):
            trajectories.append(
                {**key, "frame": frame, "red": row[0], "green": row[1], "blue": row[2], "largest_colour_share": value}
            )
        summaries.append(
            {
                **key,
                "frames": horizon,
                "plant_attempts": attempts,
                "own_plant_share": own_attempts / attempts,
                "recolourings": len(events),
                "overwrite_other_share": overwrite["other"] / len(events),
                "peak_largest_colour_share": concentration.max(),
                "final_largest_colour_share": concentration[-1],
                "late_mean_largest_colour_share": concentration[-300:].mean(),
                "final_red": counts[-1, 0],
                "final_green": counts[-1, 1],
                "final_blue": counts[-1, 2],
            }
        )
    data, episodes = pd.DataFrame(trajectories), pd.DataFrame(summaries)
    assert len(episodes) == 48 and validated == 72000
    output.mkdir(parents=True, exist_ok=False)
    data.to_csv(output / "coordination_trajectories.csv", index=False)
    episodes.to_csv(output / "coordination_episodes.csv", index=False)
    cells = episodes.groupby(["model", "reasoning", "knowledge"]).mean(numeric_only=True)
    cells.to_csv(output / "coordination_cell_means.csv")
    record = {
        "analysis_status": "exploratory; added after observing initial results",
        "initial_counts": initial,
        "map_source": "meltingpot/configs/substrates/allelopathic_harvest__open.py",
        "map_source_sha256": hashlib.sha256(map_source.read_bytes()).hexdigest(),
        "episodes": len(episodes),
        "validated_pre_action_majority_snapshots": validated,
        "final_state_included": True,
        "convergence_threshold": None,
        "peak_share_range": [episodes.peak_largest_colour_share.min(), episodes.peak_largest_colour_share.max()],
        "final_share_range": [episodes.final_largest_colour_share.min(), episodes.final_largest_colour_share.max()],
    }
    (output / "coordination_verification.json").write_text(json.dumps(record, indent=2))
    plot(episodes, output)
    print(json.dumps(record, indent=2))
    print(cells[["peak_largest_colour_share", "final_largest_colour_share", "own_plant_share"]].round(4).to_string())


def plot(episodes, output):
    """Show every episode on a common share scale, with no convergence threshold."""
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True, layout="constrained")
    levels = [("sham", "Sham"), ("eating", "Eating"), ("ripening", "Ripening"), ("mechanism", "Both rules")]
    for ax, model, name in zip(axes, ["qwen-27b", "gemma-31b"], ["Qwen3.5-27B", "Gemma 4 31B"]):
        for i, (level, _) in enumerate(levels):
            for arm, offset, color in [("off", -0.17, "#526478"), ("on", 0.17, "#007F80")]:
                cell = episodes[
                    (episodes.model == model) & (episodes.reasoning == arm) & (episodes.knowledge == level)
                ].sort_values("seed")
                for j, row in enumerate(cell.itertuples()):
                    x = i + offset + (j - 1) * 0.055
                    ax.plot(
                        [x, x], [row.final_largest_colour_share, row.peak_largest_colour_share], color=color, alpha=0.55, lw=1
                    )
                    ax.scatter(x, row.peak_largest_colour_share, marker="^", facecolors="white", edgecolors=color, s=40, zorder=3)
                    ax.scatter(x, row.final_largest_colour_share, marker="o", color=color, s=30, zorder=3)
        ax.set_title(name, loc="left", weight="bold", color="#17283D")
        ax.set_xticks(range(4), [label for _, label in levels])
        ax.set(ylim=(0.3, 1), xlim=(-0.5, 3.5))
        ax.set_yticks([1 / 3, 0.5, 0.75, 1], ["33⅓%", "50%", "75%", "100%"])
        ax.axhline(1 / 3, color="#CED8DF", ls=":", lw=1)
        ax.grid(axis="y", color="#E4E9ED", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Share of all 348 patches in the largest colour")
    fig.suptitle("Map composition remains mixed in all 48 episodes", fontsize=16, weight="bold", color="#17283D")
    fig.supxlabel(
        "Each vertical segment is one episode: open triangle = peak; filled circle = final state.\nLeft/slate = direct answer; right/teal = capped thought + answer. Initial shares: one third each. Exploratory.",
        fontsize=10,
    )
    for ext in ("png", "svg"):
        fig.savefig(output / f"12-map-concentration.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw Block A episode directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)
