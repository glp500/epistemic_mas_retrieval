"""List a block's episodes as a job file for a Slurm array, and print the sbatch line.

    slurm/py -m harvest.blocks A
    slurm/py -m harvest.blocks A --seeds 5

Writes <output_root>/jobs/<block>.txt (with -r-<arm> or -m-<model> when filtered) with one `harvest.play` argument line per
episode. Finished episodes are left out, so resubmitting reruns only what is missing.
"""

import argparse
import json
import os
import random
from pathlib import Path

import yaml

from harvest import ROOT
from harvest.protocol import block_jobs, episode_id, validate_v4


def play_arguments(job, protocol_path, root, horizon, stage="main"):
    """One job's `harvest.play` command-line arguments."""
    cell = job["cell"]
    identity = episode_id({"protocol": "live-v4"}, job["model"], job["language"], job["seed"], cell)
    output = Path(root) / f"block{job['block']}" / identity
    arguments = (
        f"--protocol {protocol_path} --model {job['model']} --language {job['language']} --seed {job['seed']}"
        f" --block {job['block']} --reasoning {cell['reasoning']} --knowledge {cell['knowledge']}"
        f" --certainty {cell['certainty']} --time {cell['time']} --false-rule {cell['false_rule']}"
        f" --wording {cell['wording']} --stage {stage} --steps {horizon} --output {output}"
    )
    return identity, output, arguments


def episode_state(folder, horizon):
    """"new", "complete" or "incomplete" (still running, or crashed). A complete
    episode of another length means two runs share one output root: stop."""
    manifest = folder / "manifest.json"
    if not folder.exists():
        return "new"
    if not manifest.exists():
        return "incomplete"
    info = json.loads(manifest.read_text())
    if info["status"] != "complete":
        return "incomplete"
    if info["horizon"] != horizon:
        raise SystemExit(f"{folder.name} has {info['horizon']} frames, not {horizon}. Use a separate HARVEST_OUTPUT.")
    return "complete"


def set_aside(folder):
    """Rename a dead episode, so its rerun starts clean. Nothing is deleted."""
    number = 1
    while folder.with_name(f"{folder.name}.failed-{number}").exists():
        number += 1
    folder.rename(folder.with_name(f"{folder.name}.failed-{number}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("block")
    parser.add_argument("--protocol", type=Path, default=Path("configs/config_v4.yaml"), help="Relative to the repo root: job lines are split on spaces")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per cell: 3 to describe, 5 to test (A1)")
    parser.add_argument("--steps", type=int, help="Frames per episode (default: the full episode)")
    parser.add_argument("--stage", choices=("smoke", "pilot", "main"), default="main", help="smoke and pilot use the pilot seed, and the baseline prompt if none is chosen yet")
    parser.add_argument("--reasoning", choices=("off", "on"), help="Only this reasoning arm")
    parser.add_argument("--model", help="Only this model")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Set incomplete episodes aside and rerun them. Only when none of this block's jobs are in squeue.",
    )
    args = parser.parse_args()
    raw = yaml.safe_load(args.protocol.read_text())
    validate_v4(raw)
    if args.stage != "main":
        raw = {**raw, "seeds": raw["pilot_seeds"], "language_seeds": raw["pilot_seeds"], "chosen_prompt": raw["chosen_prompt"] or {}}
    root = os.environ.get("HARVEST_OUTPUT", raw["storage"]["output_root"])
    horizon = args.steps or raw["episode_frames"]
    lines, done, skipped = [], 0, 0
    jobs = block_jobs(raw, args.block, args.seeds)
    if args.reasoning:
        jobs = [j for j in jobs if j["cell"]["reasoning"] == args.reasoning]
    if args.model:
        jobs = [j for j in jobs if j["model"] == args.model]
    for job in jobs:
        _, output, arguments = play_arguments(job, args.protocol, root, horizon, args.stage)
        state = episode_state(output, horizon)
        if state == "complete":
            done += 1
        elif state == "incomplete" and not args.retry_failed:
            skipped += 1  # It may still be running; never touch it by default.
        else:
            if state == "incomplete":
                set_aside(output)
            lines.append(arguments)
    random.Random(f"{args.block}-{args.stage}").shuffle(lines)  # Run order never lines up with a condition.
    block = raw["blocks"][args.block]
    name = "-".join([args.block] + [f"{flag}-{value}" for flag, value in (("r", args.reasoning), ("m", args.model)) if value])
    job_file = Path(root) / "jobs" / f"{name}.txt"  # One file per filter, so a second listing never rewrites a queued array's lines
    job_file.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)  # Slurm writes job logs here and fails silently without it
    job_file.write_text("\n".join(lines) + "\n")
    print(f"Block {args.block} ({block['doc']}): {len(lines)} to run, {done} complete, {skipped} incomplete -> {job_file}")
    if skipped:
        print("Incomplete episodes may still be running. If squeue shows none, rerun with --retry-failed.")
    if lines:
        print(
            f"sbatch --array=1-{len(lines)}%4 --partition={block['gpu']} --time={raw['wall_time'][block['gpu']]}"
            f" slurm/play.sbatch {job_file}"
        )


if __name__ == "__main__":
    main()
