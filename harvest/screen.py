"""Fixed-state screen: every prompt cell on the same saved game states, with no simulator.

    slurm/py -m harvest.screen bank --input "$HARVEST_OUTPUT/smoke" --output configs/state_bank_v4.json
    slurm/py -m harvest.screen run --model qwen-27b --output "$HARVEST_OUTPUT/screen"
    .venv/bin/python -m harvest.screen report --input outputs/v4/screen --output outputs/v4/screen-report

Live games diverge, so cells never face the same situation. Here they do. A level moves
on to games only if it changes more decisions than its sham, in all three wordings. The
wording is the replicate for "does this sentence's meaning matter"; states are repeated
measures, so they are averaged within a wording, never counted as independent.
"""

import argparse
import gzip
import json
import random
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from harvest import ROOT, game
from harvest.data_io import append, iter_rows, save
from harvest.metrics import action_group, js_distance, markdown, scenario_key
from harvest.play import decide
from harvest.prompts import BASELINE, render
from harvest.protocol import check_cell, full_cell, load_protocol

FACTORS = ["knowledge", "certainty", "time", "false_rule"]


def screen_cells(conditions):
    """The baseline, plus each non-baseline level of each factor in each wording. One
    factor at a time: the screen asks which sentences matter, not how they interact."""
    cells = [dict(BASELINE)]
    for factor in FACTORS:
        for level in conditions[factor][1:]:
            cells += [full_cell(**{factor: level, "wording": w}) for w in conditions["wording"]]
    return cells


def pick_states(rows, count, seed=7305):
    """`count` decisions spread evenly over scenarios: one from each in turn, so that
    rare situations are represented. `rows` are logged decisions with scenarios."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[scenario_key(row["scenario"])].append(row)
    for group in groups.values():
        rng.shuffle(group)
    keys = sorted(groups)
    picked = []
    while len(picked) < count and any(groups.values()):
        for key in keys:
            if groups[key] and len(picked) < count:
                picked.append(groups[key].pop())
    return picked


def build_bank(root, count):
    """Saved states from every reasoning-off episode below `root`. Memory and scenes hold
    no language, so one bank serves all cells."""
    rows = []
    for path in sorted(root.rglob("prompts.jsonl.gz")):
        info = json.loads((path.parent / "manifest.json").read_text())
        if info["status"] != "complete" or info["cell"]["reasoning"] != "off":
            continue
        decisions = {(r["frame"], r["player"]): r for r in iter_rows(path.parent / "actions.jsonl")}
        with gzip.open(path, "rt") as stream:
            for line in stream:
                logged = json.loads(line)
                scenario = decisions[(logged["frame"], logged["player"])]["scenario"]
                rows.append({**logged, "scenario": scenario, "source": info["episode_id"]})
    picked = pick_states(rows, count)
    return [
        {"state": i, "frame": r["frame"], "scene": r["scene"], "memory": r["memory"], "scenario": r["scenario"], "source": r["source"]}
        for i, r in enumerate(picked)
    ]


def list_order(state, order):
    """The same shuffled action list for one state and order, in every cell."""
    return np.random.default_rng([state, order]).permutation(len(game.ACTION_NAMES)).tolist()


def run_cell(model, cell, reasoning, bank, protocol, templates, orders, language="en"):
    """All states and orders for one cell. Returns one record per decision."""
    k, horizon = protocol["memory"]["frames"], protocol["episode_frames"]
    names = templates["languages"][language]["actions"]
    group = [
        {"state": s, "order": o, "list": list_order(s["state"], o), "prompts": [], "replies": []}
        for s in bank
        for o in range(orders)
    ]

    def prompt(p, retry):
        return render(
            p["state"]["scene"], language, "action_open", templates, p["state"]["memory"], k,
            order=p["list"], retry=retry, cell=cell, frame=p["state"]["frame"], horizon=horizon,
        )

    decide(model, group, names, reasoning, protocol, prompt)
    return [
        {
            "state": p["state"]["state"],
            "taste": p["state"]["scene"]["taste"],
            "order": p["order"],
            "action": 0 if p["action"] is None else p["action"],
            "unmatched": p["action"] is None,
            "retried": len(p["replies"]) > 1,
            "reply": p["replies"][-1]["answer"],
            "thought": p["replies"][-1].get("thought", ""),
            "thought_tokens": p["replies"][-1].get("thought_tokens", 0),
            "truncated": p["replies"][-1].get("truncated", False),
        }
        for p in group
    ]


def probe_cell(model, cell, bank, protocol, templates, language="en"):
    """The rebuilt probe on the first few states, with the same states for every cell."""
    from harvest.probe_v4 import conversation

    t = templates["languages"][language]
    k, horizon = protocol["memory"]["frames"], protocol["episode_frames"]
    rows = []
    for rotation, state in enumerate(bank[: protocol["screen"]["probe_states"]]):
        text = render(state["scene"], language, "action_open", templates, state["memory"], k,
                      order=list_order(state["state"], 0), cell=cell, frame=state["frame"], horizon=horizon)
        context = text[: text.rindex(t["action"])].rstrip()
        left = horizon - state["frame"]
        rows.append({"state": state["state"], **conversation(model, context, t["probe"], protocol["probe"]["max_new_tokens"], rotation, left)})
    return rows


def run(protocol, bank, output, device):
    """Every cell, both reasoning arms, then probes. A rerun skips finished cells, so a
    job that hits its time limit loses only the cell it was on."""
    from harvest.llm import load_model

    templates = json.loads(Path(protocol["prompts"]).read_text())
    conditions = yaml.safe_load(Path(protocol["config_path"]).read_text())["conditions"]
    cells = screen_cells(conditions)
    for cell in cells:
        check_cell(templates, "en", cell)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"screen_{protocol['model_key']}.jsonl"
    done = {(r["arm"], json.dumps(r["cell"], sort_keys=True), r["kind"]) for r in iter_rows(path)} if path.exists() else set()
    model = load_model(protocol, device)
    orders = protocol["screen"]["orders_per_state"]
    with path.open("a") as log:
        for arm in ("off", "on"):
            for cell in cells:
                if (arm, json.dumps(cell, sort_keys=True), "decision") in done:
                    continue
                model.reseed(zlib.crc32(f"{arm} {json.dumps(cell, sort_keys=True)}".encode()))
                for record in run_cell(model, cell, arm == "on", bank, protocol, templates, orders):
                    append(log, {"kind": "decision", "model": protocol["model_key"], "arm": arm, "cell": cell, **record})
                print("screened", arm, cell, flush=True)
        for cell in cells:
            if cell["wording"] == 0 and ("off", json.dumps(cell, sort_keys=True), "probe") not in done:
                model.reseed(zlib.crc32(f"probe {json.dumps(cell, sort_keys=True)}".encode()))
                for record in probe_cell(model, cell, bank, protocol, templates):
                    append(log, {"kind": "probe", "model": protocol["model_key"], "arm": "off", "cell": cell, **record})
    save(output / f"screen_{protocol['model_key']}.done.json", {"runtime": model.metadata, "peak_gpu_gb": model.peak_gb()})


def level_of(cell):
    """(factor, level) of a screen cell, or None for the baseline."""
    return next(((f, cell[f]) for f in FACTORS if cell[f] != BASELINE[f]), None)


def share_changed(decisions, reference):
    """Share of (state, order) pairs whose action differs from the reference."""
    keys = decisions.keys() & reference.keys()
    return sum(decisions[k] != reference[k] for k in keys) / len(keys) if keys else float("nan")


def mix(actions):
    counts = np.zeros(len(game.ACTION_NAMES))
    for action in actions:
        counts[action] += 1
    return counts


def report_tables(rows, min_beyond_sham):
    """One row per model, arm, level and wording, then one verdict per level. A level
    passes only if every wording changes more decisions than the same wording's sham, by
    `min_beyond_sham`; the baseline's own gap between list orders shows the noise floor."""
    decisions = defaultdict(dict)
    tastes = {}
    for r in rows:
        if r["kind"] == "decision":
            cell = r["cell"]
            decisions[(r["model"], r["arm"], level_of(cell), cell["wording"])][(r["state"], r["order"])] = r["action"]
            tastes[r["state"]] = r["taste"]
    table = []
    for (model, arm, level, wording), actions in sorted(decisions.items(), key=str):
        if level is None:
            continue
        base = decisions[(model, arm, None, 0)]
        chosen = list(actions.values())
        sham = decisions.get((model, arm, (level[0], "sham"), wording), {})
        groups = Counter(action_group(a, tastes[s]) for (s, _), a in actions.items())
        table.append(
            {"model": model, "arm": arm, "factor": level[0], "level": level[1], "wording": wording,
             "changed_vs_baseline": share_changed(actions, base), "changed_vs_sham": share_changed(actions, sham) if sham else float("nan"),
             "js_vs_baseline": js_distance(mix(chosen), mix(base.values())), "move_share": groups["move"] / len(chosen)}
        )
    table = pd.DataFrame(table)
    floor = []
    for (model, arm, level, wording), actions in decisions.items():
        if level is None:
            first = {s: a for (s, o), a in actions.items() if o == 0}
            second = {s: a for (s, o), a in actions.items() if o == 1}
            floor.append({"model": model, "arm": arm, "order_noise": share_changed(first, second)})
    verdict = []
    if len(table):
        for (model, arm, factor, level), group in table[table.level != "sham"].groupby(["model", "arm", "factor", "level"]):
            sham = table[(table.model == model) & (table.arm == arm) & (table.factor == factor) & (table.level == "sham")]
            joined = group.merge(sham[["wording", "changed_vs_baseline"]], on="wording", suffixes=("", "_sham"))
            margins = joined.changed_vs_baseline - joined.changed_vs_baseline_sham
            verdict.append({"model": model, "arm": arm, "factor": factor, "level": level, "wordings": len(joined),
                            "beyond_sham_min": margins.min(), "beyond_sham_mean": margins.mean(),
                            "passes": bool(len(joined) and margins.min() >= min_beyond_sham - 1e-9)})  # 0.30 - 0.25 is 0.0499... in floats
    return table, pd.DataFrame(floor), pd.DataFrame(verdict)


def report(root, output, min_beyond_sham):
    rows = [r for path in sorted(root.glob("screen_*.jsonl")) for r in iter_rows(path)]
    if not rows:
        raise SystemExit("No screen results found")
    output.mkdir(parents=True, exist_ok=False)
    table, floor, verdict = report_tables(rows, min_beyond_sham)
    table.to_csv(output / "screen_levels.csv", index=False)
    verdict.to_csv(output / "screen_verdict.csv", index=False)
    decisions = [r for r in rows if r["kind"] == "decision"]
    quality = pd.DataFrame(decisions).groupby(["model", "arm"])[["unmatched", "retried", "truncated", "thought_tokens"]].mean().reset_index()
    lines = ["# Fixed-state screen", "", "## Verdict: what moves on to games", "", markdown(verdict),
             "", "## Noise floor: decisions that change with the action-list order alone", "", markdown(floor),
             "", "## Reply quality", "", markdown(quality), "", "Per wording: `screen_levels.csv`."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print("wrote", output / "report.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("bank", "run", "report"))
    parser.add_argument("--input", type=Path, help="bank: smoke episodes; report: screen results")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=ROOT / "configs/state_bank_v4.json")
    parser.add_argument("--model")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/config_v4.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    raw = yaml.safe_load(args.protocol.read_text())
    if args.mode == "bank":
        bank = build_bank(args.input, raw["screen"]["states"])
        save(args.output, bank)
        print("saved", len(bank), "states:", dict(Counter(scenario_key(s["scenario"]) for s in bank)))
    elif args.mode == "run":
        protocol = {**load_protocol(args.protocol.resolve(), [args.model]), "config_path": str(args.protocol)}
        run(protocol, json.loads(args.bank.read_text()), args.output, args.device)
    else:
        report(args.input, args.output, raw["screen"]["min_beyond_sham"])


if __name__ == "__main__":
    main()
