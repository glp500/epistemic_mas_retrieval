"""Offline probes on logged game states. Never touches the game code or the simulator.

    .venv/bin/python -m harvest.probe_v4 --input outputs/v4/blockA

For chosen states of each complete episode, in a fresh conversation with thinking off:
rate -> explain -> rate -> questions with confidence -> certainty -> time (fixed order, M4).
In the reasoning-off arm it also scores all 11 action names for the entropy over actions.
Writes probes.jsonl into each episode folder. Mixed games and untranslated languages are
skipped. These states depend on how a cell played; harvest.screen compares cells fairly.
"""

import argparse
import gc
import gzip
import json
import math
import re
import zlib
from pathlib import Path

import yaml

from harvest import ROOT
from harvest.data_io import append, iter_rows
from harvest.prompts import parse_action


def pick_states(actions, scenario, count):
    """(frame, player) of `count` matching decisions, evenly spaced over the episode."""
    matches = [
        (r["frame"], r["player"])
        for r in actions
        if r["alive"] and all(r["scenario"][k] == v for k, v in scenario.items())
    ]
    if len(matches) <= count:
        return matches
    return [matches[i * len(matches) // count] for i in range(count)]


def logged_prompts(folder, wanted):
    """The first action prompt logged for each wanted (frame, player)."""
    wanted, found = set(wanted), {}
    with gzip.open(folder / "prompts.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["frame"], row["player"])
            if key in wanted:
                found[key] = row["prompts"][0]
    return found


def rating(reply):
    """The first digit from 1 to 7 in a reply, or None."""
    match = re.search(r"[1-7]", reply)
    return int(match.group()) if match else None


def frames_left(reply):
    """The first number in a reply, or None if it gives no number."""
    match = re.search(r"\d+", reply)
    return int(match.group()) if match else None


def entropy(log_scores):
    """Entropy in bits of the softmax over action names; log2(11) = 3.46 is uniform."""
    top = max(log_scores)
    weights = [math.exp(s - top) for s in log_scores]
    total = sum(weights)
    return -sum(w / total * math.log2(w / total) for w in weights if w > 0)


def user(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def assistant(text):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def conversation(model, context, probe, limit, rotation, left):
    """One fixed-order probe with thinking off: rate, explain, rate, then each question
    with its own confidence, then the certainty and time checks. `rotation` turns the
    option lists, and `left` is the true number of frames left. Answers are written
    out, without letter codes. Everything in the context can inform the answers, so
    these measure what a prompt conveys, not what a player learned by playing."""
    messages = []

    def turn(question, tokens=limit):
        messages.append(user(question))
        reply = model.generate([messages], tokens)[0]["answer"]
        messages.append(assistant(reply))
        return reply

    before = turn(context + "\n\n" + probe["rating"], 8)
    explanation = turn(probe["explain"])
    after = turn(probe["rating"], 8)
    diagnostic = []
    for question in probe["diagnostic"]:
        shift = rotation % len(question["options"])
        options = question["options"][shift:] + question["options"][:shift]
        items = ", ".join(options)
        reply = turn(question["question"] + " " + probe["diagnostic_instruction"].format(items=items), 12)
        chosen = parse_action(reply, options)
        answer = None if chosen is None else options[chosen]
        false_rule = question.get("false_rule")
        diagnostic.append(
            {
                "question": question["question"],
                "role": question["role"],
                "options": options,
                "reply": reply,
                "correct": answer == question["options"][question["correct"]],
                "believes_false_rule": None if false_rule is None else answer == question["options"][false_rule],
                "confidence": rating(turn(probe["confidence"], 8)),
            }
        )
    time_reply = turn(probe["time_check"], 12)
    return {
        "rating_before": rating(before),
        "rating_after": rating(after),
        "explanation": explanation,
        "diagnostic": diagnostic,
        "rule_certainty": rating(turn(probe["certainty_check"], 8)),
        "frames_left_reply": time_reply,
        "frames_left_said": frames_left(time_reply),
        "frames_left_true": left,
    }


def probe_episode(folder, model, protocol, templates):
    info = json.loads((folder / "manifest.json").read_text())
    t = templates["languages"][info["language"]]
    if "probe" not in t:
        print("skipping", folder.name, "- probes are not translated into", info["language"], flush=True)
        return
    settings = protocol["probe"]
    states = pick_states(iter_rows(folder / "actions.jsonl"), settings["scenario"], settings["states_per_episode"])
    prompts = logged_prompts(folder, states)
    model.reseed(zlib.crc32(info["episode_id"].encode()))
    with (folder / "probes.jsonl").open("w") as log:
        for rotation, key in enumerate(sorted(prompts)):
            prompt = prompts[key]
            # The context is the game state only: the action instruction is left out.
            context = prompt[: prompt.rindex(t["action"])].rstrip()
            names = t["actions"]
            # The entropy describes the thinking-off policy, so it only fits the reasoning-off arm.
            scores = model.score_options(prompt, names) if info["cell"]["reasoning"] == "off" else None
            left = info["horizon"] - key[0]
            append(
                log,
                {
                    "episode_id": info["episode_id"],
                    "frame": key[0],
                    "player": key[1],
                    **conversation(model, context, t["probe"], settings["max_new_tokens"], rotation, left),
                    "action_scores": dict(zip(names, scores)) if scores else None,
                    "action_entropy": entropy(scores) if scores else None,
                },
            )
    print("probed", folder.name, len(prompts), flush=True)


def main():
    import torch

    from harvest.llm import load_model
    from harvest.protocol import load_protocol

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Folder of episode folders")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/config_v4.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    raw = yaml.safe_load(args.protocol.read_text())
    templates = json.loads((args.protocol.parent / raw["prompts"]).read_text())
    folders = {}
    for manifest in sorted(args.input.glob("*/manifest.json")):
        info = json.loads(manifest.read_text())
        if info["status"] == "complete" and "+" not in info["protocol"]["model_key"]:
            folders.setdefault(info["protocol"]["model_key"], []).append(manifest.parent)
    for key, group in folders.items():
        protocol = load_protocol(args.protocol.resolve(), [key])
        model = load_model(protocol, args.device)
        for folder in group:
            probe_episode(folder, model, protocol, templates)
        del model  # Free the GPU before the next model loads.
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
