"""Play one episode: 16 players, one language, 10-frame memory, open-answer actions.

    .venv/bin/python -m harvest.play --model qwen-27b --language en --seed 500 --steps 50 \
        --reasoning on --knowledge mechanism --time horizon --output outputs/v4/smoke/example

    Mixed game (needs both models in memory): --players model-a:1-4,9-12 model-b:5-8,13-16
    v3 episode: --protocol configs/config_v3.yaml --model qwen

Each frame, the active players of each model form one batch. An unmatched reply gets
one stricter retry in a second batch, then the player does nothing. Thoughts and notes
never enter memory. An episode cannot be resumed: rerun it if it is interrupted.
"""

import argparse
import gzip
import hashlib
import json
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np

from harvest import ROOT, game
from harvest.data_io import append, digest, plain, save
from harvest.prompts import BASELINE, parse_action, render
from harvest.protocol import check_cell, episode_id, full_cell, load_protocol, model_configs, parse_players

TASTES = ["red"] * 8 + ["green"] * 8  # Melting Pot default roles, players 1-16


def event_record(event, seed, frame):
    name, values = event
    fields = {}
    if values and values[0] == b"dict":
        fields = {values[i].decode(): plain(values[i + 1]) for i in range(1, len(values), 2)}
    return {"seed": seed, "frame": frame, "name": name, "fields": fields, "raw": plain(event)}


def ask(model, prompts, reasoning, protocol):
    """One batch of decisions, with or without a thought before the answer."""
    if reasoning:
        return model.think(prompts, protocol["reasoning"]["budget_tokens"], protocol["reasoning"]["answer_tokens"])
    return model.generate(prompts, protocol["actions"]["max_new_tokens"])


def decide(model, group, names, reasoning, protocol, render_prompt):
    """Fill each player in `group` with its replies and action: one batch, then
    one stricter retry batch for players whose reply matched no action."""
    waiting = group
    for retry in (False, True):
        prompts = [render_prompt(p, retry) for p in waiting]
        for player, prompt, reply in zip(waiting, prompts, ask(model, prompts, reasoning, protocol)):
            player["prompts"].append(prompt)
            player["replies"].append(reply)
            player["action"] = parse_action(reply["answer"], names)
        waiting = [p for p in waiting if p["action"] is None]
        if not waiting:
            break


def play(protocol, output, seed, language, steps, device, cell=None, players=None, block=None, stage=None):
    """`players` maps 1-based player numbers to model keys (mixed games only)."""
    from harvest.llm import load_model

    cell = cell or BASELINE
    if language not in protocol["languages"]:
        raise ValueError(f"Unknown --language {language}")
    if seed not in protocol["pilot_seeds"] + protocol["seeds"] + protocol.get("language_seeds", []):
        raise ValueError("--seed must be one of the protocol's seeds")
    horizon = steps or protocol["episode_frames"]
    if not 1 <= horizon <= protocol["episode_frames"]:
        raise ValueError("--steps must be within the episode horizon")
    templates = json.loads(Path(protocol["prompts"]).read_text())
    check_cell(templates, language, cell)
    configs = model_configs(protocol)
    models = {key: load_model(config, device) for key, config in configs.items()}  # Torch initializes before simulator/JAX.
    next(iter(models.values())).reseed(seed)  # one global generator, shared by every model in the game
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")  # Simulation stays on CPU.
    owner = players or {i + 1: protocol["model_key"] for i in range(len(TASTES))}
    berries, avatars = game.berry_templates(), game.avatar_templates()
    k, every = protocol["memory"]["frames"], protocol["notes"]["every_frames"]
    phases = protocol["scenario"]["phase"]
    names = templates["languages"][language]["actions"]
    reasoning = cell["reasoning"] == "on"
    full_prompts = protocol.get("logging", {}).get("full_prompts", False)
    v4 = protocol["protocol"] != "live-v3"
    memories = [deque(maxlen=k) for _ in TASTES]
    rng = np.random.default_rng(seed)  # action-list orders
    this_episode = episode_id(protocol, protocol["model_key"], language, seed, cell)
    output.mkdir(parents=True, exist_ok=False)
    info = {
        "episode_id": this_episode,
        "stage": stage or ("episode" if horizon == protocol["episode_frames"] else "smoke"),
        "protocol": protocol,
        "protocol_hash": digest(protocol),
        "template_hash": digest(templates),
        "translation_status": templates["review_status"][language],
        "model_runtime": next(iter(models.values())).metadata
        if len(models) == 1
        else {key: model.metadata for key, model in models.items()},
        "seed": seed,
        "language": language,
        "horizon": horizon,
        "native_actions": dict(enumerate(game.ACTION_NAMES)),
        "status": "running",
    }
    if v4:
        info.update(cell=cell, block=block, players={str(p): m for p, m in owner.items()})
    save(output / "manifest.json", info)
    clock, step_events = {"frame": 0}, []
    env = game.build_game(seed)
    subscription = env.observables().events.subscribe(
        lambda event: step_events.append(event_record(event, seed, clock["frame"]))
    )
    counts, returns, started = Counter(), [0.0] * len(TASTES), time.perf_counter()
    prompt_log = gzip.open(output / "prompts.jsonl.gz", "wt") if full_prompts else None  # noqa: SIM115
    try:
        state = env.reset()
        with (
            (output / "actions.jsonl").open("w") as actions_log,
            (output / "events.jsonl").open("w") as events_log,
            (output / "notes.jsonl").open("w") as notes_log,
        ):
            for frame in range(horizon):
                majority = game.majority_colour(state.observation[0])
                note_frame = (frame + 1) % every == 0
                actions, rows, visible, active = [0] * len(TASTES), [], [], []
                for i, taste in enumerate(TASTES):
                    _, local, _ = game.scene(state.observation[i], i, berries, avatars, ids=True)
                    memory = list(memories[i])
                    visible.append({a["player"] for a in local["avatars"] if not a["is_self"]})
                    row = {
                        "episode_id": this_episode,
                        "model": owner[i + 1],
                        "language": language,
                        "seed": seed,
                        "frame": frame,
                        "player": i + 1,
                        "taste": taste,
                        "alive": local["alive"],
                        "majority_colour": majority,
                    }
                    rows.append(row)
                    if not local["alive"]:
                        row.update(action=0, reason="inactive_forced_noop")
                        continue
                    # Open answer: names listed in a fresh random order, no letter codes.
                    active.append(
                        {
                            "index": i,
                            "row": row,
                            "memory": memory,
                            "scene": {"taste": taste, "local_scene": local, "hidden_control": False},
                            "order": rng.permutation(len(game.ACTION_NAMES)).tolist(),
                            "prompts": [],
                            "replies": [],
                        }
                    )

                def render_prompt(p, retry, frame=frame):
                    return render(
                        p["scene"], language, "action_open", templates, p["memory"], k,
                        order=p["order"], retry=retry, cell=cell, frame=frame, horizon=horizon,
                    )

                for key, model in models.items():
                    group = [p for p in active if p["row"]["model"] == key]
                    if not group:
                        continue
                    decided = time.perf_counter()
                    decide(model, group, names, reasoning, protocol, render_prompt)
                    seconds = (time.perf_counter() - decided) / len(group)
                    for p in group:
                        actions[p["index"]] = 0 if p["action"] is None else p["action"]
                        p["row"].update(
                            action=actions[p["index"]],
                            order=p["order"],
                            replies=[r["answer"] for r in p["replies"]],
                            retried=len(p["replies"]) > 1,
                            unmatched=p["action"] is None,
                            prompt_sha256=hashlib.sha256(p["prompts"][-1].encode()).hexdigest(),
                            decision_seconds=seconds,
                            scenario=game.scenario(p["scene"]["local_scene"], p["scene"]["taste"], p["memory"], frame, phases),
                        )
                        if v4:
                            p["row"].update(
                                prompt_tokens=p["replies"][-1]["prompt_tokens"],
                                thought_tokens=[r.get("thought_tokens", 0) for r in p["replies"]],
                                truncated=any(r.get("truncated", False) for r in p["replies"]),
                            )
                        if prompt_log:
                            append(
                                prompt_log,
                                {
                                    "frame": frame,
                                    "player": p["index"] + 1,
                                    "prompts": p["prompts"],
                                    "thoughts": [r.get("thought", "") for r in p["replies"]],
                                    "scene": p["scene"],
                                    "memory": p["memory"],
                                },
                            )
                    if note_frame:
                        note_prompts = [
                            render(p["scene"], language, "note", templates, p["memory"], k, cell=cell, frame=frame, horizon=horizon)
                            for p in group
                        ]
                        notes = model.generate(note_prompts, protocol["notes"]["max_new_tokens"])
                        for p, prompt, note in zip(group, note_prompts, notes):
                            append(
                                notes_log,
                                {**p["row"], "note": note["answer"], "note_prompt": prompt, "action_prompt": p["prompts"][-1]},
                            )
                clock["frame"] = frame + 1  # Events are indexed by the resulting step.
                state = env.step(actions)
                for i, record in enumerate(rows):
                    record["reward"] = float(state.reward[i])
                    returns[i] += record["reward"]
                    counts[game.ACTION_NAMES[record["action"]]] += 1
                    append(actions_log, record)
                    memories[i].append({"action": actions[i], "events": game.seen_events(step_events, i + 1, visible[i])})
                for event in step_events:
                    append(events_log, {"episode_id": this_episode, **event})
                step_events.clear()
                if frame % 50 == 0:
                    print(this_episode, frame, dict(counts), flush=True)
                if state.last():
                    break
    finally:
        if prompt_log:
            prompt_log.close()
        subscription.dispose()
        env.close()
    summary = {
        "episode_id": this_episode,
        "frames": frame + 1,
        "action_counts": dict(counts),
        "returns": returns,
        "welfare": sum(returns),
        "seconds_per_frame": (time.perf_counter() - started) / (frame + 1),
        "peak_gpu_gb": {key: model.peak_gb() for key, model in models.items()},
    }
    save(output / "summary.json", summary)
    info["status"] = "complete"
    save(output / "manifest.json", info)
    print(summary, flush=True)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/config_v4.yaml")
    parser.add_argument("--model", help="One model key from the protocol")
    parser.add_argument("--players", nargs="+", help="Mixed game: key:1-4,9-12 key:5-8,13-16")
    parser.add_argument("--language", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, help="Below the full horizon the episode is labelled smoke")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--block", help="Block label for the manifest")
    for factor, levels in (
        ("reasoning", ["off", "on"]),
        ("knowledge", ["facts", "sham", "eating", "ripening", "mechanism"]),
        ("certainty", ["certain", "sham", "hedged"]),
        ("time", ["none", "sham", "frame", "horizon"]),
        ("false-rule", ["none", "sham", "once", "three"]),
    ):
        parser.add_argument(f"--{factor}", choices=levels, default=levels[0])
    parser.add_argument("--wording", type=int, choices=(0, 1, 2), default=0, help="0 = main text, 1 and 2 = paraphrases")
    parser.add_argument("--stage", choices=("smoke", "pilot", "main"), help="Label in the manifest (v4)")
    args = parser.parse_args(argv)
    if bool(args.model) == bool(args.players):
        parser.error("Give --model or --players, not both")
    return args


def main():
    args = parse_arguments()
    assignment = parse_players(args.players) if args.players else None
    keys = [args.model] if args.model else list(dict.fromkeys(s.split(":")[0] for s in args.players))
    cell = full_cell(
        reasoning=args.reasoning, knowledge=args.knowledge, certainty=args.certainty, time=args.time,
        false_rule=args.false_rule, wording=args.wording,
    )
    protocol = load_protocol(args.protocol.resolve(), keys)
    if protocol["protocol"] == "live-v3" and cell != BASELINE:
        raise SystemExit("v3 has one cell; use the v4 protocol for reasoning and prompt conditions")
    play(protocol, args.output, args.seed, args.language, args.steps, args.device, cell, assignment, args.block, args.stage)


if __name__ == "__main__":
    main()
