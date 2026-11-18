"""Load and validate a protocol file: pinned models, cells, and the jobs in a block."""

import itertools
import re

import yaml

from harvest.prompts import BASELINE

NEEDED = {
    ("knowledge", "sham"): ["sham_knowledge"],
    ("knowledge", "eating"): ["eating"],
    ("knowledge", "ripening"): ["ripening"],
    ("knowledge", "mechanism"): ["eating", "ripening"],
    ("certainty", "sham"): ["sham_certainty"],
    ("certainty", "hedged"): ["hedge"],
    ("time", "sham"): ["sham_time"],
    ("time", "frame"): ["time_frame"],
    ("time", "horizon"): ["time_horizon"],
    ("false_rule", "sham"): ["sham_false_rule"],
    ("false_rule", "once"): ["false_rule"],
    ("false_rule", "three"): ["false_rule"],
}


def load_protocol(path, model_keys):
    """The protocol merged with its model. Several keys make a mixed game: they
    stay under "models", each merged with the shared settings when loaded."""
    protocol = yaml.safe_load(path.read_text())
    protocol["prompts"] = str((path.parent / protocol["prompts"]).resolve())
    unknown = set(model_keys) - set(protocol["models"])
    if unknown:
        raise ValueError(f"Unknown model {sorted(unknown)}; choose from {sorted(protocol['models'])}")
    shared = {k: v for k, v in protocol.items() if k != "models"}
    if len(model_keys) == 1:
        return {**shared, **protocol["models"][model_keys[0]], "model_key": model_keys[0]}
    return {**shared, "models": {k: protocol["models"][k] for k in model_keys}, "model_key": "+".join(model_keys)}


def model_configs(protocol):
    """{model key: settings} for one model or a mixed game."""
    if "models" in protocol:
        return {key: {**protocol, **entry} for key, entry in protocol["models"].items()}
    return {protocol["model_key"]: protocol}


def validate_v4(raw):
    """Stop early on a config that can't produce the planned blocks. `raw` is the parsed YAML."""
    for key, entry in raw["models"].items():
        for field in ("model_id", "revision", "dtype", "attention", "batch_size", "thinking"):
            if field not in entry:
                raise ValueError(f"Model {key} has no {field}")
    for factor, levels in raw["conditions"].items():
        if levels[0] != BASELINE[factor]:
            raise ValueError(f"The first {factor} level must be the v3 baseline")
    for name, block in raw["blocks"].items():
        for factor, levels in block["factors"].items():
            if not set(levels) <= set(raw["conditions"][factor]):
                raise ValueError(f"Block {name}: unknown {factor} level")
        if not set(block["models"]) <= set(raw["models"]):
            raise ValueError(f"Block {name}: unknown model")
        if block["gpu"] not in raw["wall_time"]:
            raise ValueError(f"Block {name}: no wall time for {block['gpu']}")


def full_cell(**levels):
    return {**BASELINE, **levels}


def check_cell(templates, language, cell):
    """A cell can only run in a language whose new sentences have been translated."""
    t = templates["languages"][language]
    needed = [key for (factor, level), keys in NEEDED.items() if cell[factor] == level for key in keys]
    missing = [key for key in needed + (["paraphrases"] if cell["wording"] else []) if key not in t]
    if missing:
        raise ValueError(f"{language} has no translation yet for {missing}")


def episode_id(protocol, model_key, language, seed, cell):
    """v3 ids are unchanged. v4 ids name the full cell."""
    if protocol["protocol"] == "live-v3":
        return f"v3_{model_key}_{language}_{seed}"
    parts = [f"{k[0]}-{cell[k]}" for k in ("reasoning", "knowledge", "certainty", "time", "false_rule", "wording")]
    return "_".join(["v4", model_key, language, str(seed), *parts])


def parse_players(specs, players=16):
    """Mixed games: ["qwen-2b:1-4,9-12", "qwen-27b:5-8,13-16"] -> {1: "qwen-2b", ...}.
    Each model must get 4 red (1-8) and 4 green (9-16) players, so that model and
    taste are not confounded."""
    assigned = {}
    for spec in specs:
        key, ranges = spec.split(":")
        for low, high in re.findall(r"(\d+)-(\d+)", ranges):
            for player in range(int(low), int(high) + 1):
                if player in assigned:
                    raise ValueError(f"Player {player} is assigned twice")
                assigned[player] = key
    if sorted(assigned) != list(range(1, players + 1)):
        raise ValueError("Every player must be assigned to one model")
    for key in set(assigned.values()):
        numbers = [p for p, k in assigned.items() if k == key]
        if sum(p <= 8 for p in numbers) != 4 or sum(p > 8 for p in numbers) != 4:
            raise ValueError(f"{key} needs 4 red and 4 green players")
    return assigned


def job_key(job):
    return job["model"], job["language"], job["seed"], *job["cell"].values()


def block_jobs(raw, name, seed_count=3):
    """Every new episode of a block, as dicts of `harvest.play` settings. `shared_with`
    drops episodes another block already runs, so a shared control is paid for once."""
    block = raw["blocks"][name]
    base = block.get("base", {})
    if block.get("use_chosen_prompt"):
        if raw["chosen_prompt"] is None:
            raise ValueError(f"Block {name} needs chosen_prompt in the config: set it after block A (D5)")
        base = raw["chosen_prompt"]
    seeds = raw[block.get("seeds", "seeds")][: block.get("seed_count", seed_count)]
    factors = block["factors"]
    combos = [dict(zip(factors, levels)) for levels in itertools.product(*factors.values())]
    jobs = [
        {"block": name, "model": model, "language": language, "seed": seed, "cell": full_cell(**{**base, **combo})}
        for combo, model, language, seed in itertools.product(combos, block["models"], block.get("languages", ["en"]), seeds)
    ]
    if "shared_with" in block:
        shared = {job_key(j) for j in block_jobs(raw, block["shared_with"], seed_count)}
        jobs = [j for j in jobs if job_key(j) not in shared]
    return jobs
