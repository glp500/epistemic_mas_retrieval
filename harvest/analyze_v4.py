"""v4 analysis: one row per episode with every factor as a column, plus the named tests.

    .venv/bin/python -m harvest.analyze_v4 --input outputs/v4/main \
        --output outputs/v4/analysis

Reads only saved files; never runs a model. The episode stays the unit. Tests are the
comparisons named in the protocol (A1), run per model, because effects can have opposite
signs across models. Berries eaten is the primary outcome; only it is adjusted by Holm.
"""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from harvest import ROOT
from harvest.data_io import iter_rows
from harvest.metrics import GROUPS, action_group, active, markdown, outcomes
from harvest.prompts import BASELINE

FACTORS = list(BASELINE)
SCRIPTS = {"zh": (0x4E00, 0x9FFF), "ar": (0x0600, 0x06FF)}


def script_ok(text, language):
    """Most letters are in the language's script (CJK, Arabic, or Latin)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    if language in SCRIPTS:
        low, high = SCRIPTS[language]
        return sum(low <= ord(c) <= high for c in letters) / len(letters) > 0.5
    return sum(c.isascii() or "À" <= c <= "ſ" for c in letters) / len(letters) > 0.5


def load_episodes(root):
    """Every complete v4 episode below `root`, at any depth."""
    episodes = []
    for manifest in sorted(root.rglob("manifest.json")):
        info = json.loads(manifest.read_text())
        if info["status"] != "complete":
            print("skipping incomplete", manifest.parent.name)
            continue
        folder = manifest.parent
        episodes.append(
            {
                "info": info,
                "folder": folder,
                "actions": list(iter_rows(folder / "actions.jsonl")),
                "events": list(iter_rows(folder / "events.jsonl")),
            }
        )
    return episodes


def end_game_shares(rows, frames, share):
    """Action-group shares among active decisions in the last `share` of frames."""
    start = frames * (1 - share)
    late = [r for r in rows if r["frame"] >= start]
    counts = Counter(action_group(r["action"], r["taste"]) for r in late)
    return {f"end_{g.replace(' ', '_')}": counts[g] / len(late) if late else float("nan") for g in GROUPS}


def trace_action(thought, names):
    """M2: the action a thought ends up intending, as an index into `names`, or None."""
    text = thought.casefold()
    # TODO(human): decide how a thought's stated intention is read. The default takes the last name mentioned.
    positions = {i: text.rfind(n.casefold()) for i, n in enumerate(names)}
    best = max(positions, key=positions.get)
    return best if positions[best] >= 0 else None


def mentions(text, keywords):
    text = text.casefold()
    return any(word in text for word in keywords)


def trace_measures(folder, rows, language, names, keywords, script_check=script_ok):
    """Per-episode measures from the logged thoughts. Empty for reasoning-off episodes.
    Agreement uses finished thoughts only: a cut-off thought often stops mid-list, so the
    last action it names says little about what the player intended."""
    path = folder / "prompts.jsonl.gz"
    if not path.exists():
        return {}
    decisions = {(r["frame"], r["player"]): r for r in rows}
    traces, eating, growth, in_language, intended, agreed = 0, 0, 0, 0, 0, 0
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            thought = row["thoughts"][0]
            if not thought:
                continue
            traces += 1
            eating += mentions(thought, keywords.get("eating", []))
            growth += mentions(thought, keywords.get("growth", []))
            in_language += script_check(thought, language)
            decision = decisions[(row["frame"], row["player"])]
            guess = None if decision["truncated"] else trace_action(thought, names)
            if guess is not None:
                intended += 1
                agreed += guess == decision["action"]
    if not traces:
        return {}
    usable = bool(keywords)
    return {
        "traces": traces,
        "trace_mentions_eating": eating / traces if usable else float("nan"),
        "trace_mentions_growth": growth / traces if usable else float("nan"),
        "trace_in_prompt_script": in_language / traces,
        "trace_action_agreement": agreed / intended if intended else float("nan"),
        "traces_agreement_n": intended,
    }


def mean_of(values):
    kept = [v for v in values if v is not None]
    return float(np.mean(kept)) if kept else float("nan")


def explanation_score(text, keywords):
    """Share of the rubric an explanation covers: eating, ripening and own-colour reward."""
    items = [keywords.get(k, []) for k in ("eating", "growth", "preference")]
    return sum(mentions(text, words) for words in items) / len(items)


def question_measures(rows):
    """Calibration from each question's own confidence, on manipulation questions only:
    `gap` is mean confidence minus mean correctness; `brier` is the squared distance
    between confidence and the outcome. Comprehension questions are reported apart."""
    asked = [q for r in rows for q in r["diagnostic"]]
    checks = [q for q in asked if q["role"] == "manipulation"]
    rated = [q for q in checks if q["confidence"] is not None]
    sure = [(q["confidence"] - 1) / 6 for q in rated]
    right = [float(q["correct"]) for q in rated]
    return {
        "probe_accuracy": mean_of(q["correct"] for q in checks),
        "probe_comprehension": mean_of(q["correct"] for q in asked if q["role"] == "comprehension"),
        "probe_believes_false_rule": mean_of(q["believes_false_rule"] for q in asked),
        "probe_confidence": mean_of(sure),
        "probe_calibration_gap": mean_of(sure) - mean_of(right) if rated else float("nan"),
        "probe_brier": mean_of((a - b) ** 2 for a, b in zip(sure, right)),
    }


def probe_measures(folder, keywords):
    path = folder / "probes.jsonl"
    if not path.exists():
        return {}
    rows = list(iter_rows(path))
    if not rows:
        return {}
    changes = [r["rating_after"] - r["rating_before"] for r in rows if None not in (r["rating_after"], r["rating_before"])]
    return {
        "probe_states": len(rows),
        **question_measures(rows),
        "probe_rating_change": mean_of(changes),
        "probe_explanation_score": mean_of(explanation_score(r["explanation"], keywords) for r in rows) if keywords else float("nan"),
        "probe_rule_certainty": mean_of(r["rule_certainty"] for r in rows),
        "probe_time_exact": mean_of(r["frames_left_said"] == r["frames_left_true"] for r in rows),
        "probe_action_entropy": mean_of(r["action_entropy"] for r in rows),
    }


def episode_row(episode, protocol, templates):
    info, rows = episode["info"], episode["actions"]
    alive = active(episode)
    language = info["language"]
    row = {"block": info.get("block"), **outcomes(episode), **{f: info["cell"][f] for f in FACTORS}}
    decisions = len(alive) or 1
    row.update(
        truncated_rate=sum(r["truncated"] for r in alive) / decisions,
        thought_tokens=sum(r["thought_tokens"][0] for r in alive) / decisions,
        prompt_tokens=sum(r["prompt_tokens"] for r in alive) / decisions,
        seconds_per_frame=json.loads((episode["folder"] / "summary.json").read_text())["seconds_per_frame"],
        gpu=_gpu(info),
        **end_game_shares(alive, row["frames"], protocol["analysis"]["end_game_share"]),
    )
    returns = Counter()
    for r in rows:
        returns[r["model"]] += r["reward"]
    if len(returns) > 1:  # mixed game: welfare per model as well as the total (D1)
        row["welfare_by_model"] = json.dumps(dict(returns))
    keywords = protocol["analysis"]["mechanism_keywords"].get(language, {})
    names = templates["languages"][language]["actions"]
    row.update(trace_measures(episode["folder"], rows, language, names, keywords))
    row.update(probe_measures(episode["folder"], keywords))
    return row


def _gpu(info):
    runtime = info["model_runtime"]
    return runtime.get("gpu") or "; ".join(str(m.get("gpu")) for m in runtime.values() if isinstance(m, dict))


def stratified_p(table, factor, a, b, outcome, draws, seed=7304):
    """Two-sided permutation p for level `b` minus level `a`: labels are shuffled
    only among episodes that share every other factor and the model, so the
    comparison never mixes models or other conditions. Returns (difference, p)."""
    rng = np.random.default_rng(seed)
    data = table[table[factor].isin([a, b])].dropna(subset=[outcome])
    strata = [c for c in ["model", "language", *FACTORS] if c != factor and data[c].nunique() > 1]
    groups = [g for _, g in data.groupby(strata)] if strata else [data]
    groups = [(g[outcome].to_numpy(float), (g[factor] == b).to_numpy()) for g in groups if g[factor].nunique() == 2]
    if not groups:
        return float("nan"), float("nan")

    def effect(labels):
        return np.mean([v[m].mean() - v[~m].mean() for (v, _), m in zip(groups, labels)])

    observed = effect([m for _, m in groups])
    hits = sum(abs(effect([rng.permutation(m) for _, m in groups])) >= abs(observed) - 1e-12 for _ in range(draws))
    return float(observed), (hits + 1) / (draws + 1)


def holm(p_values):
    """Holm-adjusted p-values, in the input order. NaN stays NaN."""
    p = np.asarray(p_values, float)
    order = [i for i in np.argsort(p) if not np.isnan(p[i])]
    adjusted, running = np.full(len(p), np.nan), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(order) - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted.tolist()


def comparisons(table, protocol):
    """The named comparisons, per model. Holm covers the primary outcome of the confirmatory
    comparisons only; every other p is unadjusted and descriptive."""
    settings = protocol["analysis"]
    rows = []
    for c in settings["comparisons"]:
        chosen = table[table.block.isin(c["blocks"])]
        for model, group in chosen.groupby("model"):
            for outcome in [settings["primary"], *settings["secondary"]]:
                difference, p = stratified_p(group, c["factor"], c["a"], c["b"], outcome, settings["resamples"])
                rows.append({"blocks": "+".join(c["blocks"]), "model": model, "factor": c["factor"], "a": c["a"], "b": c["b"],
                             "outcome": outcome, "role": "primary" if outcome == settings["primary"] else "secondary",
                             "confirmatory": bool(c.get("confirmatory")),
                             "difference_b_minus_a": difference, "p": p})
    frame = pd.DataFrame(rows)
    if len(frame):
        primary = (frame.role == "primary") & frame.confirmatory
        frame["p_holm"] = float("nan")
        frame.loc[primary, "p_holm"] = holm(frame[primary].p)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="Folder holding raw episode folders")
    inputs.add_argument("--outcomes", type=Path, help="Saved per-episode outcomes CSV")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/config_v4.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    if args.outcomes:
        # Preserve the literal "none" treatment labels and the serialized float values.
        table = pd.read_csv(args.outcomes, keep_default_na=False, na_values=[""], float_precision="round_trip")
        if table.empty or table.episode_id.duplicated().any():
            raise SystemExit("Expected a non-empty outcomes table with one row per episode")
    else:
        templates = json.loads((args.protocol.parent / protocol["prompts"]).read_text())
        episodes = load_episodes(args.input)
        if not episodes:
            raise SystemExit("No complete episodes found")
        table = pd.DataFrame([episode_row(ep, protocol, templates) for ep in episodes])
    args.output.mkdir(parents=True, exist_ok=False)
    table.to_csv(args.output / "outcomes.csv", index=False)
    tests = comparisons(table, protocol)
    tests.to_csv(args.output / "comparisons.csv", index=False)
    columns = ["block", "model", *FACTORS]
    means = table.groupby(columns, dropna=False)[
        [c for c in ["welfare", "berries_eaten", "gini_berry", "zap_rate", "retry_rate", "unmatched_rate", "truncated_rate",
                     "thought_tokens", "prompt_tokens", "seconds_per_frame"] if c in table]
    ].mean().reset_index()
    means.insert(len(columns), "episodes", table.groupby(columns, dropna=False).size().to_numpy())
    means.to_csv(args.output / "cell_means.csv", index=False)
    report = [
        "# v4 report",
        "",
        f"{len(table)} complete episodes; stages: {dict(Counter(table.stage))}. Unit: episode. Per-episode columns: `outcomes.csv`.",
        "",
        "## Named comparisons, per model (permutation within strata; Holm on the primary outcome of confirmatory comparisons only)",
        "",
        markdown(tests),
        "",
        "## Cell means (descriptive)",
        "",
        markdown(means),
    ]
    (args.output / "report.md").write_text("\n".join(report) + "\n")
    print("wrote", args.output / "report.md")


if __name__ == "__main__":
    main()
