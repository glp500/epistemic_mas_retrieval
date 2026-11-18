"""Prompt text: one renderer for every v3 and v4 cell, plus reply parsing.

The baseline cell (no reasoning, facts only, certain, no time, no false rule) is the
v3 prompt plus the certain arm's opening sentence (decision 6B). v3's own templates
have no such sentence, so they still render the v3 prompt byte for byte;
tests/fixtures/v3_prompts.json.gz pins both.
"""

BASELINE = {"reasoning": "off", "knowledge": "facts", "certainty": "certain", "time": "none", "false_rule": "none", "wording": 0}

def memory_text(t, memory, k):
    """v3 memory: one line per past frame, oldest first; t-1 is the previous frame."""
    if not memory:
        return t["memory_empty"]
    lines = [t["memory_header"].format(k=k)]
    for index, entry in enumerate(memory):
        events = "; ".join(
            t["events"][e["kind"]].format(color=t["colors"].get(e.get("color"), ""))
            for e in entry["events"]
        )
        lines.append(
            t["memory_line"].format(
                ago=len(memory) - index,
                action=t["actions"][entry["action"]],
                events=events or t["nothing"],
            )
        )
    return "\n".join(lines)


def worded(t, cell):
    """The templates with this cell's wording. 0 is the main text; 1 and 2 are paraphrases."""
    return {**t, **t["paraphrases"][cell["wording"] - 1]} if cell["wording"] else t


def rule_lines(t, cell):
    """The rules block. The hedge names the source, so every rule stays exactly true.
    The v3 templates have no `certain` sentence; every v4 language has one (tested)."""
    before = {"certain": ["certain"] if "certain" in t else [], "sham": ["sham_certainty"], "hedged": ["hedge"]}[cell["certainty"]]
    after = {
        "facts": [],
        "sham": ["sham_knowledge"],
        "eating": ["eating"],
        "ripening": ["ripening"],
        "mechanism": ["eating", "ripening"],
    }[cell["knowledge"]]
    lines = [t[key] for key in before] + [t["common"]]
    return lines + ([" ".join(t[key] for key in after)] if after else [])


def time_line(t, cell, frame, horizon):
    """Frame numbers are zero-based; `horizon` is the run's true length."""
    if cell["time"] == "sham":
        return [t["sham_time"]]
    if cell["time"] == "frame":
        return [t["time_frame"].format(frame=frame)]
    if cell["time"] == "horizon":
        return [t["time_horizon"].format(frame=frame, horizon=horizon, left=horizon - frame)]
    return []


def render(row, language, task, templates, memory, k, order=None, retry=False, cell=BASELINE, frame=None, horizon=None):
    """One player's prompt. `task` is "action_open" (names listed in `order`, no
    letter codes; `retry` swaps in the stricter instruction) or "note"."""
    t = worded(templates["languages"][language], cell)
    scene = row["local_scene"]
    false_rule = {"none": [], "sham": [t.get("sham_false_rule")]}.get(cell["false_rule"], [t.get("false_rule")])
    repeated = false_rule if cell["false_rule"] == "three" else []
    berries = "; ".join(
        f"({p['dx']},{p['dy']};{t['colors'][p['color']]},{t['ripeness'][int(p['ripe'])]})"
        for p in scene["berries"]
    )
    avatars = "; ".join(
        f"({p['dx']},{p['dy']};{t['colors'][p['public_color']]}"
        + ("," + t["self"] if p["is_self"] else "")
        + ")"
        for p in scene["avatars"]
    )
    lines = [
        *rule_lines(t, cell),
        *false_rule,
        t["taste"].format(taste=t["colors"][row["taste"]]),
        t["alive"].format(status=t["alive_words"][int(scene["alive"])]),
        *repeated,
        t["berries"].format(items=berries or t["none"]),
        t["avatars"].format(items=avatars or t["none"]),
        memory_text(t, memory, k),
        *time_line(t, cell, frame, horizon),
        *repeated,
    ]
    if task == "note":
        return "\n".join(lines + [t["note"]])
    items = ", ".join(t["actions"][i] for i in order)
    instruction = t["instruction_retry"] if retry else t["instruction_open"]
    return "\n".join(lines + [t["action"], t["actions_open"].format(items=items), instruction])


STRIP = " \t\n.。!！?？؟,،:：;；\"'`*()[]「」"


def parse_action(reply, names):
    """Index of the one name an open reply gives, or None. An exact match after
    trimming punctuation and case wins. Otherwise names are searched longest first,
    and each match is removed, so "unripe patches" never also counts as "ripe patches"."""
    text = reply.strip(STRIP).casefold()
    folded = [n.casefold() for n in names]
    if text in folded:
        return folded.index(text)
    found = []
    for i in sorted(range(len(folded)), key=lambda i: -len(folded[i])):
        if folded[i] in text:
            found.append(i)
            text = text.replace(folded[i], " ")
    return found[0] if len(found) == 1 else None


def split_thought(text, marks, prefilled):
    """(thought, answer, closed) from a reply decoded with special tokens kept.
    `marks` holds the model's thinking delimiters. `prefilled` means the prompt
    already opened the thought (Qwen). Without it, a reply that never opens a
    thought is all answer (Gemma may skip thinking)."""
    if marks["close"] in text:
        thought, answer = text.split(marks["close"], 1)
        return thought.split(marks["open"])[-1].strip(), answer.strip(), True
    if prefilled or marks["open"] in text:
        return text.split(marks["open"])[-1].strip(), "", False
    return "", text.strip(), True
