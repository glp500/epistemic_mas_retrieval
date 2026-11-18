"""Episode outcomes and the fixed-state screen’s shared action metrics."""

from collections import Counter

import numpy as np
from scipy.spatial.distance import jensenshannon

from harvest.game import ACTION_NAMES, COLORS

PLANT = {8: "red", 9: "green", 10: "blue"}
ZAP = 7
GROUPS = ["plant own colour", "plant other colour", "move", "turn", "zap", "do nothing"]


def taste(player):
    """The colour a player (1-16) prefers: 1-8 red, 9-16 green."""
    return "red" if player <= 8 else "green"


def berry_points(fields):
    """(player, reward) of one eating event: +2 for the player's own colour, else +1."""
    player = int(fields["player_index"])
    return player, 2 if COLORS[int(fields["berry_color"])] == taste(player) else 1


def active(episode):
    """The episode's decisions by players not removed by sanctions."""
    return [r for r in episode["actions"] if r["alive"]]


def episode_labels(info):
    return {"model": info["protocol"]["model_key"], "language": info["language"], "seed": info["seed"]}


def gini(values):
    """Gini of non-negative values; NaN when the total is 0 (nobody ate)."""
    x = np.asarray(values, dtype=float)
    if x.sum() == 0:
        return float("nan")
    return float(np.abs(x[:, None] - x[None, :]).sum() / (2 * len(x) ** 2 * x.mean()))


def js_distance(p, q):
    """Jensen-Shannon distance (base 2): 0 = same distribution, 1 = disjoint."""
    return float(jensenshannon(p, q, base=2))


def cooperation(action, majority):
    """Zap = defect; planting the map's majority colour = cooperate; planting
    another colour = other_plant; anything else, or no majority, = None."""
    if action == ZAP:
        return "defect"
    if action in PLANT and majority is not None:
        return "cooperate" if PLANT[action] == majority else "other_plant"
    return None


def action_group(action, taste):
    """Action group of one decision. `action` is the native index (game.ACTION_NAMES:
    0 noop, 1-4 move, 5-6 turn, 7 zap, 8-10 plant red/green/blue); `taste` is
    the player's preferred colour ("red" or "green"). Returns one of GROUPS.
    Own colour = the player's preference (not the map's majority: that is the
    separate cooperation measure). Blue is nobody's preference."""
    if action in PLANT:
        return "plant own colour" if PLANT[action] == taste else "plant other colour"
    if action == ZAP:
        return "zap"
    # Moving and turning are separate: only moving reaches berries (eating = stepping onto one).
    return "do nothing" if action == 0 else "move" if action <= 4 else "turn"


def scenario_key(s):
    return "|".join(s[k] for k in ("ripe_in_view", "neighbours_visible", "zap_seen", "phase"))


def outcomes(episode):
    info, actions, events = episode["info"], episode["actions"], episode["events"]
    alive = active(episode)
    returns, berry, eaten, replants, removals = Counter(), Counter(), Counter(), Counter(), Counter()
    for row in actions:
        returns[row["player"]] += row["reward"]
    for e in events:
        f = e["fields"]
        if e["name"] == "eating":
            player, points = berry_points(f)
            eaten[player] += 1
            berry[player] += points
        elif e["name"] == "replanting":
            replants[int(f["player_index"])] += 1
        elif e["name"] == "removal_due_to_sanctioning":
            removals[int(f["target"])] += 1
    labels = Counter(cooperation(r["action"], r["majority_colour"]) for r in alive)
    plants = labels["cooperate"] + labels["other_plant"]
    decisions = len(alive)
    players = range(1, 17)
    return {
        "episode_id": info["episode_id"],
        **episode_labels(info),
        "stage": info["stage"],
        "frames": max(r["frame"] for r in actions) + 1,
        "welfare": sum(returns.values()),
        "gini_berry": gini([berry[p] for p in players]),
        "nobody_ate": sum(eaten.values()) == 0,
        "berries_eaten": sum(eaten.values()),
        "cooperate_share_of_plants": labels["cooperate"] / plants if plants else float("nan"),
        "zap_rate": labels["defect"] / decisions if decisions else float("nan"),
        "eat_without_planting": sum(eaten[p] > 0 and replants[p] == 0 for p in players),
        "free_rider_share": sum(eaten[p] > 0 and replants[p] == 0 for p in players) / len(players),
        "defector_share": len({r["player"] for r in alive if r["action"] == ZAP}
                              | {p for p in players if eaten[p] > 0 and replants[p] == 0}) / len(players),  # v4: zapped once, or free-rode
        "retry_rate": sum(r.get("retried", False) for r in alive) / decisions if decisions else float("nan"),
        "unmatched_rate": sum(r.get("unmatched", False) for r in alive) / decisions if decisions else float("nan"),
        "zaps": sum(e["name"] == "zap" for e in events),
        "removals": sum(removals.values()),
        "reward_residual": sum(returns[p] - (berry[p] - 10 * removals[p]) for p in players),
        **{f"n_{name}": sum(r["action"] == i for r in alive) for i, name in enumerate(ACTION_NAMES)},
    }


def markdown(frame):
    if not len(frame):
        return "_none_"
    cell = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)
