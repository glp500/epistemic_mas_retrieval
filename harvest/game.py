"""Simulator wrapper: scene extraction, seen events, majority colour and scenario labels.

Global snapshots are for logging and scenario labels; only local scenes reach prompts.
"""

import os
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np

COLORS = {1: "red", 2: "green", 3: "blue"}
ACTION_NAMES = [
    "noop",
    "forward",
    "backward",
    "step_left",
    "step_right",
    "turn_left",
    "turn_right",
    "zap",
    "plant_red",
    "plant_green",
    "plant_blue",
]
FORWARD = ((0, -1), (1, 0), (0, 1), (-1, 0))
RIGHT = ((1, 0), (0, 1), (-1, 0), (0, -1))


def build_game(seed, instrument=True):
    from meltingpot import substrate
    from meltingpot.utils.substrates import builder

    config = substrate.get_config("allelopathic_harvest__open").unlock()
    if instrument:
        original = config.lab2d_settings_builder

        def settings(roles, config):
            result = original(roles=roles, config=config)
            # Lab2D prepends its package root even to an absolute levelName.
            result["levelDirectory"] = os.path.relpath(
                Path(__file__).parent / "lua", builder._MELTINGPOT_ROOT
            )
            result["levelName"] = "harvest_snapshot"
            return result

        config.lab2d_settings_builder = settings
        config.global_observation_names = [
            "WORLD.RGB",
            "WORLD.BERRY_SNAPSHOT",
            "WORLD.AVATAR_SNAPSHOT",
        ]
    # Construction is sequential; restore the builder immediately afterward.
    with patch.object(builder, "builder", partial(builder.builder, env_seed=seed)):
        return substrate.build_from_config(config, roles=config.default_player_roles)


def berry_templates():
    from meltingpot.configs.substrates import allelopathic_harvest as base

    appearance = next(
        c["kwargs"]
        for c in base.PREFABS["berry_1"]["components"]
        if c["component"] == "Appearance"
    )
    templates = {}
    for index, (shape, palette) in enumerate(
        zip(appearance["spriteShapes"], appearance["palettes"])
    ):
        chars = [list(row) for row in shape.strip().splitlines()]
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=bool)
        for y, row in enumerate(chars):
            for x, char in enumerate(row):
                rgba = palette[char]
                if len(rgba) < 4 or rgba[3] != 0:
                    mask[y, x] = True
                    rgb[y, x] = rgba[:3]
        templates[(index % 3 + 1, index >= 3)] = (rgb, mask)
    return templates


def local_patches(observation, player_index, templates):
    """Expose only in-view patches whose complete berry sprite is visible."""
    avatar = observation["WORLD.AVATAR_SNAPSHOT"][player_index]
    if not avatar[3]:
        return [], {"reason": "focal_not_alive"}
    px, py, orientation = map(int, avatar[:3])
    rgb = observation["RGB"]
    world = observation["WORLD.RGB"]
    height, width = world.shape[0] // 8, world.shape[1] // 8
    forward, right = FORWARD[orientation], RIGHT[orientation]
    result, obscured = [], 0
    for bx, by, color, ripe in observation["WORLD.BERRY_SNAPSHOT"]:
        bx, by, color, ripe = map(int, (bx, by, color, ripe))
        # The view is smaller than the world, so each patch occurs at most once.
        locations = [
            (dx, dy)
            for dy in range(-1, 10)
            for dx in range(-5, 6)
            if (px + right[0] * dx + forward[0] * dy) % width == bx
            and (py + right[1] * dx + forward[1] * dy) % height == by
        ]
        if not locations:
            continue
        dx, dy = locations[0]
        tile = rgb[(9 - dy) * 8 : (10 - dy) * 8, (5 + dx) * 8 : (6 + dx) * 8]
        expected, mask = templates[(color, bool(ripe))]
        # Berries are configured noRotate. Require every opaque sprite pixel.
        if not np.array_equal(tile[mask], expected[mask]):
            obscured += 1
            continue
        result.append(
            {
                "dx": dx,
                "dy": dy,
                "color": COLORS[color],
                "ripe": bool(ripe),
                "world_position": [bx, by],
            }
        )
    result.sort(key=lambda p: (abs(p["dx"]) + abs(p["dy"]), p["dy"], p["dx"]))
    return result, {
        "obscured_or_ambiguous": obscured,
        "orientation": orientation,
        "position": [px, py],
    }


def describe_patches(patches):
    # World coordinates and other evaluator-only fields never enter this text.
    return [
        {"dx": p["dx"], "dy": p["dy"], "color": p["color"], "ripe": p["ripe"]}
        for p in patches
    ]


def sprite(shape, palette):
    chars = [list(row) for row in shape.strip().splitlines()]
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    for y, row in enumerate(chars):
        for x, char in enumerate(row):
            rgba = palette[char]
            if rgba[3]:
                mask[y, x] = True
                rgb[y, x] = rgba[:3]
    return rgb, mask


def avatar_templates():
    from meltingpot.configs.substrates import allelopathic_harvest as base
    from meltingpot.utils.substrates import shapes

    colors = [(125, 125, 125)] + list(base.COLORS)
    return {
        index: [
            sprite(shape, shapes.get_palette(color)) for shape in shapes.CUTE_AVATAR
        ]
        for index, color in enumerate(colors)
    }


def local_coordinates(position, focal, width, height):
    """Shortest toroidal displacement, expressed relative to focal orientation."""
    wx = (int(position[0]) - int(focal[0]) + width // 2) % width - width // 2
    wy = (int(position[1]) - int(focal[1]) + height // 2) % height - height // 2
    forward, right = FORWARD[int(focal[2])], RIGHT[int(focal[2])]
    return wx * right[0] + wy * right[1], wx * forward[0] + wy * forward[1]


def local_avatars(observation, player_index, templates, ids=False):
    focal = observation["WORLD.AVATAR_SNAPSHOT"][player_index]
    if not focal[3]:
        return [], 0
    height, width = (
        observation["WORLD.RGB"].shape[0] // 8,
        observation["WORLD.RGB"].shape[1] // 8,
    )
    result, obscured = [], 0
    for index, avatar in enumerate(observation["WORLD.AVATAR_SNAPSHOT"]):
        if not avatar[3]:
            continue
        dx, dy = local_coordinates(avatar, focal, width, height)
        if not (-5 <= dx <= 5 and -1 <= dy <= 9):
            continue
        tile = observation["RGB"][
            (9 - dy) * 8 : (10 - dy) * 8, (5 + dx) * 8 : (6 + dx) * 8
        ]
        if not any(
            np.array_equal(tile[mask], expected[mask])
            for expected, mask in templates[int(avatar[4])]
        ):
            obscured += 1
            continue
        result.append(
            {
                "dx": dx,
                "dy": dy,
                "public_color": COLORS.get(int(avatar[4]), "neutral"),
                "is_self": index == player_index,
                # v3 only: 1-based player number, to attribute events. Never rendered.
                **({"player": index + 1} if ids else {}),
            }
        )
    return sorted(result, key=lambda a: (a["dy"], a["dx"])), obscured


def scene(observation, player_index, berries, avatars, ids=False):
    patches, diagnostics = local_patches(observation, player_index, berries)
    visible_avatars, obscured = local_avatars(observation, player_index, avatars, ids)
    diagnostics["obscured_avatars"] = obscured
    return (
        patches,
        {
            "berries": describe_patches(patches),
            "avatars": visible_avatars,
            "alive": bool(observation["WORLD.AVATAR_SNAPSHOT"][player_index][3]),
        },
        diagnostics,
    )


def seen_events(events, player, visible):
    """Events one player saw during a step. `player` is 1-based; `visible` holds
    the 1-based numbers of the other players in its view before the step.
    Own planting and zapping are omitted: the memory already lists own actions."""
    seen = []
    for event in events:
        fields = event["fields"]
        if event["name"] == "eating":
            who = int(fields["player_index"])
            kind = "ate_self" if who == player else "ate_other" if who in visible else None
            color = COLORS[int(fields["berry_color"])]
        elif event["name"] == "replanting":
            who = int(fields["player_index"])
            kind = "planted_other" if who != player and who in visible else None
            color = COLORS[int(fields["target_berry"])]
        elif event["name"] == "zap":
            source, target = int(fields["source"]), int(fields["target"])
            kind = (
                "zapped_you"
                if target == player
                else "zap_other"
                if source != player and (source in visible or target in visible)
                else None
            )
            color = None
        else:
            continue
        if kind:
            seen.append({"kind": kind, "color": color} if color else {"kind": kind})
    return seen


def majority_colour(observation):
    """Most common berry colour on the whole map; None on a tie."""
    counts = np.bincount(
        observation["WORLD.BERRY_SNAPSHOT"][:, 2].astype(int), minlength=4
    )[1:]
    top = counts.max()
    return COLORS[int(counts.argmax()) + 1] if (counts == top).sum() == 1 else None


def scenario(local, taste, memory, frame, phases):
    """Coarse, language-independent situation label from game state only."""
    ripe = [b["color"] for b in local["berries"] if b["ripe"]]
    others = sum(not a["is_self"] for a in local["avatars"])
    return {
        "ripe_in_view": "own_colour"
        if taste in ripe
        else "other_colour"
        if ripe
        else "none",
        "neighbours_visible": "0" if others == 0 else "1-2" if others <= 2 else "3+",
        "zap_seen": "yes"
        if any(
            e["kind"] in ("zapped_you", "zap_other")
            for entry in memory
            for e in entry["events"]
        )
        else "no",
        "phase": next(
            name
            for name, (low, high) in zip(("early", "mid", "late"), phases)
            if low <= frame <= high
        ),
    }
