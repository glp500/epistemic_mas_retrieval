"""Contracts for v4: prompt cells, thought parsing, jobs, mixed players and analysis."""

import gzip
import json
from pathlib import Path

import pytest
import yaml

from harvest.prompts import BASELINE, parse_action, render, split_thought
from harvest.protocol import (
    block_jobs,
    check_cell,
    episode_id,
    full_cell,
    load_protocol,
    parse_players,
    validate_v4,
)

ROOT = Path(__file__).resolve().parents[1]
V3 = json.loads((ROOT / "configs/prompts_v3.json").read_text())
V4 = json.loads((ROOT / "configs/prompts_v4.json").read_text())
RAW = yaml.safe_load((ROOT / "configs/config_v4.yaml").read_text())
QWEN = RAW["models"]["qwen-27b"]["thinking"]
GEMMA = RAW["models"]["gemma-31b"]["thinking"]
ROW = {
    "taste": "red",
    "local_scene": {
        "alive": True,
        "berries": [{"dx": 1, "dy": 2, "color": "green", "ripe": True}],
        "avatars": [{"dx": 0, "dy": 0, "public_color": "neutral", "is_self": True}],
    },
}
ORDER = list(range(11))


def cell_prompt(frame=412, horizon=2000, **levels):
    return render(ROW, "en", "action_open", V4, [], 10, order=ORDER, cell=full_cell(**levels), frame=frame, horizon=horizon)


def test_v3_prompts_are_byte_identical_to_the_frozen_renderer():
    """Fixtures were written by the v3 code, before the refactor. The v4 baseline is
    the same prompt with the certain sentence as its first line (decision 6B)."""
    with gzip.open(ROOT / "tests/fixtures/v3_prompts.json.gz", "rt") as stream:
        cases = json.load(stream)
    assert len(cases) == 72
    for c in cases:
        row = {"taste": c["taste"], "local_scene": c["scene"]}
        kwargs = {"order": c["order"], "retry": c["retry"]} if c["task"] == "action_open" else {}
        assert render(row, c["language"], c["task"], V3, c["memory"], 10, **kwargs) == c["expected"]
        certain = V4["languages"][c["language"]]["certain"]
        assert render(row, c["language"], c["task"], V4, c["memory"], 10, **kwargs) == certain + "\n" + c["expected"]


def test_every_v4_language_and_wording_has_the_certain_sentence():
    for t in V4["languages"].values():
        assert t["certain"]
    for paraphrase in V4["languages"]["en"]["paraphrases"]:
        assert paraphrase["certain"] not in (None, "", V4["languages"]["en"]["certain"])


def test_v3_strings_are_untouched_in_v4():
    for language, t in V3["languages"].items():
        for key, value in t.items():
            assert V4["languages"][language][key] == value, (language, key)


@pytest.mark.parametrize(
    ("levels", "added"),
    [
        ({"knowledge": "eating"}, ["eating"]),
        ({"knowledge": "ripening"}, ["ripening"]),
        ({"knowledge": "sham"}, ["sham_knowledge"]),
        ({"certainty": "hedged"}, ["hedge"]),
        ({"certainty": "sham"}, ["sham_certainty"]),
        ({"false_rule": "once"}, ["false_rule"]),
        ({"false_rule": "sham"}, ["sham_false_rule"]),
    ],
)
def test_each_cell_adds_only_its_own_lines(levels, added):
    certain = V4["languages"]["en"]["certain"]  # a certainty level replaces this line
    base = [line for line in cell_prompt().split("\n") if line != certain]
    lines = cell_prompt(**levels).split("\n")
    extra = [V4["languages"]["en"][key] for key in added]
    assert [line for line in lines if line not in extra and line != certain] == base
    assert all(line in lines for line in extra)


def test_false_rule_repeats_three_times():
    rule = V4["languages"]["en"]["false_rule"]
    assert cell_prompt(false_rule="once").count(rule) == 1
    assert cell_prompt(false_rule="three").count(rule) == 3


def test_time_lines_use_the_true_horizon():
    assert "\nFrame 412.\n" in cell_prompt(time="frame")
    assert "\nFrame 412 of 2000 (1588 frames left).\n" in cell_prompt(time="horizon")
    assert "\nFrame 10 of 50 (40 frames left).\n" in cell_prompt(time="horizon", frame=10, horizon=50)
    assert "Frame" not in cell_prompt(time="none")


def test_notes_share_the_cell_context():
    note = render(ROW, "en", "note", V4, [], 10, cell=full_cell(knowledge="eating", time="frame"), frame=3, horizon=9)
    assert V4["languages"]["en"]["eating"] in note
    assert "Frame 3." in note


def test_untranslated_cells_stop_but_the_baseline_runs_everywhere():
    for language in RAW["languages"]:
        check_cell(V4, language, BASELINE)
    check_cell(V4, "en", full_cell(knowledge="mechanism", wording=2))
    with pytest.raises(ValueError, match="translation"):
        check_cell(V4, "ar", full_cell(knowledge="mechanism"))
    with pytest.raises(ValueError, match="paraphrases"):
        check_cell(V4, "ar", full_cell(wording=1))


def test_mechanism_states_the_real_game_rule():
    """Colours never spread on their own: patches ripen faster, and only planting recolours."""
    en = V4["languages"]["en"]
    for text in [en["ripening"], *(p["ripening"] for p in en["paraphrases"])]:
        assert "ripen" in text
        assert "spread" not in text


def test_mechanism_is_eating_plus_ripening_on_one_line():
    en = V4["languages"]["en"]
    assert en["eating"] + " " + en["ripening"] in cell_prompt(knowledge="mechanism").split("\n")


def test_every_paraphrase_keeps_the_same_fields_and_differs():
    en = V4["languages"]["en"]
    assert len(en["paraphrases"]) == 2
    for paraphrase in en["paraphrases"]:
        for key, text in paraphrase.items():
            assert text != en[key], key
            assert ("{frame}" in text) == ("{frame}" in en[key])
    assert "out of 2000" in cell_prompt(time="horizon", wording=1)


def test_long_names_are_not_mistaken_for_the_short_names_inside_them():
    options = ["unripe patches", "ripe patches", "any patch"]
    assert parse_action("The answer is unripe patches.", options) == 0
    assert parse_action("ripe patches, I think", options) == 1
    assert parse_action("unripe patches or ripe patches", options) is None
    for t in V3["languages"].values():  # the fix can't change how game actions parse
        names = [n.casefold() for n in t["actions"]]
        assert not any(a != b and a in b for a in names for b in names)


def test_action_names_inside_a_thought_are_ignored():
    names = V4["languages"]["en"]["actions"]
    reply = "plant red or fire zap? I will step left.</think>\n\nmove forward"
    thought, answer, closed = split_thought(reply, QWEN, prefilled=True)
    assert closed
    assert "fire zap" in thought
    assert parse_action(answer, names) == names.index("move forward")


def test_thought_formats_of_both_families():
    assert split_thought("no end yet", QWEN, prefilled=True) == ("no end yet", "", False)
    assert split_thought("<|channel>thought\nhm<channel|>turn left", GEMMA, prefilled=False) == ("hm", "turn left", True)
    assert split_thought("<|channel>thought\ncut off", GEMMA, prefilled=False) == ("cut off", "", False)
    assert split_thought("turn left", GEMMA, prefilled=False) == ("", "turn left", True)


def test_config_is_valid_and_loads_each_model():
    validate_v4(RAW)
    for key in RAW["models"]:
        protocol = load_protocol(ROOT / "configs/config_v4.yaml", [key])
        assert protocol["model_key"] == key and protocol["revision"]
    with pytest.raises(ValueError, match="Unknown model"):
        load_protocol(ROOT / "configs/config_v4.yaml", ["nope"])


@pytest.mark.parametrize(("block", "episodes"), [("A", 48), ("B", 24), ("H", 12), ("F", 24), ("D", 72)])
def test_block_sizes_match_the_plan(block, episodes):
    raw = {**RAW, "chosen_prompt": {"knowledge": "eating"}}
    jobs = block_jobs(raw, block)
    assert len(jobs) == episodes
    assert len({(j["model"], j["language"], j["seed"], *j["cell"].values()) for j in jobs}) == episodes


def test_blocks_that_need_the_chosen_prompt_wait_for_block_a():
    for block in ("B", "H"):
        with pytest.raises(ValueError, match="chosen_prompt"):
            block_jobs(RAW, block)
    assert block_jobs(RAW, "A")
    assert len(block_jobs({**RAW, "chosen_prompt": {}}, "B", 1)) == 12  # pilot and smoke runs


def test_shared_controls_are_paid_for_once():
    from harvest.protocol import job_key

    raw = {**RAW, "chosen_prompt": {"knowledge": "eating"}}
    jobs = {b: {job_key(j) for j in block_jobs(raw, b)} for b in raw["blocks"]}
    names = list(jobs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not jobs[a] & jobs[b], (a, b)


def test_episode_ids_name_the_cell_and_keep_v3_ids():
    assert episode_id({"protocol": "live-v3"}, "qwen", "en", 401, BASELINE) == "v3_qwen_en_401"
    cell = full_cell(reasoning="on", time="horizon")
    assert episode_id({"protocol": "live-v4"}, "qwen-9b", "en", 501, cell) == (
        "v4_qwen-9b_en_501_r-on_k-facts_c-certain_t-horizon_f-none_w-0"
    )


def test_players_split_evenly_over_taste_or_stop():
    players = parse_players(["qwen-2b:1-4,9-12", "qwen-27b:5-8,13-16"])
    assert players[1] == players[12] == "qwen-2b"
    assert players[5] == players[16] == "qwen-27b"
    with pytest.raises(ValueError, match="4 red"):
        parse_players(["a:1-8", "b:9-16"])
    with pytest.raises(ValueError, match="twice"):
        parse_players(["a:1-4,9-12", "b:4-8,13-16"])
    with pytest.raises(ValueError, match="assigned"):
        parse_players(["a:1-4,9-12"])


def fake_table():
    import pandas as pd

    rows = []
    for model in ("m1", "m2"):
        for seed, noise in enumerate((-1, 0, 1)):
            for level, effect in (("off", 0), ("on", 100)):
                rows.append({"model": model, "language": "en", **BASELINE, "reasoning": level, "seed": seed, "welfare": effect + noise})
    return pd.DataFrame(rows)


def test_stratified_permutation_finds_a_real_effect_and_no_false_one():
    from harvest.analyze_v4 import stratified_p

    difference, p = stratified_p(fake_table(), "reasoning", "off", "on", "welfare", 2000)
    assert difference == pytest.approx(100)
    assert p < 0.01
    flat = fake_table().assign(welfare=1.0)
    assert stratified_p(flat, "reasoning", "off", "on", "welfare", 200) == (0.0, 1.0)


def test_holm_and_end_game_window():
    from harvest.analyze_v4 import end_game_shares, holm

    assert holm([0.01, 0.04, float("nan")])[:2] == pytest.approx([0.02, 0.04])
    rows = [{"frame": f, "action": 7 if f >= 80 else 8, "taste": "red"} for f in range(100)]
    shares = end_game_shares(rows, 100, 0.2)
    assert shares["end_zap"] == 1.0
    assert shares["end_plant_own_colour"] == 0.0


def test_analysis_runs_on_a_fake_reasoning_episode(tmp_path):
    from harvest.analyze_v4 import episode_row

    folder = tmp_path / "episode"
    folder.mkdir()
    (folder / "summary.json").write_text('{"seconds_per_frame": 2.0}')
    names = V4["languages"]["en"]["actions"]
    with gzip.open(folder / "prompts.jsonl.gz", "wt") as log:
        for frame in (0, 1):
            thought = "I will step onto the berry, so move forward." if frame == 0 else "plant red"
            log.write(json.dumps({"frame": frame, "player": 1, "prompts": ["p"], "thoughts": [thought]}) + "\n")
    scenario = {"ripe_in_view": "none", "neighbours_visible": "0", "zap_seen": "no", "phase": "early"}
    actions = [
        {"frame": f, "player": 1, "alive": True, "action": 1, "taste": "red", "model": "m", "reward": 0.0, "majority_colour": None,
         "retried": False, "unmatched": False, "truncated": f == 1, "thought_tokens": [50], "prompt_tokens": 900, "scenario": scenario}
        for f in (0, 1)
    ]
    info = {
        "status": "complete", "episode_id": "e", "stage": "smoke", "language": "en", "seed": 500, "block": "A",
        "protocol": {"model_key": "m"}, "cell": full_cell(reasoning="on"), "model_runtime": {"gpu": "test"},
    }
    row = episode_row({"info": info, "folder": folder, "actions": actions, "events": []}, RAW, V4)
    assert row["block"] == "A" and row["reasoning"] == "on" and row["gpu"] == "test"
    assert row["truncated_rate"] == 0.5
    assert row["trace_mentions_eating"] == 0.5
    assert row["trace_action_agreement"] == 1.0  # the cut-off thought is left out
    assert row["traces_agreement_n"] == 1
    assert names[1] == "move forward"


def test_probe_helpers_and_state_selection(tmp_path):
    from harvest.probe_v4 import entropy, frames_left, logged_prompts, pick_states, rating

    assert frames_left("About 1588 frames.") == 1588
    assert frames_left("unknown") is None

    assert rating("I'd say 5.") == 5
    assert rating("no digit") is None
    assert entropy([0.0] * 11) == pytest.approx(3.4594, abs=1e-3)
    assert entropy([0.0, -100.0]) < 0.01
    rows = [{"frame": f, "player": 1, "alive": True, "scenario": {"ripe_in_view": "own_colour"}} for f in range(100)]
    assert pick_states(rows, {"ripe_in_view": "own_colour"}, 4) == [(0, 1), (25, 1), (50, 1), (75, 1)]
    with gzip.open(tmp_path / "prompts.jsonl.gz", "wt") as log:
        log.write(json.dumps({"frame": 3, "player": 2, "prompts": ["first", "retry"]}) + "\n")
    assert logged_prompts(tmp_path, [(3, 2)]) == {(3, 2): "first"}


def test_every_cell_renders_and_the_logged_scene_rebuilds_the_prompt():
    """2.4: what a run logs is enough to re-render any decision."""
    for knowledge in RAW["conditions"]["knowledge"]:
        for certainty in RAW["conditions"]["certainty"]:
            for time in RAW["conditions"]["time"]:
                for false_rule in RAW["conditions"]["false_rule"]:
                    cell = full_cell(knowledge=knowledge, certainty=certainty, time=time, false_rule=false_rule, wording=1)
                    prompt = render(ROW, "en", "action_open", V4, [], 10, order=ORDER, cell=cell, frame=5, horizon=50)
                    logged = json.loads(json.dumps({"scene": ROW, "memory": []}))
                    again = render(logged["scene"], "en", "action_open", V4, logged["memory"], 10, order=ORDER, cell=cell, frame=5, horizon=50)
                    assert prompt == again


def test_job_lines_parse_into_the_planned_episode():
    from harvest.blocks import play_arguments
    from harvest.play import parse_arguments

    job = block_jobs(RAW, "A")[-1]
    identity, _, line = play_arguments(job, Path("configs/config_v4.yaml"), "outputs/v4", 2000)
    args = parse_arguments(line.split())
    assert (args.model, args.seed, args.reasoning, args.steps) == ("gemma-31b", 503, "on", 2000)
    assert identity in str(args.output)


def test_every_block_uses_only_the_two_large_models_on_h100():
    for block in RAW["blocks"].values():
        assert block["models"] == ["qwen-27b", "gemma-31b"]
        assert block["gpu"] == "gpu_h100"
    assert set(RAW["models"]) == {"qwen-27b", "gemma-31b"}


def test_rerunning_a_block_never_touches_a_running_episode(tmp_path):
    from harvest.blocks import episode_state, set_aside

    folder = tmp_path / "episode"
    assert episode_state(folder, 2000) == "new"
    folder.mkdir()
    assert episode_state(folder, 2000) == "incomplete"
    (folder / "manifest.json").write_text('{"status": "running", "horizon": 2000}')
    assert episode_state(folder, 2000) == "incomplete"
    set_aside(folder)
    assert (tmp_path / "episode.failed-1/manifest.json").exists()
    assert episode_state(folder, 2000) == "new"
    folder.mkdir()
    (folder / "manifest.json").write_text('{"status": "complete", "horizon": 50}')
    assert episode_state(folder, 50) == "complete"
    with pytest.raises(SystemExit, match="HARVEST_OUTPUT"):
        episode_state(folder, 2000)


class FakeModel:
    """Answers by rule, so screen and probe logic can be tested without a GPU."""

    def __init__(self, action="move forward"):
        self.action = action
        self.seen = []

    def generate(self, prompts, tokens):
        self.seen += prompts
        last = prompts[-1]
        text = last if isinstance(last, str) else last[-1]["content"][0]["text"]
        if "Available actions" in text:
            answer = "plant red" if "ripen faster" in text else self.action
        elif "one digit" in text:
            answer = "6"
        elif "How many frames" in text:
            answer = "1500"
        elif "exactly one option" in text:
            answer = "The answer is " + ("stand next to it" if "next to it; " in str(self.seen[0]) else text.split(": ")[-1].split(", ")[0].rstrip(".")) + "."
        else:
            answer = "Step onto ripe berries of your preferred color."
        return [{"answer": answer, "prompt_tokens": 10} for _ in prompts]


def fake_protocol():
    return {**RAW, "model_key": "m"}


def fake_bank():
    state = {"frame": 500, "scene": ROW, "memory": [], "scenario": {}}
    return [{**state, "state": i} for i in range(2)]


def test_screen_covers_every_level_in_every_wording_once():
    from harvest.screen import screen_cells

    cells = screen_cells(RAW["conditions"])
    assert len(cells) == 1 + 3 * (4 + 2 + 3 + 3) == 37
    assert len({tuple(c.values()) for c in cells}) == 37
    assert cells[0] == BASELINE


def test_states_are_spread_over_scenarios():
    from harvest.screen import pick_states as pick

    def row(ripe):
        return {"scenario": {"ripe_in_view": ripe, "neighbours_visible": "0", "zap_seen": "no", "phase": "early"}}

    rows = [row("none")] * 50 + [row("own_colour")] * 2
    picked = pick(rows, 4)
    assert sum(r["scenario"]["ripe_in_view"] == "own_colour" for r in picked) == 2


def test_screen_gives_every_cell_the_same_action_lists():
    from harvest.screen import list_order, run_cell

    assert list_order(3, 1) == list_order(3, 1)
    assert list_order(3, 0) != list_order(3, 1)
    model = FakeModel()
    rows = run_cell(model, full_cell(knowledge="ripening"), False, fake_bank(), fake_protocol(), V4, 2)
    assert len(rows) == 4
    assert {r["action"] for r in rows} == {V4["languages"]["en"]["actions"].index("plant red")}


def test_a_level_passes_only_beyond_its_sham_in_all_wordings():
    from harvest.screen import report_tables

    def rows(cell, actions):
        return [{"kind": "decision", "model": "m", "arm": "off", "cell": cell, "state": s, "order": 0, "taste": "red", "action": a}
                for s, a in enumerate(actions)]

    data = rows(BASELINE, [0] * 10)
    for w in (0, 1, 2):
        data += rows(full_cell(knowledge="sham", wording=w), [1] + [0] * 9)
        data += rows(full_cell(knowledge="eating", wording=w), [1] * 6 + [0] * 4)
        data += rows(full_cell(knowledge="ripening", wording=w), ([1] * 6 if w else [1]) + [0] * (4 if w else 9))
    _, _, verdict = report_tables(data, 0.05)
    outcome = dict(zip(verdict.level, verdict.passes))
    assert outcome == {"eating": True, "ripening": False}  # a sentence must work in every wording


def test_a_margin_of_exactly_the_threshold_passes():
    from harvest.screen import report_tables

    def rows(cell, changed):
        return [{"kind": "decision", "model": "m", "arm": "off", "cell": cell, "state": s, "order": 0, "taste": "red", "action": int(s < changed)}
                for s in range(20)]

    data = rows(BASELINE, 0)
    for w in (0, 1, 2):
        data += rows(full_cell(knowledge="sham", wording=w), 5) + rows(full_cell(knowledge="eating", wording=w), 6)  # 0.30 - 0.25
    assert report_tables(data, 0.05)[2].passes.tolist() == [True]


def test_holm_covers_only_confirmatory_primary_tests():
    import pandas as pd

    from harvest.analyze_v4 import comparisons

    rows = [{"block": "A", "model": "m", "language": "en", **full_cell(knowledge=k, reasoning=r), "berries_eaten": i + (k == "mechanism"), "welfare": i}
            for k in ("sham", "eating", "mechanism") for r in ("off", "on") for i in range(3)]
    protocol = {"analysis": {"primary": "berries_eaten", "secondary": ["welfare"], "resamples": 200, "comparisons": [
        {"blocks": ["A"], "factor": "knowledge", "a": "sham", "b": "eating"},
        {"blocks": ["A"], "factor": "knowledge", "a": "sham", "b": "mechanism", "confirmatory": True}]}}
    frame = comparisons(pd.DataFrame(rows), protocol)
    adjusted = frame.dropna(subset=["p_holm"])
    assert list(zip(adjusted.b, adjusted.outcome)) == [("mechanism", "berries_eaten")]


def test_defectors_are_players_who_zapped_or_ate_without_replanting():
    from harvest.metrics import ZAP, outcomes

    def act(player, action):
        return {"player": player, "action": action, "alive": True, "reward": 0, "frame": 0, "majority_colour": None}

    def event(name, player):
        return {"name": name, "fields": {"player_index": player, "berry_color": 1}}

    episode = {
        "info": {"protocol": {"model_key": "m"}, "language": "en", "seed": 1, "episode_id": "e", "stage": "smoke"},
        "actions": [act(1, ZAP), act(1, ZAP), act(2, 0), act(3, 0), act(4, ZAP)],
        "events": [event("eating", 2), event("eating", 3), event("replanting", 3), event("eating", 1)],
    }
    result = outcomes(episode)
    assert result["free_rider_share"] == 2 / 16  # players 1 and 2 ate without replanting; 3 replanted
    assert result["defector_share"] == 3 / 16  # 1 zapped and free-rode (counted once), 2 free-rode, 4 zapped


def test_probe_scores_each_question_and_the_checks():
    from harvest.analyze_v4 import question_measures
    from harvest.probe_v4 import conversation

    row = conversation(FakeModel(), "Rules.", V4["languages"]["en"]["probe"], 80, 0, 1500)
    assert row["rating_before"] == 6 and row["rule_certainty"] == 6
    assert row["frames_left_said"] == row["frames_left_true"] == 1500
    assert [q["correct"] for q in row["diagnostic"]] == [True, True, True]
    assert row["diagnostic"][0]["believes_false_rule"] is False
    measures = question_measures([row])
    assert measures["probe_accuracy"] == 1.0
    assert measures["probe_comprehension"] == 1.0
    assert measures["probe_confidence"] == pytest.approx(5 / 6)
    assert measures["probe_calibration_gap"] == pytest.approx(5 / 6 - 1)


def test_belief_in_the_false_rule_is_scored_apart_from_being_wrong():
    from harvest.probe_v4 import conversation

    context = "A rule: " + V4["languages"]["en"]["false_rule"].replace("You eat a ripe berry by standing next to it", "next to it")
    row = conversation(FakeModel(), context, V4["languages"]["en"]["probe"], 80, 0, 1500)
    assert row["diagnostic"][0]["believes_false_rule"] is True
    assert row["diagnostic"][0]["correct"] is False


def test_explanations_are_scored_by_the_rubric():
    from harvest.analyze_v4 import explanation_score

    keywords = RAW["analysis"]["mechanism_keywords"]["en"]
    assert explanation_score("Step onto ripe berries of my color.", keywords) == pytest.approx(2 / 3)
    assert explanation_score("Plant things.", keywords) == 0


def test_presence_penalty_counts_reply_tokens_once():
    torch = pytest.importorskip("torch")

    from harvest.llm import presence_penalty

    (penalty,) = presence_penalty(1.5, start=2)
    ids = torch.tensor([[7, 8, 3, 3, 4]])  # prompt 7 8, reply 3 3 4
    scores = penalty(ids, torch.zeros(1, 10))
    assert scores[0].tolist() == [0, 0, 0, -1.5, -1.5, 0, 0, 0, 0, 0]
    assert penalty(ids[:, :2], torch.zeros(1, 10)).abs().sum() == 0  # nothing generated yet
