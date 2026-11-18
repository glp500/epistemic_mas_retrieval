# Preregistration: rule knowledge and reasoning in LLM players of a social dilemma (v4, block A)

Draft of 5 Nov 2026, laid out in the sections of the OSF Preregistration template. Text in [square brackets] is still to be filled in by the author before submission.

## 1. Study information

**Title.** Does telling LLM players how the game works, or letting them reason, change how they play a social dilemma?

**Authors.** Gavin Lip (Vrije Universiteit Amsterdam). Supervisor: Dr Victor de Boer

**Description.** Sixteen copies of one large language model play Melting Pot's *Allelopathic Harvest* through a text interface. Each player sees the rules, its own taste, the patches and players in view and a memory of its last 10 frames, and replies with one action. We vary two things: (1) one sentence added to the rules, which either explains how eating and ripening work or is an irrelevant control (a "sham" of similar length), and (2) whether the model may reason before it answers. We ask whether these change how many berries the players eat, beyond the effect of changing the text at all. The study runs separately on two open-weight models, Qwen3.5-27B and Gemma 4 31B.

**Hypotheses.** Each is tested separately for each model and is two-sided.

- **H1 (confirmatory).** Adding the mechanism sentence (how to eat a berry, and that a colour ripens faster when more patches on the map share it) changes the number of berries eaten per episode, compared with the sham sentence.
- **H2 (confirmatory).** Letting the model reason before acting (a thought of up to 1,024 tokens) changes the number of berries eaten per episode, compared with answering directly.
- **Secondary (not adjusted, descriptive).** The eating sentence alone and the ripening sentence alone, each compared with the sham, on berries eaten. All four comparisons on the secondary outcomes in section 4.

We expect knowledge to increase berries eaten, but we test in both directions, because a model can also act worse on information it misreads. We make no prediction about the difference between the two models: they differ in family as well as size, so a model difference is not a test of any single factor.

## 2. Design plan

**Study type.** Experiment: computer simulation with language-model agents. No human participants.

**Blinding.** Not applicable to the agents. The analysis is fully scripted (`harvest/analyze_v4.py`) and frozen at the commit named in section 6, before any main-run data exist.

**Design.** A 4 × 2 factorial between episodes, run for each of two models:

| Factor | Levels |
|---|---|
| Knowledge sentence | sham, eating, ripening, mechanism (eating + ripening) |
| Reasoning | off (answer directly), on (thought of up to 1,024 tokens, then answer) |
| Model (analysed separately) | Qwen3.5-27B, Gemma 4 31B |
| Seed (replicate) | 501, 502, 503 |

That gives 4 × 2 × 2 × 3 = **48 episodes**. In every episode, all 16 players are the same model with the same prompt. The seeds label the replicates and seed the simulator and sampler, but Melting Pot does not reproduce a world exactly from its seed (in v3, only 5 to 7 of 16 players started in the same place on a rerun). Episodes are therefore treated as independent replicates, not as matched across cells.

The sentences, in English (wording 0 of `configs/prompts_v4.json`), placed directly after the rules:

| Level | Sentence |
|---|---|
| Sham | "This game was first described in a 2020 research paper. It belongs to a collection of test environments for groups of learning agents." |
| Eating | "To eat a ripe berry, step onto its cell." |
| Ripening | "Unripe patches ripen faster when more patches on the map have the same color." |
| Mechanism | The eating and ripening sentences, on one line |

Everything else in the prompt is the v4 baseline: the v3 prompt with "These rules are correct:" as its first line, no time information and no false rule. The full prompt for every decision is logged.

**Randomisation.** Episodes run in an order shuffled with a fixed seed (`random.Random("A-main")` in `harvest/blocks.py`), so run order does not line up with any condition. Each model request draws its own sampling seed from the episode seed, so a player's draws do not depend on which other players share its batch.

## 3. Sampling plan

**Existing data.** Registration before any main-run data exist. The main seeds (501–503) have not been run. Earlier data were used only to check the code, measure speed and screen the sentences, and all used seed 500 or saved game states:

- **Smoke run** (5 Nov): 30 episodes of 20–50 frames, seed 500, every cell of every block. Used to check that replies parse (0% unmatched) and to save game states for the screen.
- **Fixed-state screen** (5 Nov): every prompt version on 30 saved states. All four block A levels passed the pre-set screening rule for at least one model and reasoning arm, so all four stay ([decisions.md](https://github.com/glp500/GameProbe/blob/a77c82d516b3bcc7d270c32ded27af19cb35b09b/docs/v4/decisions.md)).
- **Speed tests** (4–5 Nov), 20-frame episodes.

None of these produced outcome data on the main seeds, and none is used in the confirmatory analysis.

**Data collection.** Snellius (SURF) H100 GPUs, one GPU per episode, at most 4 episodes at once. The models run on vLLM 0.33.0 in bf16, inside an Apptainer container. Pinned model revisions: Qwen/Qwen3.5-27B `fc05daec18b0a78c049392ed2e771dde82bdf654`, google/gemma-4-31B-it `842da3794eaa0b77d5f08bae87a17459d91ff475`. Sampling follows each model card:

- Qwen: temperature 0.7, top-p 0.8, top-k 20 without thinking; 1.0, 0.95, 20 with thinking; presence penalty 1.5 in both.
- Gemma: temperature 1.0, top-p 0.95, top-k 64 throughout.

Game settings:

- Each episode is 1,500 frames.
- The action list is shown in a shuffled order for each player and frame.
- An unreadable reply gets one stricter retry, then counts as doing nothing.
- A thought that reaches 1,024 tokens is closed, and the model then answers.

**Sample size.** 48 episodes, 24 per model, with 3 per cell. The episode is the unit of analysis; the 16 players and 1,500 frames within an episode are not independent replicates.

**Sample size rationale.** The size is set by the compute budget (100,000 SBU), not by a power analysis. Measured speeds with reasoning on (35.4 s per frame for Qwen, 44.1 s for Gemma) put 48 episodes of 1,500 frames at about 79,000 SBU. Three seeds per cell is the smallest number at which the test below can reach p < 0.05: with two, its smallest possible p is 0.057. With three, the smallest possible p for H1 is 0.005, or 0.02 after Holm correction, so H1 can be confirmed only with a near-complete separation of the 3 mechanism episodes from the 3 sham episodes in both reasoning arms. The study can therefore detect only large effects. A non-significant result is not evidence that there is no effect.

**Stopping rule.** All 48 episodes run to completion, and nothing stops early on results. An episode that crashes or reaches the time limit is set aside (kept, not deleted) and rerun from the start with the same seed.

The first finished episodes are checked only for technical faults: crashes, unreadable replies and missing files. If a fault needs a code change, every episode run before the change is rerun after it, and the change is reported as a deviation.

## 4. Variables

**Manipulated variables.** The knowledge sentence (4 levels) and reasoning (2 levels), as in section 2.

**Measured variables.** All outcomes are computed per episode, over all frames and all 16 players.

- **Primary: berries eaten.** The number of eating events in the episode.
- **Secondary:**
  - welfare: the sum of all 16 players' rewards
  - Gini coefficient of per-player berry reward
  - cooperation: the share of planting actions that plant the map's majority colour
  - defection: the share of the 16 players who defected at least once, by zapping or by eating without ever replanting (0 to 1)
  - its two parts, each also a separate outcome:
    - zap rate: zaps per active decision
    - free-riding: the share of the 16 players who ate at least one berry but never replanted
  - removals by sanctioning
- **Diagnostics (reported, not tested):**
  - share of unmatched replies and retries
  - share of thoughts that reach the cap, and mean thought length
  - prompt length
  - seconds per frame

**Manipulation checks and probes (exploratory).** After the games, each model answers a fixed questionnaire on 8 saved states per episode, with the same prompt as in the game and no thinking. It covers:

- self-rated knowledge, before and after explaining how reward works
- three multiple-choice questions (how to eat, which patches ripen fastest, what planting changes), each followed by a confidence rating
- rule certainty
- the number of frames left

Questions and scoring are in [prompt-cells.md](https://github.com/glp500/GameProbe/blob/a77c82d516b3bcc7d270c32ded27af19cb35b09b/docs/v4/prompt-cells.md).

**Indices.** Defection combines the two behaviours on one scale, players, because they are counted in different units: zapping per action, free-riding per player over the episode. A player counts once, however often it zaps. Zapping is the game's sanction and can be aimed at free-riders; we count it as defection as in our earlier study (v3), and report it separately so it can be read either way. Other definitions are in `harvest/metrics.py` (`outcomes`) and `harvest/analyze_v4.py`.

## 5. Analysis plan

**Statistical model.** For each model and each comparison, the effect is the difference in mean berries eaten (level b minus level a), computed within each stratum and then averaged over strata:

- for H1 and the secondary knowledge comparisons, the strata are the two reasoning arms
- for H2, the strata are the four knowledge levels

The p-value comes from a two-sided permutation test that shuffles condition labels only within strata. It uses 20,000 random relabellings with a fixed generator seed (7304), and p = (hits + 1) / (draws + 1). Implementation: `stratified_p` in `harvest/analyze_v4.py`.

**Transformations.** None.

**Inference criteria.** The confirmatory family is H1 and H2 for each model: 4 tests on berries eaten, Holm-corrected at α = 0.05 across the 4. A hypothesis is supported for a model if its Holm-adjusted p is below 0.05. Every other p-value (eating and ripening against the sham, and all secondary outcomes) is unadjusted and descriptive. Every result is reported with the raw difference, the cell means and every episode's value.

**Data exclusion.** No episode is excluded on its outcomes. Unreadable replies stay in the data as "do nothing" actions, and their rate is reported per cell. A cell with more than 5% unmatched replies will be noted in the write-up, not excluded.

**Missing data.** Berries eaten is always defined. If nobody eats in an episode, the Gini coefficient is undefined; the episode is reported and kept for every other outcome.

**Exploratory analyses (not confirmatory):**
- end-game behaviour (last 20% of frames)
- actions in matched game situations (ripe berry in view, neighbours, zaps seen, phase)
- the content of thoughts: mentions of eating, ripening and own-colour reward, and whether the action named in a thought matches the action taken
- probe accuracy and calibration
- the screen results
- descriptive differences between the two models

## 6. Other

**Frozen materials.** Code, configuration (`configs/config_v4.yaml`), prompts (`configs/prompts_v4.json`) and the analysis are frozen at the commit tagged `v4-prereg` in the project repository (hash: [paste `git rev-parse v4-prereg` into the OSF form]). Every prompt and thought is logged per episode (`prompts.jsonl.gz`) and will be published with the data.

**Deviations.** Every decision and deviation is logged with its date and reason in `docs/v4/decisions.md`. Decisions made before this registration include:
- 1,500-frame episodes instead of 2,000, for budget
- skipping the planned pilot
- the screen verdict
- the choice of confirmatory tests

**Later blocks.** The design also has blocks for time information (B), rule certainty (H), a false rule (F) and language (D). None is part of this registration; each will get its own addendum before it runs. If blocks B or H run, their prompt will be the block A knowledge level with the most berries eaten, averaged over both models in block A's main data, with ties going to the shorter text.

**Known limitations:**
- With reasoning on, about 99% of thoughts reached the 1,024-token cap in testing. "Reasoning on" therefore means a capped, often unfinished thought.
- The game runs for 1,500 frames, so the late game is shorter than in the 2,000-frame original.
- The results come from one environment and two models.
- Sampling makes the episodes reproducible in distribution, not exactly.
