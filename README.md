# Epistemic information and collective outcomes in LLM populations

How does giving agents task-relevant rules change what their population achieves
when individuals prefer different outcomes? In [Melting Pot’s Allelopathic
Harvest](https://proceedings.mlr.press/v139/leibo21a.html), players can learn
**which action eats a berry** and **how the colour of planted patches affects
ripening**. Block A varies whether those two rules appear in the prompt. It
independently varies direct answers versus capped reasoning.

This repository preserves the completed v4 Block A study from
[GameProbe](https://github.com/glp500/GameProbe/tree/a77c82d516b3bcc7d270c32ded27af19cb35b09b).
Each of 48 English episodes runs for 1,500 frames with 16 copies of either
Qwen3.5-27B or Gemma 4 31B. The design crosses four information conditions
(sham, eating, ripening, both) with both reasoning settings and three seeds per
cell. The intervention is explicit rule text in the prompt. Recent action and
event memory is common to all conditions.

**Capped reasoning increased the number of berries eaten in both models.**
Effects of the rule text varied by model. Planting attempts overwhelmingly
matched each player’s private colour preference, and the patches remained mixed.
See the [results](results/README.md), [paper draft](docs/project-report-draft.md)
and [original study plan](docs/preregistration.md).

## Reproduce the statistics without a GPU

From the repository root, using Miniforge:

```bash
mamba env create -f environment.yml
conda activate epistemic-mas
python -m pytest -q
python -m harvest.analyze_v4 --outcomes results/outcomes.csv --output outputs/reproduced
```

The command regenerates the cell means, stratified permutation comparisons,
Holm adjustments and Markdown report from the 48 saved episode rows.
The optional PyTorch backend test runs when PyTorch is installed.

## Reanalyse original logs

If you have the original GameProbe outputs, pass the directory that contains
only the 48 completed Block A episode folders. Each folder needs its manifest,
summary, actions and events. The thought statistics also read the original
compressed prompt log.

```bash
python -m harvest.analyze_v4 --input /path/to/outputs/v4/main/blockA --output outputs/from-logs
```

For the exploratory map reconstruction, use Python with the original
Melting Pot 2.4.1 installed. It reads the initial executable map and reconstructs
each patch colour count from the saved recolouring events.

```bash
python -m harvest.coordination --input /path/to/outputs/v4/main/blockA --output outputs/coordination
```

## Rerun an episode

The historic model and simulator use the frozen Python 3.13 Apptainer runtime
with the packages in `requirements-vllm-lock.txt`. Reuse the image, GPU Python
environment and pinned model cache from the original run. The launcher
`slurm/py` expects `harvest.sif` and `.venv` in this checkout. On an allocated
GPU, point `HF_HOME` at the existing cache and run, for example:

```bash
slurm/py -m harvest.play --model qwen-27b --language en --seed 501 \
    --block A --knowledge eating --reasoning off --steps 1500 --stage main \
    --output outputs/replay-qwen-501
```

`python -m harvest.blocks A` prepares the original 48-episode job list and
prints an example Slurm command. It submits nothing. Review the historical
wall time and your allocation before using it. Identical seed labels do not
guarantee identical simulator worlds.

| Location | Role |
|---|---|
| `configs/` | Frozen protocol, prompts and fixed-state screen bank |
| `harvest/play.py`, `game.py`, `prompts.py`, `llm.py` | Joint simulator steps, observations, prompt construction and model decoding |
| `harvest/analyze_v4.py`, `metrics.py`, `coordination.py` | Episode outcomes, tests and exploratory map reconstruction |
| `harvest/screen.py`, `probe_v4.py` | Historical supporting diagnostics |
| `results/` | Saved tables, figures and verification records |
| `docs/` | Methods, research background and evidence provenance |

The archived protocol retains historical options beyond Block A. The
extraction keeps the shared metric definitions and removes the old
cross-language embedding and analysis entry points. The large raw logs
remain in the original archive. See [migration provenance](docs/migration.md).
