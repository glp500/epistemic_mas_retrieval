# Migration provenance

The collection code, frozen v4 configuration, prompt files, screen state bank,
GPU package lock and test fixtures originate from
[GameProbe source commit a77c82d](https://github.com/glp500/GameProbe/tree/a77c82d516b3bcc7d270c32ded27af19cb35b09b).
The live simulator, prompt construction and generation code are unchanged.
The saved episode, cell and comparison tables come from the original Block A
analysis. The map reconstruction and supporting data include subsequent
exploratory work from the original project. This repository publishes saved
results and analysis inputs alongside the core experiment code.

The v3 analysis module supplied shared outcome and fixed-state screen helpers.
The extraction moves those same function bodies to `harvest/metrics.py`.
The incomplete Block D embedding and analysis workflow stays in the source
project. Frozen v3 prompt strings remain as test fixtures for v4 prompt
compatibility. The retained v4 configuration includes options from the
original broader protocol, such as time cues and certainty.

`harvest.analyze_v4 --outcomes` uses the same comparison and report code as
raw-log analysis. This permits reproduction of the main statistics directly
from the bundled CSV. `harvest.coordination` accepts explicit input and output
directories and preserves the original frame alignment and patch count logic.

The raw archive is `outputs/v4/main/blockA` in the original workspace. Figures
and the supplemental comparison tables are historical exports. The
`coordination_trajectories.csv` in the original repository covers states 0 to
1,500 for all episodes and can also be regenerated from the raw data.
The public copy omits private machine paths, raw prompts and traces,
large raw outputs, historical environments and cluster-specific setup.
