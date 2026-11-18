# Descriptive policy baselines

The later baseline evaluation used the same 16-player Allelopathic Harvest
substrate, 1,500-frame horizon and outcome definitions as Block A. The results
are descriptive references outside the planned LLM comparisons.

The [random policy](../results/random_baseline.csv) samples among the 11 native
actions uniformly for each active player. The
[selected policy](../results/selected_baseline.csv) uses local observations to
move toward ripe berries, explore and plant red. A finite search evaluated
16 policies on three training seeds each. Selection maximized mean berries
eaten. The frozen policy uses planting probability 0.6 and exploration
probability 0.05. Three separate seed labels were used for evaluation.

| Policy | Mean berries eaten | Mean welfare | Mean berry-reward Gini |
|---|---:|---:|---:|
| Random | 210.0 | −195.3 | 0.2039 |
| Selected | 811.7 | 1,209.7 | 0.1919 |

The independent LLM and policy environments were initialized separately.
Shared seed labels do not guarantee matched worlds. The search provides a
feasible comparator from the policies evaluated and does not establish a
harvest ceiling, an equilibrium or the causal effect of a shared convention.
