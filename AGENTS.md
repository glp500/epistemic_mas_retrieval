# Project rules

- Preserve v4 Block A's prompts, model pins, simulator and statistical endpoints.
- Mark historical records and new analyses separately. Keep the episode as the statistical unit.
- Keep raw logs, caches, environments and cluster details outside Git.
- Prefer Miniforge, Conda and mamba. Use the frozen container for historical GPU reruns.
- Reproduce results on the CPU before considering GPU reruns.
