"""Download every pinned model into the Hugging Face cache. Run on a login node:

    HF_HUB_OFFLINE=0 .venv/bin/python -m harvest.download

Compute nodes then load with local_files_only: true.
"""

import argparse
from pathlib import Path

import yaml

from harvest import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/config_v4.yaml")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    raw = yaml.safe_load(args.protocol.read_text())
    for key, entry in raw["models"].items():
        print(key, snapshot_download(entry["model_id"], revision=entry["revision"]), flush=True)


if __name__ == "__main__":
    main()
