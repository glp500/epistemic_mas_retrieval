#!/bin/bash
# One episode per array task. The job file comes from `slurm/py -m harvest.blocks <block>`,
# which also prints the matching sbatch line (array size, partition, wall time).
#SBATCH --job-name=harvest
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
arguments=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
# shellcheck disable=SC2086  # the job line is a list of arguments
slurm/py -m harvest.play $arguments
