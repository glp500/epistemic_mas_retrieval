"""Language, reasoning and model size in Melting Pot's Allelopathic Harvest."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Set before torch, transformers or the simulator load. Override any of these from the shell.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# vLLM: PyTorch top-k/top-p sampling. FlashInfer's sampler compiles CUDA code at run time,
# and the container has no CUDA compiler.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
