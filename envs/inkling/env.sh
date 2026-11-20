# Source before running anything in .venv-inkling (Inkling 2 Mini; see envs/inkling/README.md).
export HF_HOME=/workspace/cache/hf UV_CACHE_DIR=/workspace/cache/uv
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH  # flashinfer JIT compiles with nvcc
export PY_INKLING=/workspace/mats_projects/loop_steer/.venv-inkling/bin/python
