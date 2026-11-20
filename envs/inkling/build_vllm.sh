#!/bin/bash
set -e
export RUSTUP_HOME=/workspace/cache/rust/rustup CARGO_HOME=/workspace/cache/rust/cargo
export PATH=/workspace/cache/rust/cargo/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=96 NVCC_THREADS=2 VLLM_TARGET_DEVICE=cuda
export CPATH=/workspace/cache/sysroot/usr/include:${CPATH:-}
export UV_CACHE_DIR=/workspace/cache/uv TMPDIR=/workspace/mats_projects/loop_steer/scratch/tmp CCACHE_DIR=/workspace/cache/ccache
mkdir -p $TMPDIR
cd /workspace/src/vllm-inkling
date
/workspace/mats_projects/loop_steer/.venv-inkling/bin/python -m build --wheel --no-isolation --outdir /workspace/cache/wheels .
date
