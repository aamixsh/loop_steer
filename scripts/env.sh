# Keep every tool cache and temp file inside this checkout. Source it in each shell on a machine without
# the lab storage profile (e.g. a rented GPU box):   source scripts/env.sh
# Layout: data/ (run artifacts and caches) and scratch/ (temp) are plain directories here; both are git-ignored.
# Do not source it on the shared lab server, whose profile already decides where caches go.
# The Hugging Face, uv and uv-Python locations keep a value that is already set (e.g. a cache shared by several
# projects, exported from ~/.bashrc); only when unset do they default to data/.cache.
LOOP_STEER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export LOOP_STEER_ROOT
C="$LOOP_STEER_ROOT/data/.cache"
export HF_HOME="${HF_HOME:-$C/huggingface}"
export XDG_CACHE_HOME="$C/xdg"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$C/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$C/uv-python}"
export TRITON_CACHE_DIR="$C/triton"
export TORCH_HOME="$C/torch"
export VLLM_CACHE_ROOT="$C/vllm"
export TORCHINDUCTOR_CACHE_DIR="$C/torchinductor"
export TMPDIR="$LOOP_STEER_ROOT/scratch"
mkdir -p "$C/bin" "$TMPDIR"
case ":$PATH:" in *":$C/bin:"*) ;; *) export PATH="$C/bin:$PATH" ;; esac
umask 077
unset C
