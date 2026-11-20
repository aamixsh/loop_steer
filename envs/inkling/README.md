# Inkling 2 Mini environment (`.venv-inkling`)

Inkling 2 Mini (`thinkingmachines/Inkling-2-Mini`, pinned in `src/loop_steer/models.py`) is not in any vLLM or
transformers release yet (Nov 20 2026), so it runs in its own environment, separate from the locked `.venv`:

- torch 2.14.0+cu130 (driver 580 = CUDA 13.0)
- vLLM built from vllm-project/vllm#69028 at `b426b46e63408c41b0aec069fdaf24dd640abe58` (it already contains the SM120
  shared-memory handling that #69021 adds to main), for `TORCH_CUDA_ARCH_LIST=12.0` (RTX PRO 6000 Blackwell)
- transformers 5.23.0 (tokenizer only; vLLM requires < 5.24). The HF modeling code (huggingface/transformers#51050)
  dequantizes the experts to bf16, so we do not use it: every Inkling run, activation capture included, goes through
  vLLM with the checkpoint's NVFP4 / MXFP8 experts (`loop_steer.vllm_inkling`).

`requirements.lock.txt` is `uv pip freeze` of the working environment.

## Rebuild

```bash
# sources and build tools, all under /workspace
mkdir -p /workspace/src && cd /workspace/src
git init vllm-inkling && cd vllm-inkling && git remote add origin https://github.com/vllm-project/vllm.git
git fetch --depth 50 origin pull/69028/head:pr69028 && git checkout pr69028   # expect b426b46e6
export RUSTUP_HOME=/workspace/cache/rust/rustup CARGO_HOME=/workspace/cache/rust/cargo
curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal
# DeepGEMM (built although Inkling does not use it) needs elfutils headers: unpack them, no system install
mkdir -p /workspace/cache/sysroot/debs && cd /workspace/cache/sysroot/debs && apt-get download libdw-dev libelf-dev \
  && for d in *.deb; do dpkg -x $d /workspace/cache/sysroot; done

cd /workspace/mats_projects/loop_steer
uv venv .venv-inkling --python 3.12
uv pip install --python .venv-inkling/bin/python torch==2.14.0 torchvision==0.29.1 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv-inkling/bin/python "cmake>=3.26.1" ninja "packaging>=24.2" "setuptools>=78.1.1,<81" \
  "setuptools-scm>=8" "setuptools-rust>=1.9.0" wheel "jinja2>=3.1.6" regex build "protobuf>=5.29.6"
bash envs/inkling/build_vllm.sh            # ~1 h on 256 cores; wheel -> /workspace/cache/wheels
uv pip install --python .venv-inkling/bin/python /workspace/cache/wheels/vllm-*.whl -r envs/inkling/requirements.lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
uv pip install --python .venv-inkling/bin/python -e . --no-deps
```

## Use

`source envs/inkling/env.sh`, then run the usual scripts with `$PY_INKLING` (generation, `extract_activations.py`,
`intervene.py`); judging, `build_sets.py`, `directions.py` and the analysis scripts run in `.venv` as before.

Model use follows the release: the official chat template (reasoning effort 0.9, its default; no generation prefix,
the model writes its own `<|message_model|>`), sampling temperature 1.0 and top_p 1.0 (`generation_config.json`), text
only (`limit_mm_per_prompt={"image": 0}`), no DSpark speculative decoding (it would change nothing in distribution but
reads hidden states we intervene on).
