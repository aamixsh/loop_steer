#!/usr/bin/env python3
"""Serve the pinned Ouro model on one physical GPU."""

import argparse
import os
import sys

from loop_steer import MODEL_ID, MODEL_REVISION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="1", help="Physical GPU index or UUID (default: 1)")
    args, vllm_args = parser.parse_known_args()
    if not args.gpu or "," in args.gpu or args.gpu.startswith("-"):
        parser.error("--gpu must select exactly one GPU")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Keep vLLM's compile cache on /data instead of the backed-up home directory.
    data_dir = os.environ.get("DATA_DIR", f"/data/{os.environ.get('USER', '')}")
    os.environ.setdefault("VLLM_CACHE_ROOT", f"{data_dir}/.cache/vllm")
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL_ID,
        "--revision", MODEL_REVISION,
        "--code-revision", MODEL_REVISION,
        "--trust-remote-code",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--dtype", "bfloat16",
        "--tensor-parallel-size", "1",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.2",
        "--enforce-eager",
        *vllm_args,
    ]
    print(f"Serving {MODEL_ID} on physical GPU {args.gpu}", flush=True)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
