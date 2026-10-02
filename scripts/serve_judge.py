#!/usr/bin/env python3
"""Serve the StrongREJECT judge (gpt-oss-20b) on one physical GPU at 127.0.0.1:8001."""

import argparse
import os
import sys

from loop_steer.judge import JUDGE_MODEL
from loop_steer.paths import setup_job_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="1", help="Physical GPU index or UUID (default: 1)")
    args, vllm_args = parser.parse_known_args()
    if not args.gpu or "," in args.gpu or args.gpu.startswith("-"):
        parser.error("--gpu must select exactly one GPU")

    setup_job_env()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", JUDGE_MODEL,
        "--served-model-name", JUDGE_MODEL,
        "--host", "127.0.0.1",
        "--port", "8001",
        "--tensor-parallel-size", "1",
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.45",
        *vllm_args,
    ]
    print(f"Serving judge {JUDGE_MODEL} on physical GPU {args.gpu}", flush=True)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
