#!/usr/bin/env python3
"""Download the models used here at the pinned revisions into the Hugging Face cache.

Qwen3-8B (about 16 GB), Ouro-1.4B-Thinking (3 GB), Nanbeige4.2-3B (8 GB) and the judge gpt-oss-20b (13 GB).
Set HF_HOME / HF_HUB_CACHE first to choose where they go. Re-running skips files that are already there.
    scripts/prefetch_models.py [--skip qwen ouro nanbeige judge]
"""

import argparse

from loop_steer.paths import setup_job_env

setup_job_env()  # before importing huggingface_hub: caches go under data/.cache unless the environment says otherwise

from huggingface_hub import snapshot_download  # noqa: E402

from loop_steer import MODEL_ID as OURO_ID
from loop_steer import MODEL_REVISION as OURO_REVISION
from loop_steer.judge import JUDGE_MODEL, JUDGE_REVISION
from loop_steer.models import NANBEIGE_ID, NANBEIGE_REVISION, QWEN_ID, QWEN_REVISION

MODELS = {
    "qwen": (QWEN_ID, QWEN_REVISION),
    "ouro": (OURO_ID, OURO_REVISION),
    "nanbeige": (NANBEIGE_ID, NANBEIGE_REVISION),
    "judge": (JUDGE_MODEL, JUDGE_REVISION),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip", nargs="*", default=[], choices=MODELS)
    args = ap.parse_args()
    for key, (repo, revision) in MODELS.items():
        if key in args.skip:
            continue
        print(f"{repo} @ {revision[:8]} ...", flush=True)
        # the judge repo also ships duplicate "original/" and "metal/" weights that vLLM does not use
        path = snapshot_download(repo, revision=revision, ignore_patterns=["original/*", "metal/*"])
        print(f"  -> {path}", flush=True)


if __name__ == "__main__":
    main()
