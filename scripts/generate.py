#!/usr/bin/env python3
"""Two-stage CoT/output sampling with vLLM (reasoning-manipulation protocol).

Stage 1 samples ``--cot-reps`` full responses per prompt; responses without
``</think>`` are dropped. Stage 2 resamples ``--out-reps`` answers conditioned
on each prompt + CoT. Output: a parquet with one row per (prompt, cot, answer).

Example:
  uv run python scripts/generate.py --gpu 0 --model Qwen/Qwen3-8B \
      --split train --n-prompts 500 --out train_clean.parquet
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", default="0")
    p.add_argument("--model", required=True, help="HF id or local (orthogonalized) checkpoint")
    p.add_argument("--tokenizer", default=None, help="Defaults to --model")
    p.add_argument("--run-name", default=None, help="Run directory name; defaults to the tokenizer's basename")
    p.add_argument("--split", default="train")
    p.add_argument("--kind", default="harmful")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--n-prompts", type=int, default=None)
    p.add_argument("--cot-reps", type=int, default=3)
    p.add_argument("--out-reps", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=1.0, help="Upstream code leaves vLLM's default (1.0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--out", required=True, help="Parquet filename inside the run's generations/ dir, or absolute path")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from loop_steer.paths import run_dir, setup_job_env
    setup_job_env()

    from vllm import LLM, SamplingParams

    from loop_steer.cot import load_prompts
    from loop_steer.models import load_tokenizer, vllm_kwargs
    from loop_steer.sampling import two_stage

    tok_name = args.tokenizer or args.model
    out = Path(args.out)
    if not out.is_absolute():
        out = run_dir(args.run_name or tok_name) / "generations" / out
    out.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.split, args.kind)[args.offset:]
    if args.n_prompts is not None:
        prompts = prompts[: args.n_prompts]
    tokenizer = load_tokenizer(tok_name)
    llm = LLM(
        model=args.model, tokenizer=tok_name, seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
        **vllm_kwargs(tok_name),
    )
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature,
                              top_p=args.top_p, skip_special_tokens=False)
    df = two_stage(llm, tokenizer, prompts, cot_reps=args.cot_reps, out_reps=args.out_reps,
                   sampling=sampling, prompt_offset=args.offset)
    invalid = df.attrs.pop("invalid")
    if len(invalid):  # kept outside the scored dir's *.parquet glob
        (out.parent / "invalid").mkdir(exist_ok=True)
        invalid.to_parquet(out.parent / "invalid" / out.name)
    df.to_parquet(out)
    print(f"Wrote {len(df)} rows ({df.prompt_idx.nunique()} prompts) to {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
