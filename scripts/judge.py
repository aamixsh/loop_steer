#!/usr/bin/env python3
"""Add StrongREJECT scores (and a substring-refusal flag) to generation parquets.

Default: bounded batch job that loads the judge (gpt-oss-20b) in-process with
vLLM on ``--gpu``, scores every input, and exits. ``--url`` instead targets a
running OpenAI-compatible server (``scripts/serve_judge.py``).
Directories are expanded to their unscored ``*.parquet`` files. Files from
``invalid/`` (stage-1 responses that never closed </think>) are judged on the
whole response. Writes
``<input>.scored.parquet`` next to each input.
"""

import argparse
import os
from pathlib import Path


def expand(paths, rescore):
    files = []
    for p in paths:
        if p.is_dir():
            files += [f for f in sorted(p.glob("*.parquet")) if not f.name.endswith(".scored.parquet")]
        else:
            files.append(p)
    return [f for f in files if rescore or not f.with_suffix(".scored.parquet").exists()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--gpu", default="1")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    p.add_argument("--url", default=None, help="Use a running judge server instead of in-process vLLM")
    p.add_argument("--concurrency", type=int, default=192)
    p.add_argument("--rescore", action="store_true", help="Re-judge files that already have scores")
    p.add_argument("--suffix", default=".scored.parquet")
    args = p.parse_args()

    files = expand(args.inputs, args.rescore)
    if not files:
        print("Nothing to score.")
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from loop_steer.paths import setup_job_env
    setup_job_env()
    import pandas as pd

    from loop_steer.judge import judge, judge_offline, load_offline_judge, substring_refusal

    frames = [pd.read_parquet(f) for f in files]
    for df in frames:
        if "output" not in df and "response" in df:  # invalid/: judge the whole stage-1 response
            df["output"] = df.response.str.removeprefix("<think>").str.strip()
        elif "output" not in df:  # no valid CoTs at all -> empty frame
            df["prompt"], df["output"] = pd.Series(dtype=str), pd.Series(dtype=str)
    pairs = [(pr, out) for df in frames for pr, out in zip(df.prompt, df.output)]
    if args.url:
        results = judge(pairs, url=args.url, concurrency=args.concurrency)
    else:
        results = judge_offline(pairs, load_offline_judge(args.gpu_memory_utilization))
    results = pd.DataFrame(results)

    start = 0
    for path, df in zip(files, frames):
        part = results.iloc[start: start + len(df)]
        start += len(df)
        for col in part.columns:
            df[f"sr_{col}"] = part[col].values
        df["substring_refusal"] = df.output.map(substring_refusal)
        out = path.with_name(path.name.removesuffix(".parquet") + args.suffix)
        df.to_parquet(out)
        print(f"{path.name}: mean score {df.sr_score.mean():.3f}, sr-refusal {df.sr_refusal.mean():.3f}, "
              f"substring-refusal {df.substring_refusal.mean():.3f}, unparsed {int(df.sr_score.isna().sum())} -> {out}")


if __name__ == "__main__":
    main()
