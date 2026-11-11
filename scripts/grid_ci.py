#!/usr/bin/env python3
"""Refusal rate per candidate with a prompt-bootstrap 95% interval, for one generations directory.

    scripts/grid_ci.py data/runs/Ouro-1.4B-Thinking/generations/grid_test [--n-boot 2000] [--csv out.csv]

Same quantity as ``summarize.py``'s ``all_refusal``: per stage-1 sample (CoT) the mean judge refusal over its answers,
samples whose CoT never closed judged whole (``invalid/``, unscored counted as 0), averaged over samples. The interval
resamples prompts (each with its 3 samples). Candidates share prompts, so differences between candidates are tighter
than the overlap of their intervals suggests. CPU only.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for f in sorted(args.dir.glob("*.scored.parquet")):
        parts = [pd.read_parquet(f)]
        inv = args.dir / "invalid" / f.name
        if inv.exists():
            parts.append(pd.read_parquet(inv))
        df = pd.concat(parts)
        df["sr_refusal"] = df.sr_refusal.fillna(0)
        per_sample = df.groupby(["prompt_idx", "cot_rep"]).sr_refusal.mean()
        per_prompt = per_sample.groupby("prompt_idx").mean().to_numpy()
        boots = rng.choice(per_prompt, (args.n_boot, len(per_prompt))).mean(1)
        rows.append({"candidate": f.name.removesuffix(".scored.parquet"), "n_prompts": len(per_prompt),
                     "refusal": per_sample.mean(), "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5)})
    out = pd.DataFrame(rows).sort_values("refusal")
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.csv:
        out.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
