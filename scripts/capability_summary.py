#!/usr/bin/env python3
"""Tabulate the capability runs (scripts/capability.py): accuracy per model, dataset and candidate.

    scripts/capability_summary.py [--out data/runs/capability_summary.csv]

Reads data/runs/<model>/capability/<dataset>/<tag>/*.parquet. ``se`` is the binomial standard error over all samples
(it ignores that samples of one problem are correlated, so it is optimistic, especially for AIME 2025: 30 problems);
``closed`` is the share of samples whose CoT ended with ``</think>``. CPU only.
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "runs" / "capability_summary.csv")
    args = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data" / "runs" / "*" / "capability" / "*" / "*" / "*.parquet"))):
        df = pd.read_parquet(f)
        p = Path(f).parts
        acc = df.correct.mean()
        rows.append({"model": p[-5], "dataset": p[-3], "tag": p[-2], "candidate": df.candidate.iloc[0], "accuracy": acc,
                     "se": np.sqrt(acc * (1 - acc) / len(df)), "closed": df.closed.mean(),
                     "mean_tokens": df.n_tokens.mean(), "hit_limit": df.hit_limit.mean(), "n": len(df)})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
