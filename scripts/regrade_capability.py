#!/usr/bin/env python3
"""Regrade stored capability runs with the current grader of ``capability.py`` (``extract_answer`` / ``is_correct``).

    scripts/regrade_capability.py                      # every data/runs/*/capability/mmlupro/*/*.parquet
    scripts/regrade_capability.py --dataset math500 --dry-run

Recomputes ``pred`` and ``correct`` of closed samples from the stored ``response`` (``split_cot`` gives back the answer
after the CoT) and rewrites the file only if something changed. Made for the MMLU-Pro "answer is (X)" fallback: files
written before it count Ouro's answers in MMLU-Pro's own format as unanswered. Idempotent; rerun after ``exp.py pull``,
which brings back the other hosts' files as they were graded there. CPU only.
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from capability import extract_answer, is_correct  # noqa: E402
from loop_steer.cot import split_cot  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="mmlupro")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for f in sorted(glob.glob(str(ROOT / "data" / "runs" / "*" / "capability" / args.dataset / "*" / "*.parquet"))):
        df = pd.read_parquet(f)
        pred = [extract_answer(split_cot(r)[1], t) if c else None
                for r, t, c in zip(df.response, df.truth, df.closed)]
        correct = [is_correct(p, t) for p, t in zip(pred, df.truth)]
        changed = sum(a != b for a, b in zip(correct, df.correct))
        before = df.correct.mean()
        if changed:
            df["pred"], df["correct"] = pred, correct
            if not args.dry_run:
                df.to_parquet(f)
        rel = Path(f).relative_to(ROOT / "data" / "runs")
        print(f"{str(rel):90s} {before:.3f} -> {df.correct.mean():.3f} ({changed} changed)")


if __name__ == "__main__":
    main()
