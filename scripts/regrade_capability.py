#!/usr/bin/env python3
"""Re-grade every stored capability run with the current grader (scripts/capability.py ``is_correct``); CPU.

    .venv/bin/python scripts/regrade_capability.py [--dry-run]

Grader v2 (2026-11-21) accepts answers v1 marked wrong only for their form: a unit after the number
("5\\text{ cm}"), "\\$78", \\sqrt2 vs \\sqrt{2}, x^{5} vs x^5, solutions listed in another order, "1 \\pm \\sqrt{19}"
written as two roots, matrices with \\\\[2pt] row spacing or a "\\mathbf v=" prefix, "(C)" vs "C", "x \\in [-2,7]",
"12^{th}", and base-n digits without their subscript. It rejects nothing v1 accepted. Each changed file keeps v1's
verdict in ``correct_v1`` (written once, so a second run keeps the original); ``pred`` is unchanged.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capability import is_correct  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    files = [f for f in sorted((ROOT / "data" / "runs").glob("*/capability/*/**/*.parquet")) if "cutoff" not in f.parts]
    changed = 0
    for f in files:
        d = pd.read_parquet(f)
        if not {"truth", "pred", "correct"} <= set(d.columns):
            continue
        new = pd.Series([is_correct(p, t) for p, t in zip(d.pred, d.truth)], index=d.index)
        old = d["correct_v1"] if "correct_v1" in d else d.correct
        flips = int((new != d.correct).sum())
        if not flips:
            continue
        changed += 1
        print(f"{f.relative_to(ROOT)}: {flips} re-graded, accuracy {old.mean():.3f} -> {new.mean():.3f}", flush=True)
        if not a.dry_run:
            if "correct_v1" not in d:
                d["correct_v1"] = d.correct
            d["correct"] = new
            d.to_parquet(f)
    print(f"{changed} of {len(files)} files {'would change' if a.dry_run else 'changed'}")


if __name__ == "__main__":
    main()
