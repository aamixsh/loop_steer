#!/usr/bin/env python3
"""Accuracy under random kicks per site and size, with paired problem-bootstrap intervals, plus the teacher-forced KL.

    scripts/kick_summary.py --model Ouro-1.4B-Thinking --tag kickp [--csv out.csv]

Reads ``capability/math500/<tag>/*.parquet`` (``capability.py`` with ``kick:randp<seed>:<site>:eps=<e>[:apply=t,t]``
candidates and ``none``) and, if present, ``analysis/kick_kl/<tag>.parquet`` (``kick_kl.py``). Per kicked candidate: accuracy, its
95% interval and the 95% interval of the drop from the unkicked run (problems resampled jointly, so the drop is
paired), the share of closed CoTs, and the mean KL / top-1 agreement over the problems. CPU only.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="run directory name under data/runs")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--kl-tag", action="append", default=None,
                    help="analysis/kick_kl/<name>.parquet to read (repeatable; default: --tag)")
    args = ap.parse_args()
    rd = ROOT / "data" / "runs" / args.model
    d = rd / "capability" / args.dataset / args.tag
    acc = {f.stem: pd.read_parquet(f).groupby("problem_id").agg(correct=("correct", "mean"), closed=("closed", "mean"))
           for f in sorted(d.glob("*.parquet"))}
    if "none" not in acc:
        raise SystemExit(f"no none.parquet in {d}")
    base = acc["none"]
    kl = None
    klps = [rd / "analysis" / "kick_kl" / f"{t}.parquet" for t in (args.kl_tag or [args.tag])]
    klps = [p for p in klps if p.exists()]
    if klps:
        kl = pd.concat([pd.read_parquet(p) for p in klps], ignore_index=True)
        kl["apply"] = kl["apply"].fillna("") if "apply" in kl else ""
        kl = kl.groupby(["loop", "layer", "apply", "eps"])[["kl", "agree"]].mean()
    rng = np.random.default_rng(0)
    rows = []
    for name, a in acc.items():
        a = a.reindex(base.index)
        x, y = a.correct.to_numpy(), base.correct.to_numpy()
        idx = rng.integers(0, len(x), (args.n_boot, len(x)))
        bx, bd = x[idx].mean(1), (y[idx] - x[idx]).mean(1)
        row = {"candidate": name, "acc": x.mean(), "lo": np.percentile(bx, 2.5), "hi": np.percentile(bx, 97.5),
               "drop": y.mean() - x.mean(), "drop_lo": np.percentile(bd, 2.5), "drop_hi": np.percentile(bd, 97.5),
               "closed": a.closed.mean(), "n": len(x)}
        m = re.fullmatch(r"kick_randp?\d+_(?:t(\d+)\.)?l(\d+)_eps([\d.]+)(?:_apply([\d-]+))?(?:_layers([\d-]+))?(_indep1)?", name)
        if m:  # apply: the loops of an apply= kick ("0,1"), the layers of a layers= kick ("l4,l13"), else empty
            apply = (m.group(4) or "").replace("-", ",")
            if m.group(5):
                apply = ",".join(f"l{x}" for x in m.group(5).split("-"))
            if m.group(6):  # indep=1: a different direction in each kicked loop / layer (kick_kl.py --indep)
                apply += "/indep"
            loop = int(apply.split(",")[0]) if apply and m.group(4) else int(m.group(1) or 0)
            row.update(loop=loop, layer=int(m.group(2)), apply=apply, eps=float(m.group(3)))
            key = (loop, row["layer"], apply, row["eps"])
            if kl is not None and key in kl.index:
                row.update(kl.loc[key].to_dict())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["eps", "loop", "layer", "apply"], na_position="first")
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if args.csv:
        out.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
