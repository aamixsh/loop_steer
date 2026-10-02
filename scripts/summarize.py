#!/usr/bin/env python3
"""Summarize scored generation parquets (one row per candidate).

Columns: mean StrongREJECT score, judge refusal rate, substring refusal rate,
fraction of CoTs missing </think>, and upstream's layer-selection metric
``weighted = mean_score * n_valid_rows / n_expected_rows`` (penalizes candidates
that break generation). When ``invalid/<name>.scored.parquet`` exists, also:
``all_score`` (mean over all stage-1 samples, invalid ones judged on the whole
response), ``no_think_frac`` (samples that never opened <think>) and
``hit_max_frac`` (samples that ran to max tokens, e.g. degenerate loops).
"""

import argparse
from pathlib import Path

import pandas as pd


def summarize(path: Path, max_tokens: int = 2048) -> dict:
    df = pd.read_parquet(path)
    inv_path = path.parent / "invalid" / path.name
    inv = pd.read_parquet(inv_path) if inv_path.exists() else None
    n_valid = df.groupby(["prompt_idx", "cot_rep"]).ngroups if len(df) else 0
    if len(df) and "n_cot_total" in df:
        n_cot_total = int(df.n_cot_total.iloc[0])
    elif inv is not None:  # no valid CoT at all
        n_cot_total = n_valid + len(inv)
    else:  # clean generations: n_cot_total not stored; assume 3 CoTs per prompt seen
        n_cot_total = max(df.prompt_idx.nunique() * 3, n_valid)
    out_reps = int(df.out_rep.max()) + 1 if len(df) else 3
    row = {
        "candidate": path.name.removesuffix(".scored.parquet"),
        "n_rows": len(df),
        "mean_score": df.sr_score.mean() if len(df) else float("nan"),
        "median_score": df.sr_score.median() if len(df) else float("nan"),
        "judge_refusal": df.sr_refusal.mean() if len(df) else float("nan"),
        "substring_refusal": df.substring_refusal.mean() if len(df) else float("nan"),
        "invalid_cot_frac": 1 - n_valid / n_cot_total,
        "unparsed": int(df.sr_score.isna().sum()) if len(df) else 0,
        "weighted": (df.sr_score.fillna(0).sum() if len(df) else 0.0) / (n_cot_total * out_reps),
    }
    if inv is not None and "sr_score" in inv:
        # Per stage-1 sample, counting responses that never closed </think> (judged whole).
        per_cot = df.groupby(["prompt_idx", "cot_rep"]).sr_score.mean().sum() if len(df) else 0.0
        row["all_score"] = (per_cot + inv.sr_score.fillna(0).sum()) / n_cot_total
        row["no_think_frac"] = (~inv.response.str.lstrip().str.startswith("<think>")).sum() / n_cot_total
        row["hit_max_frac"] = (inv.n_tokens >= max_tokens).sum() / n_cot_total
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path, help="Scored parquets or directories of them")
    ap.add_argument("--sort", default="weighted")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()
    files = []
    for p in args.paths:
        files += sorted(p.glob("*.scored.parquet")) if p.is_dir() else [p]
    # invalid/ files are folded into their main file's row by summarize()
    files = [f for f in files if f.parent.name != "invalid"]
    table = pd.DataFrame([summarize(f) for f in files]).sort_values(args.sort, ascending=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.csv:
        table.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
