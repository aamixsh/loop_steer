#!/usr/bin/env python3
"""Summarize scored generation parquets (one row per candidate).

A stage-1 sample is *valid* if its CoT closes with ``</think>`` (its answers are in the main file) and
*invalid* otherwise (``invalid/<name>.parquet``): either it ran into the token cap (``capped``), or it stopped
by itself without thinking, i.e. a direct answer (``direct``; Ouro does this often). Invalid samples are judged
on the whole response, as is.

Columns: mean StrongREJECT score, judge and substring refusal rate, ``invalid_cot_frac``, and upstream's
layer-selection metric ``weighted = mean_score * n_valid_rows / n_expected_rows``. These cover valid samples
only (except ``weighted``). With ``invalid/<name>.scored.parquet`` present, also the all-samples metrics:
``all_score`` / ``all_refusal`` (mean over every stage-1 sample; per sample the mean over its answers, or the
whole-response judgement if invalid), ``direct_frac`` and ``capped_frac`` (shares of samples).

``--cont DIR`` merges a ``intervene.py --extend-from`` run: capped samples of the main run are replaced by
their continuations (``--max-tokens`` must be the cap of the main run).
"""

import argparse
from pathlib import Path

import pandas as pd


def _read(path: Path):
    return pd.read_parquet(path) if path.exists() else None


def _is_capped(inv: pd.DataFrame, max_tokens: int) -> pd.Series:
    by_length = inv.n_tokens >= max_tokens  # older runs did not store hit_limit
    return inv.hit_limit.astype("boolean").fillna(by_length).astype(bool) if "hit_limit" in inv else by_length


def summarize(path: Path, max_tokens: int = 2048, cont_dir: Path | None = None) -> dict:
    df = pd.read_parquet(path)
    inv = _read(path.parent / "invalid" / path.name)
    n_valid = df.groupby(["prompt_idx", "cot_rep"]).ngroups if len(df) else 0
    if len(df) and "n_cot_total" in df:
        n_cot_total = int(df.n_cot_total.iloc[0])
    elif inv is not None:  # no valid CoT at all
        n_cot_total = n_valid + len(inv)
    else:  # clean generations: n_cot_total not stored; assume 3 CoTs per prompt seen
        n_cot_total = max(df.prompt_idx.nunique() * 3, n_valid)
    if cont_dir is not None and (cont_dir / path.name).exists():
        cdf, cinv = pd.read_parquet(cont_dir / path.name), _read(cont_dir / "invalid" / path.name)
        if inv is not None:
            inv = inv[~_is_capped(inv, max_tokens)]  # superseded by the continuations
        df = pd.concat([df, cdf], ignore_index=True)
        if cinv is not None:
            inv = cinv if inv is None else pd.concat([inv, cinv], ignore_index=True)
        n_valid = df.groupby(["prompt_idx", "cot_rep"]).ngroups if len(df) else 0
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
        per = df.groupby(["prompt_idx", "cot_rep"])[["sr_score", "sr_refusal"]].mean().sum() if len(df) else None
        for name, col in (("all_score", "sr_score"), ("all_refusal", "sr_refusal")):
            row[name] = ((0.0 if per is None else per[col]) + inv[col].fillna(0).sum()) / n_cot_total
        capped = _is_capped(inv, max_tokens)
        row["direct_frac"] = (~capped).sum() / n_cot_total
        row["capped_frac"] = capped.sum() / n_cot_total
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="Scored parquets or directories of them")
    ap.add_argument("--sort", default="weighted")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--max-tokens", type=int, default=2048, help="Stage-1 cap of the main run (2048; Nanbeige 4096)")
    ap.add_argument("--cont", type=Path, default=None, help="Directory of continuations (intervene.py --extend-from)")
    args = ap.parse_args()
    files = []
    for p in args.paths:
        files += sorted(p.glob("*.scored.parquet")) if p.is_dir() else [p]
    # invalid/ files are folded into their main file's row by summarize()
    files = [f for f in files if f.parent.name != "invalid"]
    table = pd.DataFrame([summarize(f, args.max_tokens, args.cont) for f in files]).sort_values(args.sort, ascending=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.csv:
        table.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
