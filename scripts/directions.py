#!/usr/bin/env python3
"""Difference-in-means refusal directions from cached activations.

Methods (label granularity x token window):
  v4_cot       CoT-level labels, mean over all CoT tokens        (reasoning-manipulation v4 'cot')
  v12_cot150   CoT-level labels, mean over first 150 CoT tokens  (v1/v2)
  cot_last150  CoT-level labels, mean over last 150 CoT tokens   (decision-adjacent control)
  v4_baseline  prompt-level labels, mean over EOI tokens         (v4 'baseline', Arditi-style)
  paired_cot   within-prompt: mean over prompts of (refusal CoT mean - non-refusal CoT mean),
               using only prompts that have both labels (controls prompt content)
  random       Gaussian control, same per-site norm as v4_cot

direction[site] = mean(refusal) - mean(non_refusal), raw (unnormalized), per (loop, layer).
Writes ``directions/<name>.pt`` = {method: {"dirs": [S, D] fp32, "n_ref", "n_non"}, "sites": ...}.
"""

import argparse
from pathlib import Path

import torch

from loop_steer.paths import run_dir

METHODS = {
    "v4_cot": ("label_cot", "cot"),
    "v12_cot150": ("label_cot", "cot_first150"),
    "cot_last150": ("label_cot", "cot_last150"),
    "v4_baseline": ("label_prompt", "eoi"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--activations", required=True, help="Name under activations/")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    rd = run_dir(args.model)
    blob = torch.load(rd / "activations" / f"{args.activations}.pt", weights_only=False)
    meta, acts, windows = blob["meta"], blob["acts"], list(blob["windows"])
    result = {"sites": blob["sites"], "windows": windows}

    for method, (label_col, window) in METHODS.items():
        w = windows.index(window)
        sub = meta
        if label_col == "label_prompt":  # one row per prompt for prompt-level sets
            sub = meta.drop_duplicates("prompt_idx")
        ref = sub.index[sub[label_col] == "refusal"].to_numpy()
        non = sub.index[sub[label_col] == "non_refusal"].to_numpy()
        X = acts[:, w].float()
        dirs = X[ref].mean(0) - X[non].mean(0)
        result[method] = {"dirs": dirs, "n_ref": len(ref), "n_non": len(non), "window": window}
        print(f"{method:12s} n_ref={len(ref):4d} n_non={len(non):4d}")

    # Within-prompt paired contrast on full-CoT means.
    w = windows.index("cot")
    X = acts[:, w].float()
    diffs = []
    for _, g in meta.groupby("prompt_idx"):
        r, n = g.index[g.label_cot == "refusal"], g.index[g.label_cot == "non_refusal"]
        if len(r) and len(n):
            diffs.append(X[r].mean(0) - X[n].mean(0))
    if diffs:
        result["paired_cot"] = {"dirs": torch.stack(diffs).mean(0), "n_ref": len(diffs), "n_non": len(diffs),
                                "window": "cot"}
        print(f"{'paired_cot':12s} n_prompts={len(diffs)}")

    # Control: random Gaussian directions with the same per-site norm as v4_cot.
    gen = torch.Generator().manual_seed(0)
    rand = torch.randn(result["v4_cot"]["dirs"].shape, generator=gen)
    rand = rand / rand.norm(dim=-1, keepdim=True) * result["v4_cot"]["dirs"].norm(dim=-1, keepdim=True)
    result["random"] = {"dirs": rand, "n_ref": 0, "n_non": 0, "window": None}

    out = rd / "directions" / f"{args.name or args.activations}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
