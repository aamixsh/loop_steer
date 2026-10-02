#!/usr/bin/env python3
"""Per-site diagnostics for a difference-in-means direction.

Splits labeled examples by prompt into two halves A/B. For every (loop, layer):
  norm        |mean-diff| on all data
  splithalf   cosine(dir_A, dir_B)                       -- stability
  auc         AUC of projection onto dir_A, evaluated on B -- held-out separability
  dprime      (mu_ref - mu_non) / pooled sd of projections on B
Also plots cosine similarity of the full-data direction across all sites.
Writes ``analysis/<method>_sites.csv`` and ``figures/<method>_*.png`` under the run dir.
"""

import argparse

import numpy as np
import pandas as pd
import torch

from loop_steer.paths import run_dir

LABEL_WINDOW = {
    "v4_cot": ("label_cot", "cot"), "v12_cot150": ("label_cot", "cot_first150"),
    "cot_last150": ("label_cot", "cot_last150"), "v4_baseline": ("label_prompt", "eoi"),
}


def auc(pos, neg):
    """Mann-Whitney AUC: P(score(pos) > score(neg))."""
    s = np.concatenate([pos, neg])
    ranks = s.argsort().argsort() + 1
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--activations", default="train")
    ap.add_argument("--method", default="v4_cot", choices=list(LABEL_WINDOW))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rd = run_dir(args.model)
    blob = torch.load(rd / "activations" / f"{args.activations}.pt", weights_only=False)
    meta, acts, sites = blob["meta"], blob["acts"], blob["sites"]
    label_col, window = LABEL_WINDOW[args.method]
    X = acts[:, list(blob["windows"]).index(window)].float()  # [N, S, D]
    sub = meta.drop_duplicates("prompt_idx") if label_col == "label_prompt" else meta
    sub = sub[sub[label_col].isin(["refusal", "non_refusal"])]
    y = (sub[label_col] == "refusal").to_numpy()

    rng = np.random.default_rng(args.seed)
    prompts = sub.prompt_idx.unique()
    half_a = set(rng.choice(prompts, len(prompts) // 2, replace=False))
    in_a = sub.prompt_idx.isin(half_a).to_numpy()
    idx = sub.index.to_numpy()

    def mean_diff(rows, labels):
        return X[rows[labels]].mean(0) - X[rows[~labels]].mean(0)  # [S, D]

    full = mean_diff(idx, y)
    dir_a, dir_b = mean_diff(idx[in_a], y[in_a]), mean_diff(idx[~in_a], y[~in_a])
    unit_a = dir_a / dir_a.norm(dim=-1, keepdim=True)
    proj_b = torch.einsum("nsd,sd->ns", X[idx[~in_a]], unit_a).numpy()  # [N_B, S]
    yb = y[~in_a]

    rows = []
    for s, (loop, layer) in enumerate(sites):
        pr, pn = proj_b[yb, s], proj_b[~yb, s]
        pooled = np.sqrt((pr.var(ddof=1) + pn.var(ddof=1)) / 2)
        rows.append({
            "loop": loop, "layer": layer, "norm": float(full[s].norm()),
            "splithalf": float(torch.nn.functional.cosine_similarity(dir_a[s], dir_b[s], dim=0)),
            "auc": auc(pr, pn), "dprime": (pr.mean() - pn.mean()) / pooled,
        })
    table = pd.DataFrame(rows)
    (rd / "analysis").mkdir(exist_ok=True)
    table.to_csv(rd / "analysis" / f"{args.method}_sites.csv", index=False)
    print(f"{args.method}: n_ref={y.sum()} n_non={(~y).sum()}  (half A {in_a.sum()}, half B {(~in_a).sum()})")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unit = full / full.norm(dim=-1, keepdim=True)
    cos = (unit @ unit.T).numpy()
    n_loops = max(t for t, _ in sites) + 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im = axes[0].imshow(cos, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_title(f"{args.method}: cosine between sites (loop-major)")
    if n_loops > 1:
        n_layers = len(sites) // n_loops
        for k in range(1, n_loops):
            axes[0].axhline(k * n_layers - 0.5, color="k", lw=0.5)
            axes[0].axvline(k * n_layers - 0.5, color="k", lw=0.5)
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    for t in range(n_loops):
        part = table[table.loop == t]
        axes[1].plot(part.layer, part.auc, marker="o", ms=3, label=f"loop {t}" if n_loops > 1 else "AUC")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("held-out AUC (refusal vs non-refusal)")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    (rd / "figures").mkdir(exist_ok=True)
    out = rd / "figures" / f"{args.method}_sites.png"
    fig.savefig(out, dpi=120)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
