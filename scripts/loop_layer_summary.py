#!/usr/bin/env python3
"""Heatmaps for the single-loop x single-layer actadd map (step ``layermap`` of scripts/run_followups.sh).

    scripts/loop_layer_summary.py [TAG] [--model Ouro-1.4B-Thinking] [--clean-tag loop_sweep]

Reads generations/TAG (judged), writes analysis/TAG.csv and figures/TAG.png: refusal rate (all CoTs) and the share
of unclosed CoTs for each (loop, layer) where the direction of that loop and layer was added once. The clean
refusal rate comes from ``--clean-tag``'s ``none`` run (same prompts). Loops are numbered 1-4. CPU only.
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from summarize import summarize

ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"actadd_.+?_t\d\.l(?P<layer>\d+)_c(?P<c>-?[\d.]+)_apply(?P<loop>\d)_dirperloop$")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag", nargs="?", default="loop_layer_map")
    ap.add_argument("--model", default="Ouro-1.4B-Thinking")
    ap.add_argument("--clean-tag", default="loop_sweep")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()
    run = ROOT / "data" / "runs" / args.model
    rows = []
    for f in sorted((run / "generations" / args.tag).glob("*.scored.parquet")):
        row = summarize(f, args.max_tokens)
        m = NAME.fullmatch(row["candidate"])
        if m:
            rows.append({**row, "loop": int(m["loop"]) + 1, "layer": int(m["layer"]), "c": float(m["c"])})
    df = pd.DataFrame(rows).sort_values(["loop", "layer"])
    clean_file = run / "generations" / args.clean_tag / "none.scored.parquet"
    clean = summarize(clean_file, args.max_tokens) if clean_file.exists() else None
    with pd.option_context("display.width", 200):
        print(df[["loop", "layer", "c", "all_refusal", "judge_refusal", "all_score", "invalid_cot_frac"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if clean:
        print(f"clean: refusal {clean['all_refusal']:.3f}, unclosed {clean['invalid_cot_frac']:.3f}")
    (run / "analysis").mkdir(exist_ok=True)
    (run / "figures").mkdir(exist_ok=True)
    df.to_csv(run / "analysis" / f"{args.tag}.csv", index=False)

    layers, loops = sorted(df.layer.unique()), sorted(df.loop.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ax, col, title, cmap, vmax in ((axes[0], "all_refusal", "refusal rate (all CoTs)", "viridis_r", 0.7),
                                       (axes[1], "invalid_cot_frac", "unclosed CoTs", "magma_r", 0.7)):
        grid = df.pivot(index="loop", columns="layer", values=col).reindex(index=loops, columns=layers)
        im = ax.imshow(grid.values, vmin=0, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(layers)), layers)
        ax.set_yticks(range(len(loops)), loops)
        ax.set_xlabel("layer where the vector is added")
        ax.set_ylabel("loop")
        ax.set_title(title + (f" (clean {clean[col]:.2f})" if clean else ""))
        for (i, j), v in np.ndenumerate(grid.values):
            ax.text(j, i, "" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > vmax / 2 else "black")
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"{args.model}: actadd c={df.c.iloc[0]:g}, one loop and one layer at a time ({args.tag})")
    fig.tight_layout()
    fig.savefig(run / "figures" / f"{args.tag}.png", dpi=160)
    print(f"wrote analysis/{args.tag}.csv and figures/{args.tag}.png")


if __name__ == "__main__":
    main()
