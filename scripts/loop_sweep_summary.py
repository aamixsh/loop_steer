#!/usr/bin/env python3
"""Summarize loop-subset sweeps (scripts/run_loop_sweep.sh): refusal per set of loops that were intervened in.

    scripts/loop_sweep_summary.py TAG [TAG ...] [--model Ouro-1.4B-Thinking] [--n-loops 4] [--max-tokens 2048]

Reads data/runs/<model>/generations/TAG (judged: *.scored.parquet and invalid/*.scored.parquet), writes
analysis/NAME.csv and figures/NAME.png under the model's run directory and prints the table. With several tags
(typically ``loop_sweep loop_sweep_perloop``) the variants are drawn side by side: ``shared`` (one direction used in
every selected loop) and ``perloop`` (each loop's own direction, candidates ending in ``_dirperloop``); the clean and
random-direction reference lines come from the first tag that has them. NAME is the tag, or the last tag + ``_compare``.
Loops are numbered 1-N here (0-based in the code). ``all_refusal`` averages over every sampled CoT, including unclosed
ones; ``invalid_cot_frac`` is the share of CoTs that never reached ``</think>`` (an intervention that breaks
generation also lowers refusal, so read the two together). CPU only.
"""

import argparse
import re
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from summarize import summarize

ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"(?P<kind>ablate|actadd)_(?P<method>.+?)_t(?P<site_loop>\d)\.l(?P<layer>\d+)"
                  r"(?:_c(?P<c>-?[\d.]+))?(?:_apply(?P<apply>\d(?:-\d)*))?(?P<perloop>_dirperloop)?$")
COLORS = {"ablate": ("#9ecae1", "#08519c"), "actadd": ("#a1d99b", "#006d2c")}  # (shared, perloop)


def parse(name: str, n_loops: int) -> dict:
    """Candidate name -> kind, method, coefficient, variant and the 1-based set of loops intervened in."""
    if name == "none":
        return {"kind": "none", "method": "none", "variant": "-", "loops": ()}
    m = NAME.fullmatch(name)
    if m is None:
        return {"kind": "other", "method": name, "variant": "-", "loops": ()}
    loops = tuple(int(x) + 1 for x in m["apply"].split("-")) if m["apply"] else tuple(range(1, n_loops + 1))
    return {"kind": m["kind"], "method": m["method"], "loops": loops, "c": float(m["c"]) if m["c"] else None,
            "variant": "perloop" if m["perloop"] else "shared"}


def build_table(gen_dirs: list[Path], max_tokens: int, n_loops: int) -> pd.DataFrame:
    rows = []
    for gen_dir in gen_dirs:
        for f in sorted(gen_dir.glob("*.scored.parquet")):
            row = summarize(f, max_tokens)
            row.update(parse(row["candidate"], n_loops), tag=gen_dir.name)
            rows.append(row)
    df = pd.DataFrame(rows)
    # runs with no unclosed CoTs have no invalid/ file: every sample is valid, so the all-samples metrics equal these
    df["all_refusal"] = df.all_refusal.fillna(df.judge_refusal)
    df["all_score"] = df.all_score.fillna(df.mean_score)
    df["invalid_cot_frac"] = df.invalid_cot_frac.fillna(0.0)
    df["loops_label"] = df.loops.map(lambda t: ",".join(map(str, t)) if t else "-")
    df["n_loops"] = df.loops.map(len)
    return df.sort_values(["kind", "variant", "n_loops", "loops"]).reset_index(drop=True)


def plot(df: pd.DataFrame, path: Path, title: str, n_loops: int) -> None:
    kinds = [k for k in ("ablate", "actadd") if (df.kind == k).any()]
    fig, axes = plt.subplots(3, len(kinds), figsize=(max(6.2, 0.55 * 2 ** n_loops) * len(kinds), 7.2), sharex="col",
                             squeeze=False, gridspec_kw={"height_ratios": [3, 1.6, 0.35 * n_loops + 0.4], "hspace": 0.08})
    clean = df[df.kind == "none"]
    subsets = sorted({t for t in df[df.kind.isin(kinds) & (df.method != "random")].loops},
                     key=lambda t: (len(t), t))
    for j, kind in enumerate(kinds):
        sel = df[(df.kind == kind) & (df.method != "random")]
        rnd = df[(df.kind == kind) & (df.method == "random") & (df.variant == "shared")]
        top, mid, dots = axes[0, j], axes[1, j], axes[2, j]
        variants = [v for v in ("shared", "perloop") if (sel.variant == v).any()]
        width = 0.8 / len(variants)
        for k, variant in enumerate(variants):
            sub = sel[sel.variant == variant].set_index("loops").reindex(subsets)
            x = np.arange(len(subsets)) + (k - (len(variants) - 1) / 2) * width
            color = COLORS[kind][k if len(variants) > 1 else (1 if variant == "perloop" else 0)]
            top.bar(x, sub.all_refusal, width=width * 0.95, color=color, label=variant)
            top.plot(x, sub.invalid_cot_frac, "o", color="#d62728", ms=3.5, label="unclosed CoTs" if k == 0 else None)
            mid.bar(x, sub.all_score, width=width * 0.95, color=color)
        if len(clean):
            top.axhline(clean.all_refusal.iloc[0], color="#7f7f7f", ls="--", lw=1.2, label="clean")
        if len(rnd):
            top.axhline(rnd.all_refusal.iloc[0], color="#aaaaaa", ls=":", lw=1.2, label="random direction, all loops")
        top.set_ylim(0, 1)
        coeffs = sel.c.dropna().unique()
        top.set_title(kind + (f" (c={coeffs[0]:g})" if kind == "actadd" and len(coeffs) else ""))
        top.set_ylabel("refusal rate (all CoTs)")
        if j == 0:
            top.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
        mid.set_ylim(0, 1)
        mid.set_ylabel("StrongREJECT score")
        for i, loops in enumerate(subsets):  # which loops are intervened in (matrix of dots)
            for k in range(1, n_loops + 1):
                dots.plot(i, k, "o", ms=5, color="black" if k in loops else "#dddddd")
        dots.set_yticks(range(1, n_loops + 1))
        dots.set_ylabel("loop")
        dots.invert_yaxis()
        dots.set_xticks([])
        for ax in (top, mid, dots):
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="*", default=["loop_sweep"])
    ap.add_argument("--model", default="Ouro-1.4B-Thinking", help="Run directory name under data/runs")
    ap.add_argument("--n-loops", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=2048, help="Stage-1 cap of the runs (Nanbeige: 4096)")
    ap.add_argument("--name", default=None, help="Output name (default: the tag, or the last tag + _compare)")
    args = ap.parse_args()
    run = ROOT / "data" / "runs" / args.model
    df = build_table([run / "generations" / t for t in args.tags], args.max_tokens, args.n_loops)
    # references (clean, random direction) come from the first tag that has them
    for key in (("none", "none"), ("ablate", "random"), ("actadd", "random")):
        hit = df[(df.kind == key[0]) & (df.method == key[1])]
        df = pd.concat([df[~df.index.isin(hit.index)], hit.drop_duplicates(["kind", "method", "variant"])])
    df = df.sort_values(["kind", "variant", "n_loops", "loops"]).reset_index(drop=True)
    name = args.name or (args.tags[0] if len(args.tags) == 1 else args.tags[-1] + "_compare")
    cols = ["kind", "method", "variant", "loops_label", "all_refusal", "judge_refusal", "all_score",
            "invalid_cot_frac", "direct_frac", "capped_frac", "n_rows"]
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(df[[c for c in cols if c in df]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    (run / "analysis").mkdir(exist_ok=True)
    (run / "figures").mkdir(exist_ok=True)
    df.drop(columns=["loops"]).to_csv(run / "analysis" / f"{name}.csv", index=False)
    plot(df, run / "figures" / f"{name}.png", f"{args.model}: intervention in subsets of loops ({name})", args.n_loops)
    print(f"wrote analysis/{name}.csv and figures/{name}.png")


if __name__ == "__main__":
    main()
