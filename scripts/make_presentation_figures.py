#!/usr/bin/env python3
"""Figures for docs/presentation.md (reads saved results under data/runs; CPU only)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "runs"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C = {"clean": "#7f7f7f", "hook": "#1f77b4", "ortho": "#d62728", "ortho_na": "#ff7f0e",
     "actadd": "#2ca02c", "random": "#c7c7c7", "u": "#9467bd"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def arrow(ax, start, vec, color, label=None, lw=2.2, ls="-", alpha=1.0):
    ax.annotate("", xy=(start[0] + vec[0], start[1] + vec[1]), xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, alpha=alpha, mutation_scale=16))
    if label:
        ax.text(start[0] + vec[0] * 1.04, start[1] + vec[1] * 1.04, label, color=color, fontsize=11,
                ha="left", va="bottom")


def setup(ax, title, lim=1.6):
    ax.set_xlim(-0.4, lim); ax.set_ylim(-0.6, lim)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=12)


# ---------------------------------------------------------------- 1. the three interventions
def fig_interventions():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    u = np.array([0.0, 1.0])
    # (a) directional ablation (hooks)
    ax = axes[0]; setup(ax, "Ablation (hooks)\nremove u from the stream at every layer")
    ax.axvline(0, color=C["u"], ls=":", lw=1); ax.text(0.03, 1.5, "refusal direction u", color=C["u"])
    h = np.array([1.1, 1.2])
    arrow(ax, (0, 0), h, C["clean"], "h")
    arrow(ax, (0, 0), h - (h @ u) * u, C["hook"], "h − (h·u)u")
    ax.plot([h[0], h[0]], [0, h[1]], color=C["hook"], ls="--", lw=1)
    # (b) weight orthogonalization
    ax = axes[1]; setup(ax, "Weight orthogonalization\nmatrices can no longer write along u")
    ax.axvline(0, color=C["u"], ls=":", lw=1); ax.text(0.03, 1.5, "u", color=C["u"])
    for w in ([1.2, 0.9], [0.5, 1.3], [1.4, -0.3]):
        w = np.array(w)
        arrow(ax, (0, 0), w, C["clean"], lw=1.4, alpha=0.6)
        arrow(ax, (0, 0), w - (w @ u) * u, C["ortho"], lw=2)
    ax.text(0.55, -0.5, "W ← W − u uᵀW   (embed, o_proj, down_proj)", color=C["ortho"], ha="center", fontsize=10)
    # (c) activation addition
    ax = axes[2]; setup(ax, "Activation addition\nshift the stream at one layer")
    ax.axvline(0, color=C["u"], ls=":", lw=1); ax.text(0.03, 1.5, "r (mean diff)", color=C["u"])
    h = np.array([1.1, 1.2]); r = np.array([0.0, -0.9])
    arrow(ax, (0, 0), h, C["clean"], "h")
    arrow(ax, tuple(h), r, C["u"], lw=1.5, ls="--")
    arrow(ax, (0, 0), h + r, C["actadd"], "h + c·r  (c < 0)")
    fig.tight_layout()
    fig.savefig(OUT / "interventions.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 2. RMSNorm gain re-introduces u
def fig_norm_leak():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    u = np.array([1, 1]) / np.sqrt(2)
    x = np.array([1, -1]) / np.sqrt(2) * 1.2  # orthogonal to u
    for ax, g, title in ((axes[0], np.array([1.0, 1.0]), "Uniform gain g = (1, 1)"),
                         (axes[1], np.array([2.0, 0.5]), "Learned, uneven gain g = (2, 0.5)")):
        ax.set_xlim(-1.6, 2.6); ax.set_ylim(-1.6, 1.8); ax.set_aspect("equal")
        ax.axhline(0, color="#ddd", lw=0.8); ax.axvline(0, color="#ddd", lw=0.8)
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
        t = np.linspace(-1.6, 1.8, 2)
        ax.plot(t, t, color=C["u"], ls=":", lw=1.2); ax.text(1.25, 1.5, "u", color=C["u"])
        y = g * x / np.sqrt((x ** 2).mean())
        arrow(ax, (0, 0), x, C["clean"])
        ax.text(x[0] - 0.15, x[1] - 0.3, "x  (x·u = 0)", color=C["clean"], ha="right")
        arrow(ax, (0, 0), y, C["ortho"])
        ax.text(y[0] + 0.1, y[1] - 0.35 if g[0] == g[1] else y[1] + 0.12,
                f"g ⊙ x / rms(x)\n(·u = {y @ u:+.2f})", color=C["ortho"], ha="left")
        p = (y @ u) * u
        if abs(y @ u) > 1e-6:
            ax.plot([y[0], p[0]], [y[1], p[1]], color=C["ortho"], ls="--", lw=1)
            arrow(ax, (0, 0), p, C["u"], lw=2.5)
        ax.set_title(title)
    fig.suptitle("RMSNorm(x) = g ⊙ x / rms(x): an uneven gain rotates x, so a vector with no u-component gets one",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "norm_leak.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 3. per-loop growth (real data)
def load_cmp(model, site):
    return json.load(open(RUNS / model / "analysis" / f"ortho_compare_v4_baseline_{site}.json"))


def fig_loop_growth():
    ouro = load_cmp("Ouro-1.4B-Thinking", "t3l16")["full"]
    nanb = load_cmp("Nanbeige4.2-3B", "t1l15")["full"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})

    ax = axes[0]
    for data, name, mk in ((ouro, "Ouro", "o"), (nanb, "Nanbeige", "s")):
        clean = np.array([data["clean"]["refusal"][str(t)]["resid_rms"] for t in range(len(data["clean"]["refusal"]))])
        loops = np.arange(len(clean))
        for cond, lab in (("ortho", "weight ortho"), ("ortho_na", "norm-aware ortho"), ("hook", "hook ablation")):
            if cond not in data:
                continue
            v = np.array([data[cond]["refusal"][str(t)]["resid_rms"] for t in loops]) / clean
            ax.plot(loops, v, marker=mk, color=C[cond], ls="-" if name == "Ouro" else "--", lw=2,
                    label=f"{name}: {lab}")
    ax.axhline(1, color=C["clean"], lw=1.5); ax.text(3.05, 1.08, "clean", color=C["clean"])
    ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xlabel("loop")
    ax.set_ylabel("|projection on u| in the residual\nrelative to the clean model")
    ax.set_title("Refusal direction left in the stream (log scale)")
    ax.set_ylim(5e-4, 20)
    ax.legend(fontsize=8.5, loc="upper left", ncol=2)

    ax = axes[1]
    bars = []
    for data, name in ((ouro, "Ouro"), (nanb, "Nanbeige")):
        rows = data["clean"]["refusal"]
        T = len(rows)
        for t in range(T - 1):  # passes that feed the next loop
            r = rows[str(t)]
            bars.append((f"{name}\nafter loop {t}", r["Lout_len"] / r["Lin_len"], name))
    x = np.arange(len(bars))
    ax.bar(x, [b[1] for b in bars], color=["#8c564b" if b[2] == "Ouro" else "#17becf" for b in bars])
    for i, b in enumerate(bars):
        ax.text(i, b[1] * 1.1, f"×{b[1]:.2f}", ha="center")
    ax.axhline(1, color="k", lw=0.8)
    ax.set_yscale("log"); ax.set_ylim(0.008, 5)
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax.set_ylabel("stream length after / before the norm")
    ax.set_title("Inter-loop norm multiplies the carried stream\nup (Ouro) or down (Nanbeige)")
    fig.tight_layout()
    fig.savefig(OUT / "loop_growth.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 3b. compounding schematic (real numbers)
def fig_compounding():
    from matplotlib.patches import Circle, FancyBboxPatch

    ouro = load_cmp("Ouro-1.4B-Thinking", "t3l16")["full"]
    nanb = load_cmp("Nanbeige4.2-3B", "t1l15")["full"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 6.4))
    for ax, data, name, n_layers, note in (
        (axes[0], ouro, "Ouro-1.4B", 24, "each of 48 writes passes a sandwich norm\n→ leaks u (weights can't stop it)"),
        (axes[1], nanb, "Nanbeige4.2-3B", 22, "writes added raw\n→ exactly ⊥ u after the edit"),
    ):
        rows_c, rows_o = data["clean"]["refusal"], data["ortho"]["refusal"]
        T = len(rows_c)
        ax.set_xlim(-0.5, 17.0); ax.set_ylim(-1.9, 1.4); ax.axis("off")
        ax.text(-0.4, 1.25, name, fontsize=13, weight="bold")
        ax.text(3.6, 1.25, note.replace("\n", " "), fontsize=10, style="italic", va="center")
        x = 0.0
        ax.add_patch(FancyBboxPatch((x, -0.35), 1.0, 0.7, boxstyle="round,pad=0.05", fc="#eee", ec="k"))
        ax.text(x + 0.5, 0, "embed", ha="center", va="center", fontsize=10)
        x += 1.4
        for t in range(T):
            ax.annotate("", xy=(x, 0), xytext=(x - 0.4, 0), arrowprops=dict(arrowstyle="-|>", color="k"))
            ax.add_patch(FancyBboxPatch((x, -0.45), 2.0, 0.9, boxstyle="round,pad=0.05", fc="#e8f0fa", ec="k"))
            ax.text(x + 1.0, 0.12, f"loop {t}", ha="center", fontsize=11, weight="bold")
            ax.text(x + 1.0, -0.22, f"{n_layers} layers", ha="center", fontsize=9)
            c, o = rows_c[str(t)]["resid_rms"], rows_o[str(t)]["resid_rms"]
            ax.text(x + 1.0, -0.8, f"clean {c:.2f}", ha="center", color=C["clean"], fontsize=10)
            ax.text(x + 1.0, -1.15, f"ortho {o:.2f}", ha="center", color=C["ortho"], fontsize=10, weight="bold")
            ax.text(x + 1.0, -1.5, f"({o / c:.0%} of clean)", ha="center", color=C["ortho"], fontsize=9)
            x += 2.0
            ax.annotate("", xy=(x + 0.4, 0), xytext=(x, 0), arrowprops=dict(arrowstyle="-|>", color="k"))
            x += 0.85
            last = t == T - 1
            ax.add_patch(Circle((x, 0), 0.42, fc="#fde8d0" if not last else "#eee", ec="k"))
            ax.text(x, 0, "norm", ha="center", va="center", fontsize=9)
            r = rows_c[str(t)]
            if not last:
                ax.text(x, 0.62, f"× {r['Lout_len'] / r['Lin_len']:.2f}\n+ fresh leak", ha="center", fontsize=9,
                        color="#8c564b")
            else:
                ax.text(x, 0.62, "→ lm_head", ha="center", fontsize=9, color="#555")
            x += 0.85
    axes[0].text(-0.4, -1.15, "|u| in stream\n(rms, refusal u)", fontsize=9, color="#555", va="center")
    fig.suptitle("Weight ortho on a looped model: the leak is re-amplified by every inter-loop norm and "
                 "nothing cancels it", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "compounding.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 4. reduced loop count (Ouro)
def fig_loop_count():
    d = load_cmp("Ouro-1.4B-Thinking", "t3l16")
    runs = {1: d["loops1"], 2: d["loops2"], 4: d["full"]}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    conds = [("hook", "hook ablation"), ("ortho", "weight ortho"), ("ortho_na", "norm-aware ortho")]
    for ax, dname in zip(axes, ("random", "refusal")):
        for j, (cond, lab) in enumerate(conds):
            vals = [runs[T][cond][dname][str(T - 1)]["nll"] for T in runs]
            ax.bar(np.arange(3) + (j - 1) * 0.26, vals, 0.26, color=C[cond], label=lab)
        clean = [runs[T]["clean"]["refusal"][str(T - 1)]["nll"] for T in runs]
        for i, c in enumerate(clean):
            ax.plot([i - 0.42, i + 0.42], [c, c], color="k", lw=1.5, label="clean" if i == 0 else None)
        ax.set_xticks(range(3)); ax.set_xticklabels([f"{T} loop{'s' * (T > 1)}" for T in runs])
        ax.set_title(f"{dname} direction")
    axes[0].set_ylabel("NLL of clean CoT tokens\n(lower = less damage)")
    axes[0].legend(fontsize=9)
    fig.suptitle("Ouro run with fewer loops: the plain edit hurts even at 1 loop (sandwich leak);\n"
                 "the norm-aware edit is fine at 1-2 loops and degrades at 4 (inter-loop leak)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "ouro_loop_count.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 5. headline results
def fig_results():
    # (label, judge refusal %, invalid CoT %, kind); values from analysis CSVs / summarize.py
    data = {
        "Qwen3-8B\nv4_baseline, layer 17": [("clean", 54.2, 0.7, "clean"), ("weight ortho", 8.3, 3.3, "ortho"),
                     ("act-add\nc = −2", 4.7, 0.7, "actadd")],
        "Ouro-1.4B\nv4_baseline, loop 3 layer 16": [("clean", 57.0, 10.2, "clean"), ("hook,\nrandom u", 56.7, 19.3, "random"),
                      ("hook\nablation", 6.6, 19.3, "hook"),
                      ("weight\northo", np.nan, 100.0, "ortho")],
        "Nanbeige4.2-3B\nv4_baseline, loop 1 layer 15": [("clean", 80.3, 0.7, "clean"), ("hook,\nrandom u", 80.7, 0.0, "random"),
                           ("ortho,\nrandom u", 76.9, 0.0, "random"),
                           ("hook\nablation", 14.9, 7.3, "hook"),
                           ("weight\northo", 10.5, 13.3, "ortho"),
                           ("act-add\nc = −1", 55.1, 3.3, "actadd")],
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [3, 4, 6]}, sharey=True)
    for ax, (model, rows) in zip(axes, data.items()):
        x = np.arange(len(rows))
        ref = [r[1] for r in rows]
        ax.bar(x, np.nan_to_num(ref), color=[C[r[3]] for r in rows])
        for i, r in enumerate(rows):
            if np.isnan(r[1]):
                ax.text(i, 3, "no valid\noutputs", ha="center", color=C["ortho"], fontsize=9)
            else:
                ax.text(i, r[1] + 1.5, f"{r[1]:.0f}%", ha="center", fontsize=10)
            ax.text(i, -18, f"invalid {r[2]:.0f}%", ha="center", fontsize=8, color="#555")
        ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=9)
        ax.set_title(model); ax.set_ylim(0, 95)
    axes[0].set_ylabel("judge refusal rate (%)")
    fig.suptitle("Refusal on the 50 selection prompts (StrongREJECT judge). "
                 "'invalid' = CoTs that never close </think>", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "results.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- 6. where the direction lives
def fig_sites():
    models = [("Qwen3-8B", "Qwen3-8B"), ("Ouro-1.4B-Thinking", "Ouro-1.4B (4 loops)"),
              ("Nanbeige4.2-3B", "Nanbeige4.2-3B (2 loops)")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, (m, title) in zip(axes, models):
        a = pd.read_csv(RUNS / m / "analysis" / "v4_cot_sites.csv")
        if "loop" not in a:
            a["loop"] = 0
        T = a.loop.max() + 1
        for t in range(T):
            s = a[a.loop == t].sort_values("layer")
            ax.plot(s.layer, s.auc, marker=".", color=cmap(t / max(T - 1, 1)) if T > 1 else "k",
                    label=f"loop {t}" if T > 1 else None)
        ax.set_title(title); ax.set_xlabel("layer")
        if T > 1:
            ax.legend(fontsize=9)
    axes[0].set_ylabel("AUC: refusing vs complying CoTs\n(projection on direction)")
    fig.suptitle("Whole-CoT direction separates refusing from complying CoTs at almost every site", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "sites.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_interventions(); fig_norm_leak(); fig_loop_growth(); fig_compounding(); fig_loop_count()
    fig_results(); fig_sites()
    print("figures ->", OUT)
