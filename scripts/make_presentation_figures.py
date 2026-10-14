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
    ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels(range(1, 5)); ax.set_xlabel("loop")
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
            bars.append((f"{name}\nafter loop {t + 1}", r["Lout_len"] / r["Lin_len"], name))
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
            ax.text(x + 1.0, 0.12, f"loop {t + 1}", ha="center", fontsize=11, weight="bold")
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
def _test_row(model, run, name, cap, ext=None):
    """Summary row of a test-set candidate, with its continuations (``ext``) merged in."""
    from summarize import summarize

    g = RUNS / model / "generations"
    return summarize(g / run / f"{name}.scored.parquet", max_tokens=cap, cont_dir=None if ext is None else g / ext)


def fig_results():
    """Same five columns in the same order for every model: the best test-set candidate of each intervention."""
    q, o, n, g = "Qwen3-8B", "Ouro-1.4B-Thinking", "Nanbeige4.2-3B", "gpt-oss-20b"
    cols = [("clean model", "clean"), ("random direction\n(hook ablation)", "random"), ("hook ablation", "hook"),
            ("weight\northogonalization", "ortho"), ("activation\naddition", "actadd")]
    # per model: (summary row or None, note under the bar) for each column
    data = {
        "Qwen3-8B  (36 layers)": [
            (_test_row(q, "clean_test_8k", "none", 8192), ""),
            (_test_row(q, "test_hooks", "ablate_random_l17", 2048, "test_hooks_ext"), "same norm as u"),
            (_test_row(q, "test_hooks", "ablate_v4_baseline_l21", 2048, "test_hooks_ext"), "u from L21"),
            (_test_row(q, "test_ortho", "ortho_v4_baseline_l21", 2048, "test_ortho_ext"), "u from L21"),
            (_test_row(q, "test_hooks", "actadd_v4_baseline_l17_c-2", 2048, "test_hooks_ext"), "added at L17, c=−2")],
        "Ouro-1.4B  (24 layers × 4 loops)": [
            (_test_row(o, "clean_test_8k", "none", 8192), ""),
            (_test_row(o, "test_hooks", "ablate_random_t3.l16", 2048, "test_hooks_ext"), "same norm as u"),
            (_test_row(o, "test_hooks", "ablate_v4_baseline_t3.l16", 2048, "test_hooks_ext"), "u from loop 4, L16"),
            (None, "no valid outputs\n(50 prompts)"),
            (_test_row(o, "test_loops", "actadd_v4_baseline_t3.l16_c-1", 2048, "test_loops_ext"),
             "added at L16 of\nevery loop, c=−1")],
        "Nanbeige4.2-3B  (22 layers × 2 loops)": [
            (_test_row(n, "clean_test", "none", 4096, "clean_test_ext"), ""),
            (_test_row(n, "test_int", "ablate_random_t0.l14", 4096, "test_int_ext"), "same norm as u"),
            (_test_row(n, "test_int", "ablate_v4_baseline_t1.l15", 4096, "test_int_ext"), "u from loop 2, L15"),
            (_test_row(n, "test_int", "ortho_v4_baseline_t1.l15", 4096, "test_int_ext"), "u from loop 2, L15"),
            (_test_row(n, "test_loops", "actadd_v4_baseline_t1.l15_c-2", 4096, "test_loops_ext"),
             "added at L15 of\nboth loops, c=−2")],
        "gpt-oss-20b  (24 MoE layers)": [
            (_test_row(g, "test", "none", 4096, "test_ext"), ""),
            (_test_row(g, "test", "ablate_random_l15", 4096, "test_ext"), "same norm as u"),
            (_test_row(g, "test", "ablate_v4_baseline_l15", 4096, "test_ext"), "u from L15"),
            (_test_row(g, "test", "ortho_v4_baseline_l15", 4096, "test_ext"),
             "u from L15, biases\nedited (kept: {bias0:.0f}%)"),
            (_test_row(g, "test", "actadd_v4_baseline_l13_c-2", 4096, "test_ext"), "added at L13, c=−2")],
    }
    bias0 = 100 * _test_row(g, "test", "ortho_v4_baseline_l15_bias0", 4096, "test_ext")["all_refusal"]
    fig, axes = plt.subplots(1, 4, figsize=(23, 5.2), sharey=True)
    x = np.arange(len(cols))
    for ax, (model, rows) in zip(axes, data.items()):
        for i, ((_, kind), (r, note)) in enumerate(zip(cols, rows)):
            if r is None:
                ax.text(i, 4, note, ha="center", color=C["ortho"], fontsize=9)
                continue
            v = 100 * r["all_refusal"]
            ax.bar(i, v, color=C[kind], width=0.72)
            ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=11, weight="bold")
            if note:
                ax.text(i, v + 7.5, note.format(bias0=bias0), ha="center", fontsize=7.8, color="#444")
        ax.set_xticks(x); ax.set_xticklabels([c for c, _ in cols], fontsize=9)
        ax.set_title(model, fontsize=11); ax.set_ylim(0, 108); ax.set_yticks(range(0, 101, 20))
    axes[0].set_ylabel("refusal rate (%)")
    fig.suptitle("Refusal on 487 held-out harmful prompts. Per prompt 3 sampled CoTs, per CoT 3 sampled answers; a CoT counts "
                 "as the share of its answers the judge marks as refusals\n(a CoT that never closes is judged on its raw "
                 "text); the bar is the mean over all 1,461 CoTs. u = the refusal direction, measured at the layer named",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "results.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- 6. where the direction lives
def fig_sites():
    """Held-out AUC of the refusal direction at every site, for both windows, against a random direction."""
    models = [("Qwen3-8B", "Qwen3-8B (36 layers)"), ("Ouro-1.4B-Thinking", "Ouro-1.4B (24 layers × 4 loops)"),
              ("Nanbeige4.2-3B", "Nanbeige4.2-3B (22 layers × 2 loops)"), ("gpt-oss-20b", "gpt-oss-20b (24 MoE layers)")]
    windows = [("v4_cot", "CoT direction: activations averaged over all CoT tokens, CoT labels", "CoTs"),
               ("v4_baseline", "Prompt-end direction: activations averaged over the template tokens before thinking, "
                               "prompt labels", "prompts")]
    fig, axes = plt.subplots(2, 4, figsize=(20, 7.4), sharey=True, gridspec_kw={"width_ratios": [36, 96, 44, 30]})
    for i, (meth, wtitle, unit) in enumerate(windows):
        for j, (m, title) in enumerate(models):
            ax = axes[i, j]
            a = pd.read_csv(RUNS / m / "analysis" / f"{meth}_sites.csv")
            L = a.layer.max() + 1
            a["x"] = a.loop * L + a.layer
            a = a.sort_values("x")
            T = a.loop.max() + 1
            for t in range(T):
                if t % 2:
                    ax.axvspan(t * L - 0.5, (t + 1) * L - 0.5, color="#f2f2f2", zorder=0)
                if T > 1:
                    ax.text(t * L + L / 2, 0.42, f"loop {t + 1}", ha="center", fontsize=9, color="#666")
            ax.plot(a.x, a.auc, color=C["hook"], lw=1.8, label="refusal direction")
            ax.plot(a.x, a.auc_random, color="#999", lw=1.0, label="random direction (chance reference)")
            ax.axhline(0.5, color="#ccc", lw=0.8, ls=":")
            ax.set_ylim(0.38, 1.02); ax.set_xlim(-0.5, T * L - 0.5)
            ax.set_xlabel("layer (loop by loop)" if T > 1 else "layer")
            ax.set_title(f"{title}\nn = {a.n_ref.iloc[0]} refusing / {a.n_non.iloc[0]} complying {unit}", fontsize=9.5)
        axes[i, 0].set_ylabel("held-out AUC")

    axes[0, 0].legend(fontsize=8.5, loc="lower right")
    fig.suptitle("Is refusal linearly readable at each site? Direction from half of the training prompts, AUC of the "
                 "projection on the other half (1 = perfect separation, 0.5 = chance)", fontsize=11)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.83, bottom=0.08, hspace=0.75, wspace=0.06)
    for i, (_, wtitle, _) in enumerate(windows):
        pos = axes[i, 0].get_position()
        fig.text(pos.x0, pos.y1 + 0.085, f"{'AB'[i]}. {wtitle}", fontsize=11, weight="bold")
    fig.savefig(OUT / "sites.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 7. the four models (schematic)
def fig_models():
    from matplotlib.patches import Circle, FancyBboxPatch

    rows = [("Qwen3-8B", 36, 1, False, "standard pre-norm: writes added raw"),
            ("Ouro-1.4B-Thinking", 24, 4, True, "sandwich norm on every write; shared norm after every loop"),
            ("Nanbeige4.2-3B", 22, 2, False, "writes added raw; shared norm after every loop"),
            ("gpt-oss-20b", 24, 1, False, "standard pre-norm, MoE: each MLP routes a token to 4 of 32 experts; "
                                          "biases on o_proj and experts")]
    fig, ax = plt.subplots(figsize=(15, 6.0))
    ax.set_xlim(-0.2, 17.6); ax.set_ylim(-0.2, len(rows) * 1.5 + 0.1); ax.axis("off")
    loop_colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, 4))
    for i, (name, L, T, sandwich, note) in enumerate(rows):
        y = (len(rows) - 1 - i) * 1.5 + 0.55
        ax.text(0, y + 0.55, name, fontsize=12, weight="bold")
        ax.text(4.2, y + 0.55, f"{L} layers × {T} loop{'s' * (T > 1)}  ·  {note}", fontsize=10, style="italic")
        x = 0.0
        ax.add_patch(FancyBboxPatch((x, y - 0.22), 0.9, 0.44, boxstyle="round,pad=0.03", fc="#eee", ec="k"))
        ax.text(x + 0.45, y, "embed", ha="center", va="center", fontsize=9)
        x += 1.15
        width = 2.6 if T > 1 else 9.0 * L / 36  # one-pass models: length in proportion to depth
        for t in range(T):
            ax.annotate("", xy=(x, y), xytext=(x - 0.25, y), arrowprops=dict(arrowstyle="-|>", color="k"))
            ax.add_patch(FancyBboxPatch((x, y - 0.25), width, 0.5, boxstyle="round,pad=0.03",
                                        fc=loop_colors[t] if T > 1 else "#d9d9d9", ec="k", alpha=0.85))
            lab = f"loop {t + 1}: same {L} layers" if T > 1 else f"{L} layers"
            ax.text(x + width / 2, y + 0.04, lab, ha="center", va="center", fontsize=9,
                    color="white" if T > 1 and t < 2 else "k")
            if sandwich:
                ax.text(x + width / 2, y - 0.4, "h += Norm₂(Attn/MLP)", ha="center", fontsize=8, color="#8c564b")
            if name.startswith("gpt-oss"):  # the 32 experts of one MLP, 4 of them routed to
                for e in range(32):
                    ax.add_patch(plt.Rectangle((x + 0.35 + e * 0.165, y - 0.43), 0.13, 0.13, lw=0.5, ec="#666",
                                               fc="#2ca02c" if e in (3, 11, 12, 26) else "white"))
                ax.text(x + 0.35 + 32 * 0.165 + 0.1, y - 0.37, "experts per MLP (4 of 32 used per token)",
                        fontsize=8, va="center", color="#2ca02c")
            x += width + 0.25
            ax.annotate("", xy=(x + 0.05, y), xytext=(x - 0.25, y), arrowprops=dict(arrowstyle="-|>", color="k"))
            ax.add_patch(Circle((x + 0.35, y), 0.3, fc="#fde8d0" if t < T - 1 else "#eee", ec="k"))
            ax.text(x + 0.35, y, "norm", ha="center", va="center", fontsize=8)
            x += 0.95
        ax.annotate("", xy=(x + 0.05, y), xytext=(x - 0.25, y), arrowprops=dict(arrowstyle="-|>", color="k"))
        ax.add_patch(FancyBboxPatch((x + 0.05, y - 0.22), 1.0, 0.44, boxstyle="round,pad=0.03", fc="#eee", ec="k"))
        ax.text(x + 0.55, y, "lm_head", ha="center", va="center", fontsize=9)
    fig.suptitle("Looped models run the same layer stack several times; gpt-oss runs once but its MLPs are mixtures of "
                 "experts.\nA site is (loop, layer): 36 (Qwen), 96 (Ouro), 44 (Nanbeige), 24 (gpt-oss). "
                 "Orange = shared norm between loops", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "models.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 8. pipeline (schematic)
def fig_pipeline():
    from matplotlib.patches import FancyBboxPatch

    steps = [("1. Sample", "3 CoTs per harmful prompt,\n3 answers per CoT\n(T = 0.6, CoT ≤ 8192 tok)", "train: 500 prompts"),
             ("2. Judge", "StrongREJECT rubric,\nlocal gpt-oss-20b\n→ refusal, score ∈ [0,1]", ""),
             ("3. Label", "CoT refusing if all 3\nanswers score < 0.2,\ncomplying if all > 0.8", ""),
             ("4. Direction", "mean(refusing) −\nmean(complying) at every\n(loop, layer) and window", ""),
             ("5. Select", "intervention × direction\n× site; pick by refusal\nand unclosed CoTs", "selection: 50 held-out\ntrain prompts"),
             ("6. Test", "refusal on held-out prompts;\ncapability; over-refusal", "test: 487 harmful, 250 harmless;\nMATH-500, AIME 2025")]
    fig, ax = plt.subplots(figsize=(15, 2.9))
    ax.set_xlim(0, 15); ax.set_ylim(-1.2, 1.4); ax.axis("off")
    w, gap = 2.15, 0.36
    for i, (title, body, data) in enumerate(steps):
        x = 0.1 + i * (w + gap)
        fc = "#e8f0fa" if i < 4 else "#e5f5e0"
        ax.add_patch(FancyBboxPatch((x, -0.1), w, 1.35, boxstyle="round,pad=0.05", fc=fc, ec="k"))
        ax.text(x + w / 2, 1.0, title, ha="center", fontsize=12, weight="bold")
        ax.text(x + w / 2, 0.45, body, ha="center", va="center", fontsize=9.2)
        if data:
            ax.text(x + w / 2, -0.5, data, ha="center", va="center", fontsize=9, color="#555", style="italic")
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.02, 0.5), xytext=(x + w + 0.07, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))
    ax.text(0.1, -1.05, "Prompt sets of reasoning-manipulation (train 1459 / test 487 harmful). Blue: building the "
            "direction; green: evaluating interventions.", fontsize=9, color="#555")
    fig.tight_layout()
    fig.savefig(OUT / "pipeline.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- 9. loop profile on the test set
def _sel(model, kind, loops, c=None):
    d = pd.read_csv(ROOT / "results" / model / "loop_sweep_perloop_compare.csv")
    d = d[(d.kind == kind) & (d.method == "v4_baseline") & (d.variant == "shared") & (d.loops_label == loops)]
    return float(d.all_refusal.iloc[0]) if len(d) else np.nan


def fig_loops_test():
    o, n = "Ouro-1.4B-Thinking", "Nanbeige4.2-3B"
    ot = lambda name: _test_row(o, "test_loops", name, 2048, "test_loops_ext")
    nt = lambda name: _test_row(n, "test_loops", name, 4096, "test_loops_ext")
    panels = [
        (o, "Ouro-1.4B (4 loops), ablation at loop 4 · layer 16 direction", 4,
         _test_row(o, "clean_test_8k", "none", 8192), [
            ((1,), ot("ablate_v4_baseline_t3.l16_apply0"), "ablate"), ((2,), ot("ablate_v4_baseline_t3.l16_apply1"), "ablate"),
            ((3,), ot("ablate_v4_baseline_t3.l16_apply2"), "ablate"), ((4,), ot("ablate_v4_baseline_t3.l16_apply3"), "ablate"),
            ((1, 2), ot("ablate_v4_baseline_t3.l16_apply0-1"), "ablate"), ((3, 4), ot("ablate_v4_baseline_t3.l16_apply2-3"), "ablate"),
            ((1, 2, 3, 4), _test_row(o, "test_hooks", "ablate_v4_baseline_t3.l16", 2048, "test_hooks_ext"), "ablate"),
            ((1, 2, 3, 4), ot("actadd_v4_baseline_t3.l16_c-1"), "actadd")]),
        (n, "Nanbeige4.2-3B (2 loops), loop 2 · layer 15 direction", 2,
         _test_row(n, "clean_test", "none", 4096, "clean_test_ext"), [
            ((1,), nt("ablate_v4_baseline_t1.l15_apply0"), "ablate"), ((2,), nt("ablate_v4_baseline_t1.l15_apply1"), "ablate"),
            ((1, 2), _test_row(n, "test_int", "ablate_v4_baseline_t1.l15", 4096, "test_int_ext"), "ablate"),
            ((1, 2), nt("actadd_v4_baseline_t1.l15_c-2"), "actadd")])]
    fig, axes = plt.subplots(2, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [8, 4.4], "height_ratios": [4, 1.1],
                                                                  "hspace": 0.05, "wspace": 0.12})
    for j, (model, title, T, clean, rows) in enumerate(panels):
        ax, dots = axes[0, j], axes[1, j]
        x = np.arange(len(rows))
        ref = [100 * r["all_refusal"] for _, r, _ in rows]
        ax.bar(x, ref, color=[C["hook"] if k == "ablate" else C["actadd"] for _, _, k in rows], width=0.7)
        sel = [100 * _sel(model, k, ",".join(map(str, l))) for l, _, k in rows]
        ax.plot(x, sel, "D", mfc="none", mec="k", ms=7, label="same config, 50 selection prompts")
        for i, v in enumerate(ref):
            ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=10, weight="bold")
        ax.axhline(100 * clean["all_refusal"], color=C["clean"], ls="--", lw=1.3)
        ax.text(len(rows) - 0.55, 100 * clean["all_refusal"] + 1.5, f"clean {100 * clean['all_refusal']:.0f}%",
                ha="right", color=C["clean"], fontsize=9)
        ax.set_ylim(0, 90); ax.set_xticks([]); ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_title(title, fontsize=11)
        for i, (loops, _, kind) in enumerate(rows):
            for k in range(1, T + 1):
                dots.plot(i, k, "o", ms=7, color=("black" if kind == "ablate" else C["actadd"]) if k in loops else "#dddddd")
        dots.set_yticks(range(1, T + 1)); dots.set_ylim(T + 0.6, 0.4); dots.set_xlim(-0.6, len(rows) - 0.4)
        dots.set_xticks(x); dots.set_xticklabels(["actadd" if k == "actadd" else "" for _, _, k in rows], fontsize=9,
                                                 color=C["actadd"])
        dots.set_ylabel("loop"); dots.spines[["left", "bottom"]].set_visible(False)
    axes[0, 0].set_ylabel("judge refusal, all samples (%)")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    fig.suptitle("Test set (487 prompts): hooks in a subset of loops (black dots). Later loops carry the effect; "
                 "the 50-prompt selection set (◇) was optimistic", fontsize=11)
    fig.savefig(OUT / "loops_test.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- 10. loop subsets: ablation needs a late loop, actadd is a dose
def fig_loops_dose():
    o, n = "Ouro-1.4B-Thinking", "Nanbeige4.2-3B"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True, gridspec_kw={"width_ratios": [4, 4, 2.6]})
    cmap = plt.get_cmap("viridis")
    for ax, model, kind, T, title in ((axes[0], o, "ablate", 4, "Ouro: ablation"),
                                      (axes[1], o, "actadd", 4, "Ouro: actadd c = −1"),
                                      (axes[2], n, None, 2, "Nanbeige: ablation (●), actadd c = −2 (▲)")):
        d = pd.read_csv(ROOT / "results" / model / "loop_sweep_perloop_compare.csv")
        clean = d[d.kind == "none"].all_refusal.iloc[0]
        ax.axhline(100 * clean, color=C["clean"], ls="--", lw=1.2)
        ax.text(0.55, 100 * clean + 1.5, "clean", color=C["clean"], fontsize=9)
        kinds = [kind] if kind else ["ablate", "actadd"]
        for kk in kinds:
            s = d[(d.kind == kk) & (d.method == "v4_baseline")]
            for variant, filled in (("shared", True), ("perloop", False)):
                v = s[s.variant == variant]
                for _, r in v.iterrows():
                    loops = [int(t) for t in r.loops_label.split(",")]
                    off = (0.12 if kk == "actadd" and not kind else 0) + (0 if filled else 0.25)
                    xpos = len(loops) + off - 0.12
                    col = cmap((max(loops) - 1) / 3)
                    ax.plot(xpos, 100 * r.all_refusal, "^" if kk == "actadd" else "o", ms=8,
                            mfc=col if filled else "none", mec=col, mew=1.5)
                    if filled and T == 4 and len(loops) <= 2:
                        ax.text(xpos - 0.07, 100 * r.all_refusal, r.loops_label, ha="right", va="center", fontsize=7.5,
                                color="#444")
                means = v.groupby(v.loops_label.str.count(",") + 1).all_refusal.mean() * 100
                if filled:
                    ax.plot(means.index - 0.12 + (0.12 if kk == "actadd" and not kind else 0), means.values, "-",
                            color="#999", lw=1, zorder=0)
        ax.set_xticks(range(1, T + 1)); ax.set_xlim(0.4, T + 0.6)
        ax.set_xlabel("number of loops intervened in"); ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("judge refusal, all samples (%)"); axes[0].set_ylim(0, 85)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 4))
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01, ticks=[1, 2, 3, 4]); cb.set_label("latest loop in the set")
    fig.suptitle("50 selection prompts, every subset of loops. Filled = last-loop direction in every loop, hollow = each "
                 "loop's own direction.\nAblation needs one late loop (loop 1 alone: 40%); actadd falls with the "
                 "number of loops (a dose)", fontsize=11, y=1.06)
    fig.savefig(OUT / "loops_dose.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- 11. collateral: capability and over-refusal
def fig_capability():
    cap = pd.read_csv(ROOT / "results" / "capability_summary.csv")
    cap = cap[cap.tag == "cap"]
    o, n, g = "Ouro-1.4B-Thinking", "Nanbeige4.2-3B", "gpt-oss-20b"
    models = [(o, "Ouro-1.4B"), (n, "Nanbeige4.2-3B"), (g, "gpt-oss-20b")]
    configs = {  # label -> ({model: candidate}, color)
        "clean": ({o: "none", n: "none", g: "none"}, C["clean"]),
        "ablate (all loops)": ({o: "ablate_v4_baseline_t3.l16", n: "ablate_v4_baseline_t1.l15",
                                g: "ablate_v4_baseline_l15"}, C["hook"]),
        "ablate, loop 1 only": ({o: "ablate_v4_baseline_t3.l16_apply0", n: "ablate_v4_baseline_t1.l15_apply0"}, "#9ecae1"),
        "ablate, late loop only": ({o: "ablate_v4_baseline_t3.l16_apply2", n: "ablate_v4_baseline_t1.l15_apply1"},
                                   "#4292c6"),
        "weight ortho": ({g: "ortho_v4_baseline_l15"}, C["ortho"]),
        "actadd (Ouro c=−1, others c=−2)": ({o: "actadd_v4_baseline_t3.l16_c-1", n: "actadd_v4_baseline_t1.l15_c-2",
                                             g: "actadd_v4_baseline_l13_c-2"}, C["actadd"]),
    }
    w = 0.14
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))

    def bars(ax, value):
        for j, (lab, (cands, col)) in enumerate(configs.items()):
            for i, (model, _) in enumerate(models):
                v = value(model, cands[model]) if model in cands else None
                if v is None:
                    continue
                x = i + (j - 2.5) * w
                ax.bar(x, v[0], w * 0.95, yerr=v[1], color=col, capsize=2, label=lab if lab not in seen else None)
                seen.add(lab)
                if v[1] is None:
                    ax.text(x, v[0] + 0.2, f"{v[0]:.1f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(len(models))); ax.set_xticklabels([m for _, m in models])

    def acc(ds):
        def f(model, cand):
            r = cap[(cap.model == model) & (cap.dataset == ds) & (cap.candidate == cand)]
            return (100 * r.accuracy.iloc[0], 100 * r.se.iloc[0]) if len(r) else None
        return f

    seen = set()
    for ax, ds, title, ylim in ((axes[0], "math500", "MATH-500 (500 problems × 1)", (50, 95)),
                                (axes[1], "aime25", "AIME 2025 (30 problems × 8)", (0, 80))):
        bars(ax, acc(ds))
        ax.set_ylim(*ylim); ax.set_title(title, fontsize=11); ax.set_ylabel("accuracy (%) ± s.e.")
    ov = {m: pd.read_csv(f).set_index("candidate") for m, _ in models
          if (f := ROOT / "results" / m / "overrefusal.csv").exists()}
    bars(axes[2], lambda m, c: (100 * ov[m].loc[c, "all_refusal"], None) if m in ov and c in ov[m].index else None)
    axes[2].set_ylim(0, 10); axes[2].set_ylabel("refusal on harmless prompts (%)")
    axes[2].set_title("Over-refusal (250 harmless test prompts)", fontsize=11)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Collateral effects (CoT ≤ 8192 tokens): ablation is free in the looped models; on gpt-oss it lengthens "
                 "CoTs, so more hit the cap (MATH −3, AIME −12 points);\nactadd costs more everywhere; no intervention "
                 "adds refusals", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "capability.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- 12. MoE: why the edit is still an ablation
def fig_moe():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4.6); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=10, ec="k"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=fc, ec=ec))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arr(x0, y0, x1, y1, color="k", ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, ls=ls,
                                     lw=1.3))

    ax.plot([0.2, 13.8], [0.55, 0.55], color="#555", lw=6, alpha=0.25, solid_capstyle="round")
    ax.text(0.25, 0.15, "residual stream h (one vector per token; the direction u lives here)", fontsize=9.5, color="#555")
    box(0.5, 1.5, 1.4, 0.8, "RMSNorm", "#eee")
    arr(1.2, 0.6, 1.2, 1.5)
    box(2.4, 1.5, 1.6, 0.8, "router\ntop-4 of 32", "#fff2cc")
    arr(1.9, 1.9, 2.4, 1.9)
    ys = [3.55, 2.75, 1.95, 1.15]
    for k, y in enumerate(ys):
        sel = k < 2
        label = ("selected expert:  W_down · act + b", "selected expert:  W_down · act + b",
                 "not selected for this token (edited too)", "… 32 experts per layer")[k]
        box(4.8, y, 3.0, 0.62, label, "#e5f5e0" if sel else "#f4f4f4", fs=9, ec="k" if sel else "#bbb")
        if sel:
            arr(4.0, 1.9, 4.8, y + 0.31, color="#2ca02c")
    ax.text(6.3, 4.35, "edit every expert: W_down ← (I − uuᵀ) W_down,  b ← b − (b·u) u", ha="center", fontsize=9.5,
            color=C["ortho"])
    box(8.6, 1.9, 2.0, 0.9, "Σₖ wₖ(h) · expertₖ\n(router weights)", "#e8f0fa", fs=9.5)
    for y in ys[:2]:
        arr(7.8, y + 0.31, 8.6, 2.35, color="#2ca02c")
    arr(10.6, 2.35, 11.3, 0.62)
    ax.text(11.0, 1.55, "+", fontsize=18, ha="center")
    ax.text(11.45, 3.2, "every term ⊥ u  ⇒  the sum ⊥ u\nfor any routing: exact ablation", fontsize=10.5,
            color=C["hook"], weight="bold")
    ax.text(11.45, 2.2, "traps:\n• biases left unedited keep writing\n  a constant along u\n• MXFP4 re-quantization of the\n"
            "  edited experts brings u back", fontsize=9, color=C["ortho"], va="top")
    fig.suptitle("MoE layer (gpt-oss-20b): hooks act on the summed output; the weight edit must cover every expert and "
                 "every bias", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "moe.png", dpi=150); plt.close(fig)


def copy_sweep_figures():
    """Loop-subset sweep and loop x layer map figures (made by scripts/loop_sweep_summary.py and
    scripts/loop_layer_summary.py under data/runs/<model>/figures) -> docs/figures."""
    import shutil

    for model, name, out in (("Ouro-1.4B-Thinking", "loop_sweep_perloop_compare", "loops_ouro.png"),
                             ("Nanbeige4.2-3B", "loop_sweep_perloop_compare", "loops_nanbeige.png"),
                             ("Ouro-1.4B-Thinking", "loop_layer_map", "loop_layer_map.png")):
        src = RUNS / model / "figures" / f"{name}.png"
        if src.exists():
            shutil.copy(src, OUT / out)


if __name__ == "__main__":
    fig_interventions(); fig_norm_leak(); fig_loop_growth(); fig_compounding(); fig_loop_count()
    fig_results(); fig_sites(); copy_sweep_figures()
    fig_models(); fig_pipeline(); fig_loops_test(); fig_loops_dose(); fig_capability(); fig_moe()
    print("figures ->", OUT)
