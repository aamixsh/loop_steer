#!/usr/bin/env python3
"""CPU-only figures for docs/presentation_clean.md.

Default: redraw from the portable, numerical evidence snapshot beside the figures.
--refresh-data: rebuild that snapshot from completed runs (never reads gpt-oss jobs).
Uses the existing project environment; does not load any model or change run data.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "scratch/presentation-cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

OUT = ROOT / "docs/figures/clean"
RUNS = ROOT / "data/runs"
Q, O, N = "Qwen3-8B", "Ouro-1.4B-Thinking", "Nanbeige4.2-3B"
C = dict(ink="#152A43", muted="#66768A", grid="#DFE5EC", clean="#788598",
         random="#B8C1CD", hook="#3157D5", ortho="#C47716", actadd="#087F73",
         signal="#7C4DB5", fail="#C64B52", pale="#EDF2FA")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "text.color": C["ink"],
                     "axes.labelcolor": C["muted"], "xtick.color": C["muted"],
                     "ytick.color": C["ink"], "axes.spines.top": False,
                     "axes.spines.right": False, "axes.spines.left": False,
                     "axes.spines.bottom": False, "axes.titleweight": "bold",
                     "axes.titlepad": 18, "svg.fonttype": "path", "pdf.fonttype": 42,
                     "savefig.facecolor": "#F8FAFC", "figure.facecolor": "#F8FAFC",
                     "axes.facecolor": "none", "axes.axisbelow": True})


def records(path):
    return json.loads(pd.read_csv(path).to_json(orient="records"))


def refresh():
    from summarize import summarize
    sources = []

    def row(model, run, candidate, cap, ext=None):
        path = RUNS / model / "generations" / run / f"{candidate}.scored.parquet"
        continuation = None if ext is None else path.parent.parent / ext
        sources.append({"path": str(path.relative_to(ROOT)), "initial_cap": cap,
                        "continuations": str(continuation.relative_to(ROOT)) if continuation else None})
        return summarize(path, max_tokens=cap, cont_dir=continuation)

    headline = {
        Q: [row(Q, "clean_test_8k", "none", 8192),
            row(Q, "test_hooks", "ablate_random_l17", 2048, "test_hooks_ext"),
            row(Q, "test_hooks", "ablate_v4_baseline_l21", 2048, "test_hooks_ext"),
            row(Q, "test_ortho", "ortho_v4_baseline_l21", 2048, "test_ortho_ext"),
            row(Q, "test_hooks", "actadd_v4_baseline_l17_c-2", 2048, "test_hooks_ext")],
        O: [row(O, "clean_test_8k", "none", 8192),
            row(O, "test_hooks", "ablate_random_t3.l16", 2048, "test_hooks_ext"),
            row(O, "test_hooks", "ablate_v4_baseline_t3.l16", 2048, "test_hooks_ext"), None,
            row(O, "test_loops", "actadd_v4_baseline_t3.l16_c-1", 2048, "test_loops_ext")],
        N: [row(N, "clean_test", "none", 4096, "clean_test_ext"),
            row(N, "test_int", "ablate_random_t0.l14", 4096, "test_int_ext"),
            row(N, "test_int", "ablate_v4_baseline_t1.l15", 4096, "test_int_ext"),
            row(N, "test_int", "ortho_v4_baseline_t1.l15", 4096, "test_int_ext"),
            row(N, "test_loops", "actadd_v4_baseline_t1.l15_c-2", 4096, "test_loops_ext")],
    }
    d = {"description": "Numerical snapshot for presentation_clean.md; no prompts or generations.",
         "headline": headline, "sources": sources, "loops": {}, "rebuild": {},
         "norms": {}, "positions": {}, "auc": {}, "capability": records(ROOT / "results/capability_summary.csv"),
         "dose": records(ROOT / f"results/{O}/loop_sweep_perloop_compare.csv"),
         "layer_map": records(ROOT / f"results/{O}/loop_layer_map.csv")}
    for model, site, count in [(O, "t3l16", 4), (N, "t1l15", 2)]:
        analysis = RUNS / model / "analysis"
        d["loops"][model] = records(ROOT / f"results/{model}/test_loops.csv")
        rebuild = pd.read_csv(analysis / f"loop_rebuild_v4_baseline_{site}.csv")
        shown = ["clean", *map(str, range(1, count + 1)), ",".join(map(str, range(1, count + 1)))]
        rebuild = rebuild[(rebuild.window == "prompt_end") & rebuild.condition.isin(shown)]
        d["rebuild"][model] = json.loads(rebuild[["condition", "loop", "layer", "window", "dir", "sep", "sep_clean"]].to_json(orient="records"))
        normpath = analysis / f"ortho_compare_v4_baseline_{site}.json"
        d["norms"][model] = json.loads(normpath.read_text())
        d["positions"][model] = records(analysis / "select_pos.csv")
        for name in [f"loop_rebuild_v4_baseline_{site}.csv", normpath.name, "select_pos.csv"]:
            sources.append({"path": str((analysis / name).relative_to(ROOT))})
    for model in (Q, O, N):
        auc_path = RUNS / model / "analysis/v4_baseline_sites.csv"
        d["auc"][model] = records(auc_path)
        sources.append({"path": str(auc_path.relative_to(ROOT))})
    d["reduced"] = records(RUNS / O / "analysis/loops2.csv")
    d["norm_aware"] = records(ROOT / f"results/{O}/ortho_na.csv")
    for path in [ROOT / "results/capability_summary.csv", ROOT / f"results/{O}/loop_sweep_perloop_compare.csv",
                 ROOT / f"results/{O}/loop_layer_map.csv", RUNS / O / "analysis/loops2.csv",
                 ROOT / f"results/{O}/ortho_na.csv", *[ROOT / f"results/{m}/test_loops.csv" for m in (O, N)]]:
        sources.append({"path": str(path.relative_to(ROOT))})
    # DataFrames convert missing values to JSON null (not non-standard NaN).
    def sanitize(value):
        if isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items() if k != "generations"}
        if isinstance(value, list):
            return [sanitize(v) for v in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        return value
    (OUT / "evidence.json").write_text(json.dumps(sanitize(d), indent=2, allow_nan=False) + "\n")


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def box(ax, x, y, w, h, text, fc=C["pale"], color=C["ink"], fontsize=16, ec="none"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                              facecolor=fc, edgecolor=ec, linewidth=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=fontsize)


def arrow(ax, start, end, color=C["muted"], lw=2):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=color,
                                                       lw=lw, mutation_scale=16))


def models():
    fig, ax = plt.subplots(figsize=(13, 3.1))
    ax.axis("off"); ax.set(xlim=(0, 13), ylim=(-0.3, 3.2))
    for y, name, layers, loops in [(2.55, "Qwen3-8B", 36, 1), (1.35, "Ouro-1.4B", 24, 4),
                                  (0.15, "Nanbeige-3B", 22, 2)]:
        ax.text(0, y + 0.17, name, fontsize=19, weight="bold")
        ax.text(0, y - 0.17, f"{layers} layers × {loops} pass{'es' if loops > 1 else ''}", fontsize=14, color=C["muted"])
        for t in range(loops):
            x = 3.4 + t * 2.15
            box(ax, x, y - .3, 1.48, .67, f"{layers} layers", fc=C["hook"], color="white")
            if t < loops - 1:
                arrow(ax, (x + 1.48, y + .035), (x + 2.1, y + .035))
                ax.text(x + 1.81, y + .47, "norm", color=C["ortho"], ha="center", fontsize=12)
        ax.text(12.85, y, f"{layers * loops}\nsites", ha="right", va="center", fontsize=17, weight="bold")
    save(fig, "models")


def sampling():
    fig, ax = plt.subplots(figsize=(12.8, 1.85))
    ax.axis("off"); ax.set(xlim=(0, 13), ylim=(0, 2))
    for x, num, title, detail in [(0, "500", "LEARN DIRECTIONS", "training prompts 1–500"),
                                 (4.5, "50", "SELECT CANDIDATES", "training prompts 501–550"),
                                 (9, "487", "EVALUATE", "held-out test prompts")]:
        box(ax, x, .1, 3.8, 1.8, "", fc=C["pale"])
        ax.text(x+.25, 1.52, num, fontsize=36, weight="bold", va="center", color=C["hook"])
        ax.text(x+.25, .82, title, fontsize=13, weight="bold")
        ax.text(x+.25, .4, detail, fontsize=13, color=C["muted"])
        if x < 9: arrow(ax, (x+3.94, 1), (x+4.33, 1))
    save(fig, "sampling")


def direction():
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 2.65), gridspec_kw={"width_ratios": [1, 1.7]})
    ax = axes[0]; ax.axis("off"); ax.set(xlim=(-2, 2.8), ylim=(-1.5, 1.7))
    rng = np.random.default_rng(4)
    for mu, col, label in [(np.array([-1, -.35]), C["actadd"], "complying"),
                           (np.array([1.4, .6]), C["fail"], "refusing")]:
        a = mu + rng.normal(size=(20, 2)) * .24
        ax.scatter(a[:,0], a[:,1], color=col, alpha=.3, s=35)
        ax.scatter(*mu, color=col, s=150, edgecolor="white", linewidth=2, zorder=3)
        ax.text(mu[0], mu[1] - .85, label, ha="center", color=col, fontsize=17)
    arrow(ax, (-1,-.35), (1.4,.6), C["signal"], 3)
    ax.text(.1,.6,"r", color=C["signal"], fontsize=24)
    ax.text(.2,1.45,"Schematic activation space", fontsize=13, ha="center", color=C["muted"])
    ax = axes[1]; ax.axis("off")
    ax.text(.02,.75,"r = mean(refusing) − mean(complying)", fontsize=23, weight="bold")
    ax.text(.02,.45,"u = r / ‖r‖", fontsize=24, color=C["signal"])
    ax.text(.02,.13,"One vector per model, token window, and (loop, layer).", fontsize=16)
    save(fig, "direction")


def headline(d):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    labels = ["Clean", "Random ablation", "Hook ablation", "Weight edit", "Activation addition"]
    colors = [C[x] for x in ["clean", "random", "hook", "ortho", "actadd"]]
    for ax, model, title in zip(axes, (Q,O,N), ("Qwen3-8B", "Ouro-1.4B", "Nanbeige-3B")):
        for y, (r,col) in enumerate(zip(d["headline"][model], colors)):
            if r is None:
                ax.text(8,y,"×  generation fails*", color=C["fail"], va="center", fontsize=14)
                continue
            v = r["all_refusal"] * 100
            ax.plot([0,v], [y,y], color=col, lw=4, alpha=.3, solid_capstyle="round")
            ax.scatter(v,y, color=col, s=110, zorder=3)
            ax.text(v+4,y,f"{v:.1f}%",va="center",fontsize=15,weight="bold", color=col if y>1 else C["ink"])
        ax.set(xlim=(0,106), ylim=(4.7,-.7), xticks=[0,50,100], yticks=range(5), title=title)
        ax.set_yticklabels(labels, fontsize=15); ax.tick_params(length=0)
        ax.grid(axis="x", color=C["grid"], lw=.7)
        ax.set_xlabel("Refusal (%)",fontsize=14)
    fig.subplots_adjust(left=.18,right=.99,bottom=.18,top=.86,wspace=.16)
    save(fig,"results")


def readability(d):
    fig, axes=plt.subplots(1,3,figsize=(13,3),sharey=True)
    for ax, model, L, T, title in zip(axes,(Q,O,N),(36,24,22),(1,4,2),("Qwen3-8B","Ouro-1.4B","Nanbeige-3B")):
        a=pd.DataFrame(d["auc"][model]); x=a.loop*L+a.layer
        ax.plot(x,a.auc,color=C["hook"],lw=3,label="Refusal direction")
        ax.plot(x,a.auc_random,color=C["random"],lw=1.6,label="Random direction")
        ax.axhline(.5,color=C["muted"],lw=1,ls=":")
        for t in range(1,T): ax.axvline(t*L-.5,color=C["grid"],lw=1)
        ax.set(xlim=(0,L*T-1),ylim=(.46,1.02),title=title,yticks=[.5,.75,1])
        ax.set_xticks([t*L+(L-1)/2 for t in range(T)])
        ax.set_xticklabels([f"Loop {t+1}" for t in range(T)] if T>1 else ["Layers 0–35"],fontsize=13)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Held-out AUC",fontsize=15)
    axes[0].legend(frameon=False,fontsize=12,loc="lower right")
    fig.subplots_adjust(left=.075,right=.99,bottom=.2,top=.83,wspace=.16)
    save(fig,"readability")


def loops(d):
    fig, axes=plt.subplots(1,2,figsize=(13,3.6),gridspec_kw={"width_ratios":[1.65,1]})
    for ax,model,T,title in zip(axes,(O,N),(4,2),("Ouro-1.4B","Nanbeige-3B")):
        rows=pd.DataFrame(d["loops"][model]).set_index("candidate")
        stem="ablate_v4_baseline_t3.l16" if model==O else "ablate_v4_baseline_t1.l15"
        vals=[rows.loc[f"{stem}_apply{t}","all_refusal"]*100 for t in range(T)]
        labels=[str(t+1) for t in range(T)]
        if model==O:
            vals.append(rows.loc[f"{stem}_apply2-3","all_refusal"]*100); labels.append("3 + 4")
        vals.append(d["headline"][model][2]["all_refusal"]*100); labels.append("All")
        cols=[C["hook"]]*(len(vals)-1)+[C["signal"]]
        ax.bar(range(len(vals)),vals,color=cols,width=.6)
        for i,v in enumerate(vals): ax.text(i,v+2,f"{v:.0f}%",ha="center",weight="bold",fontsize=18)
        clean=d["headline"][model][0]["all_refusal"]*100
        ax.axhline(clean,color=C["clean"],ls="--",lw=1.5)
        ax.text(len(vals)-.45,clean+1.5,f"Clean {clean:.0f}%",ha="right",fontsize=13,color=C["clean"])
        ax.set(ylim=(0,92),xticks=range(len(vals)),xticklabels=labels,title=title,xlabel="Loops with ablation")
        ax.set_yticks([0,40,80]); ax.tick_params(length=0); ax.grid(axis="y",color=C["grid"],lw=.7)
    axes[0].set_ylabel("Refusal (%)")
    fig.subplots_adjust(left=.07,right=.98,bottom=.2,top=.85,wspace=.22)
    save(fig,"loops")


def rebuild(d):
    fig, axes=plt.subplots(1,2,figsize=(13,3.2),gridspec_kw={"width_ratios":[1.65,1]})
    for ax,model,T,title in zip(axes,(O,N),(4,2),("Ouro: signal in loop 4","Nanbeige: signal in loop 2")):
        a=pd.DataFrame(d["rebuild"][model]); a=a[(a.loop==T)&(a.dir=="own")&(a.window=="prompt_end")]
        ratio=a.groupby("condition").apply(lambda g:100*g.sep.sum()/g.sep_clean.sum(),include_groups=False)
        conds=[str(t) for t in range(1,T+1)]+[",".join(map(str,range(1,T+1)))]
        vals=ratio.loc[conds].to_numpy(); xs=np.arange(len(vals))
        ax.bar(xs,vals,color=[C["hook"]]*T+[C["signal"]],width=.6)
        for x,v in zip(xs,vals): ax.text(x,v+3,f"{v:.0f}%",ha="center",fontsize=20,weight="bold")
        ax.axhline(100,color=C["clean"],lw=1.2,ls="--")
        ax.set(ylim=(0,118),xticks=xs,xticklabels=list(map(str,range(1,T+1)))+["All"],
               yticks=[0,50,100],title=title,xlabel="Loops with ablation")
        ax.tick_params(length=0); ax.grid(axis="y",color=C["grid"],lw=.7)
    axes[0].set_ylabel("Separation / clean (%)")
    fig.subplots_adjust(left=.075,right=.98,bottom=.21,top=.82,wspace=.22)
    save(fig,"rebuild")


def recovery(d):
    """Track the exact erased vector, keeping its direction fixed across depth."""
    fig,axes=plt.subplots(1,2,figsize=(13,3.5),gridspec_kw={"width_ratios":[1.6,1]})
    for ax,model,L,T,title in zip(axes,(O,N),(24,22),(4,2),("Ouro-1.4B","Nanbeige-3B")):
        a=pd.DataFrame(d["rebuild"][model]); a=a[(a.dir=="u")&(a.window=="prompt_end")]
        normalizer=a[(a.condition=="clean")&(a.loop==T)].sep.mean()
        ax.axvspan(.85,1.15,color="#E6ECFA")
        for cond,label,col in [("clean","Clean",C["clean"]),("1","Ablate loop 1",C["hook"])]:
            sub=a[a.condition==cond].groupby("loop").sep.mean()/normalizer
            ax.plot(sub.index,sub.values,color=col,lw=3,marker="o",ms=9,label=label,
                    ls="--" if cond=="clean" else "-")
        recovered=a[(a.condition=="1")&(a.loop==T)].sep.mean()/normalizer
        ax.annotate(f"{recovered:.0%} recovered",xy=(T,recovered),xytext=(T-.1,recovered+(.26 if model==N else -.26)),
                    ha="right",color=C["hook"],fontsize=17,weight="bold")
        ax.axhline(0,color=C["muted"],lw=.8)
        ax.set(xlim=(.8,T+.2),ylim=(-.18,1.5),title=title,yticks=[0,.5,1,1.5])
        ax.set_xticks(range(1,T+1))
        ax.set_xticklabels([f"Loop {t+1}" for t in range(T)],fontsize=14)
        ax.tick_params(length=0)
        ax.text(1,.08,"erased",color=C["hook"],ha="center",fontsize=13)
        if model==O:ax.legend(frameon=False,fontsize=13,loc="upper left")
    axes[0].set_ylabel("Separation along the erased direction\n(clean final-loop mean = 1)",fontsize=14)
    fig.subplots_adjust(left=.095,right=.99,top=.84,bottom=.17,wspace=.2)
    save(fig,"recovery")


def positions(d):
    fig,axes=plt.subplots(1,2,figsize=(13,3.4),sharey=True)
    names=["Clean","Prompt","Thinking","Answer","Prompt + thinking","All tokens"]
    for ax,model,stem,title in zip(axes,(O,N),("ablate_v4_baseline_t3.l16","ablate_v4_baseline_t1.l15"),("Ouro-1.4B","Nanbeige-3B")):
        rows=pd.DataFrame(d["positions"][model]).set_index("candidate")
        keys=["none",stem+"_posprompt",stem+"_poscot",stem+"_posanswer",stem+"_posprompt-cot",stem]
        for i,key in enumerate(keys):
            r=rows.loc[key]; v=r.all_refusal
            if pd.isna(v):
                assert r.invalid_cot_frac==0, "Missing all-sample score with invalid samples"
                v=r.judge_refusal
            v*=100
            col=C["clean"] if i==0 else C["signal"] if i==5 else C["hook"]
            ax.barh(i,v,color=col,height=.48,alpha=.9)
            ax.text(v+2,i,f"{v:.0f}%",va="center",fontsize=16,weight="bold")
        ax.set(xlim=(0,100),ylim=(5.7,-.7),xticks=[0,50,100],yticks=range(6),title=title,xlabel="Refusal (%)")
        ax.set_yticklabels(names,fontsize=15); ax.tick_params(length=0); ax.grid(axis="x",color=C["grid"],lw=.7)
    fig.subplots_adjust(left=.22,right=.97,top=.85,bottom=.18,wspace=.2)
    save(fig,"positions")


def mechanism():
    fig,ax=plt.subplots(figsize=(13,2.95)); ax.axis("off"); ax.set(xlim=(0,13),ylim=(0,3))
    for y,label,last,fc in [(2.05,"Weight edit","u returns",C["ortho"]),(.65,"Hook ablation","u removed",C["hook"])]:
        ax.text(.05,y,label,fontsize=19,weight="bold",va="center")
        for x,w,txt,col in [(2.6,2.35,"Matrix write",C["pale"]),(5.5,2.4,"Uneven norm", "#FCEDD7"),
                            (8.5,2.2,"Project out u" if y<1 else "Residual stream",C["pale"])]:
            box(ax,x,y-.42,w,.84,txt,fc=col,fontsize=16)
            if x<8: arrow(ax,(x+w+.05,y),(x+w+.5,y))
        ax.text(12.8,y,last,ha="right",va="center",fontsize=17,color=fc,weight="bold")
        if y>1:
            ax.text(3.77,y-.7,"write · u = 0",ha="center",fontsize=14,color=C["ortho"])
        else:
            ax.text(9.6,y-.7,"after normalization",ha="center",fontsize=14,color=C["hook"])
    save(fig,"mechanism")


def norm_growth(d):
    fig,axes=plt.subplots(1,2,figsize=(13,3.2),gridspec_kw={"width_ratios":[1.5,1]})
    for ax,model,title in zip(axes,(O,N),("Ouro: leak compounds","Nanbeige: leak stays small")):
        raw=d["norms"][model]["full"]; T=raw["T"]; x=np.arange(1,T+1)
        clean=np.array([raw["clean"]["refusal"][str(t)]["resid_rms"] for t in range(T)])
        for key,label,col in [("ortho","Weight edit",C["ortho"]),("ortho_na","Norm-aware",C["signal"]),("hook","Hooks",C["hook"])]:
            if key not in raw: continue
            v=np.array([raw[key]["refusal"][str(t)]["resid_rms"] for t in range(T)])/clean
            ax.plot(x,v,color=col,lw=3,marker="o",ms=7)
            ypos = v[-1] if model==O else .55 if key=="ortho" else -.12
            ax.annotate(f"{label}  {v[-1]:.2f}×" if key!="hook" else "Hooks < 0.004×",
                        xy=(T,v[-1]),xytext=(T+.14,ypos),va="center",color=col,fontsize=13,
                        arrowprops=dict(arrowstyle="-",color=col,lw=.8))
        ax.axhline(1,color=C["clean"],ls="--",lw=1.2)
        ax.text(.98,1.08,"Clean = 1",color=C["clean"],fontsize=12)
        ax.set(xlim=(.9,T+1.8),ylim=(-.17,6.3),xticks=x,yticks=[0,1,3,5],title=title,xlabel="Loop")
        ax.tick_params(length=0); ax.grid(axis="y",color=C["grid"],lw=.7)
    axes[0].set_ylabel("RMS projection / clean")
    fig.subplots_adjust(left=.075,right=.99,top=.83,bottom=.2,wspace=.18)
    save(fig,"norm_growth")


def dose_layers(d):
    fig,(ax,hx)=plt.subplots(1,2,figsize=(13,3.4),gridspec_kw={"width_ratios":[1,1.2]})
    a=pd.DataFrame(d["dose"]); a=a[(a.variant=="shared")&(a.method=="v4_baseline")]
    for kind,label,color in [("ablate","Ablation",C["hook"]),("actadd","Addition",C["actadd"])]:
        sub=a[a.kind==kind]
        for n,g in sub.groupby("n_loops"):
            xs=np.linspace(n-.08,n+.08,len(g))
            ax.scatter(xs,g.all_refusal*100,color=color,s=40,alpha=.35)
        means=sub.groupby("n_loops").all_refusal.mean()*100
        ax.plot(means.index,means.values,color=color,marker="o",lw=2.5,label=label)
    ax.set(xlim=(.65,4.4),ylim=(0,60),xticks=[1,2,3,4],yticks=[0,20,40,60],xlabel="Number of loops intervened on",ylabel="Refusal (%)",title="Addition accumulates across loops")
    ax.legend(frameon=False,fontsize=13); ax.tick_params(length=0); ax.grid(axis="y",color=C["grid"],lw=.7)
    a=pd.DataFrame(d["layer_map"]); p=a.pivot(index="loop",columns="layer",values="all_refusal")*100
    hx.imshow(p,cmap="Blues",vmin=0,vmax=60,aspect="auto")
    for (y,x),v in np.ndenumerate(p.to_numpy()):
        hx.text(x,y,f"{v:.0f}",ha="center",va="center",color="white" if v>34 else C["ink"],fontsize=17,weight="bold")
    hx.set(xticks=range(len(p.columns)),xticklabels=p.columns,yticks=range(len(p.index)),
           yticklabels=[f"Loop {i}" for i in p.index],xlabel="Layer receiving the addition",title="One loop × one layer: refusal (%)")
    hx.tick_params(length=0)
    fig.subplots_adjust(left=.065,right=.99,top=.84,bottom=.2,wspace=.34)
    save(fig,"dose_layers")


def capability(d):
    a=pd.DataFrame(d["capability"]); a=a[a.dataset=="math500"]
    fig,axes=plt.subplots(1,2,figsize=(13,3.2),sharey=True)
    for ax,model,stem,title in zip(axes,(O,N),("t3.l16","t1.l15"),("Ouro-1.4B","Nanbeige-3B")):
        df=a[a.model==model].set_index("candidate")
        c=-1 if model==O else -2
        keys=["none",f"ablate_v4_baseline_{stem}",f"actadd_v4_baseline_{stem}_c{c}"]
        vals=[df.loc[k,"accuracy"]*100 for k in keys]
        ax.bar(range(3),vals,color=[C["clean"],C["hook"],C["actadd"]],width=.55)
        for x,v in enumerate(vals): ax.text(x,v+2,f"{v:.1f}%",ha="center",fontsize=20,weight="bold")
        ax.set(ylim=(0,100),xticks=range(3),xticklabels=["Clean","Ablation","Addition"],yticks=[0,50,100],title=title)
        ax.tick_params(length=0); ax.grid(axis="y",color=C["grid"],lw=.7)
    axes[0].set_ylabel("MATH-500 accuracy (%)")
    fig.subplots_adjust(left=.08,right=.99,top=.82,bottom=.19,wspace=.22)
    save(fig,"capability")


def moe():
    fig,ax=plt.subplots(figsize=(13,2.35)); ax.axis("off"); ax.set(xlim=(0,13),ylim=(0,2.8))
    box(ax,.1,.75,2,.85,"Token state",fontsize=18)
    arrow(ax,(2.2,1.17),(2.65,1.17))
    box(ax,2.7,.75,2.1,.85,"Router",fontsize=18)
    for i in range(4):
        y=.2+i*.53
        box(ax,5.6,y,2.25,.4,f"Expert {i+1}",fc=C["pale"],fontsize=13)
        arrow(ax,(4.9,1.17),(5.53,y+.2),lw=1.3)
        arrow(ax,(7.92,y+.2),(8.5,1.17),lw=1.3)
    box(ax,8.55,.75,2,.85,"Weighted sum",fontsize=15)
    arrow(ax,(10.6,1.17),(11.1,1.17))
    box(ax,11.15,.75,1.7,.85,"Residual",fc=C["hook"],color="white",fontsize=16)
    ax.text(6.7,2.55,"4 selected from 32 experts per token",ha="center",fontsize=15,color=C["muted"])
    save(fig,"moe")


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--refresh-data",action="store_true")
    args=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    if args.refresh_data: refresh()
    d=json.loads((OUT/"evidence.json").read_text())
    for fn in (models,sampling,direction,mechanism,moe): fn()
    for fn in (headline,readability,loops,rebuild,recovery,positions,norm_growth,dose_layers,capability): fn(d)
    print(f"14 figures (SVG + PNG) and evidence snapshot: {OUT}")


if __name__=="__main__": main()
