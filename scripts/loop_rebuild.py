#!/usr/bin/env python3
"""Do later loops rebuild the refusal signal that ablation removes in earlier loops?

    scripts/loop_rebuild.py --gpu 0 --model ByteDance/Ouro-1.4B-Thinking --direction v4_baseline:t3.l16
    scripts/loop_rebuild.py --gpu 0 --model Nanbeige/Nanbeige4.2-3B --direction v4_baseline:t1.l15

Teacher-forces the model's own clean selection-set traces (formatted prompt + first ``--max-cot-tokens`` CoT tokens),
labelled refusing (all 3 answers score < 0.2) or complying (all > 0.8), under hook ablation of the direction ``u`` in a
subset of loops (same hooks as generation, ``loop_steer.hooks.ablation_hooks``). At every (loop, layer) input it
records each sequence's mean projection over two token windows -- ``prompt_end`` (the template tokens after the user
message, where ``v4_baseline`` is defined) and ``cot`` -- onto two unit directions:

  u      the ablated direction (zero wherever ablation is active)
  own    the site's own clean direction (same method at that (loop, layer)): refusal information the model encodes
         there, whether or not it lies along u

and reports the **separation** = mean(refusing) - mean(complying) and the AUC of that projection. Comparing a condition
with the clean model at sites after the ablated loops shows how much of the signal the later loops rebuild.
Writes analysis/loop_rebuild_<method>_<site>.csv and figures/loop_rebuild_<method>_<site>.png. HF backend, one GPU.
"""

import argparse
import os
from itertools import combinations


def parse_site(site):
    loop, layer = site.replace("t", "").replace("l", "").split(".")
    return int(loop), int(layer)


def auc(pos, neg):
    """P(score of a random refusing sequence > score of a random complying one)."""
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).float().mean() + 0.5 * (diff == 0).float().mean())


def plot(res, path, model_name, direction, T, L):
    """Separation along depth for clean, ablation in each single loop, and in all loops."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = ["clean"] + [str(t) for t in range(1, T + 1)] + [",".join(map(str, range(1, T + 1)))]
    colors = dict(zip(show, ["k", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][: T + 1] + ["#999999"]))
    fig, axes = plt.subplots(2, 2, figsize=(15, 7.5), sharex=True)
    for j, dname in enumerate(("own", "u")):
        for i, window in enumerate(("prompt_end", "cot")):
            ax = axes[i, j]
            for name in show:
                d = res[(res.condition == name) & (res.window == window) & (res.dir == dname)].sort_values(["loop", "layer"])
                ax.plot((d.loop - 1) * L + d.layer, d.sep, color=colors[name], lw=2.4 if name == "clean" else 1.4,
                        ls=":" if name == show[-1] else "-",
                        label="clean" if name == "clean" else f"ablate loop{'s' * (',' in name)} {name}")
            for t in range(1, T):
                ax.axvline(t * L - 0.5, color="#ccc", lw=0.8)
            ax.axhline(0, color="#999", lw=0.6)
            ax.set_title(f"{'prompt-end' if window == 'prompt_end' else 'CoT'} tokens, projection on "
                         f"{'each site’s own refusal direction' if dname == 'own' else 'the ablated direction u'}",
                         fontsize=10)
            ax.set_ylabel("mean refusing − mean complying")
    for ax in axes[1]:
        ax.set_xticks([t * L + L // 2 for t in range(T)]); ax.set_xticklabels([f"loop {t + 1}" for t in range(T)])
    axes[0, 0].legend(fontsize=8.5, ncol=2)
    fig.suptitle(f"{model_name.split('/')[-1]}: refusal signal along depth (clean traces, teacher-forced) when u "
                 f"({direction}) is ablated in one loop or all", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", required=True)
    ap.add_argument("--direction", required=True, help="method:site into directions/train.pt, e.g. v4_baseline:t3.l16")
    ap.add_argument("--max-per-class", type=int, default=40)
    ap.add_argument("--max-cot-tokens", type=int, default=256)
    ap.add_argument("--plot-only", action="store_true", help="Redraw the figure from the saved CSV")
    ap.add_argument("--subsets", default=None, help="Semicolon-separated 1-based loop sets, e.g. '1;1,2;3,4' "
                                                    "(default: every subset)")
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from loop_steer.paths import run_dir, setup_job_env
    setup_job_env()
    import numpy as np
    import pandas as pd
    import torch

    from loop_steer.cot import format_prompt
    from loop_steer.hooks import _layer_input, _loop, ablation_hooks, decoder_layers, hooked
    from loop_steer.models import eoi_length, is_ouro, load_model, load_tokenizer, n_loops

    rd = run_dir(args.model)
    dirs = torch.load(rd / "directions" / "train.pt", weights_only=False)
    method, site = args.direction.split(":")
    sites = dirs["sites"]
    table = dirs[method]["dirs"].float()
    own = table / table.norm(dim=-1, keepdim=True)  # [n_sites, D]
    u = own[sites.index(parse_site(site))]

    tagname = f"{method}_{site.replace('.', '')}"
    if args.plot_only:
        from transformers import AutoConfig

        from loop_steer.models import _remote_kwargs

        res = pd.read_csv(rd / "analysis" / f"loop_rebuild_{tagname}.csv")
        cfg = AutoConfig.from_pretrained(args.model, **_remote_kwargs(args.model))
        plot(res, rd / "figures" / f"loop_rebuild_{tagname}.png", args.model, args.direction, n_loops(cfg),
             cfg.num_hidden_layers)
        return

    tok = load_tokenizer(args.model)
    gen_path = (rd / "generations" / "select_clean.scored.parquet") if is_ouro(args.model) \
        else (rd / "generations" / "clean_select" / "none.scored.parquet")
    df = pd.read_parquet(gen_path).dropna(subset=["sr_score"])
    cots = df.groupby(["prompt_idx", "cot_rep"]).agg(prompt=("prompt", "first"), cot=("cot", "first"),
                                                     lo=("sr_score", "max"), hi=("sr_score", "min")).reset_index()
    ref = cots[cots.lo < 0.2].head(args.max_per_class)
    com = cots[cots.hi > 0.8].head(args.max_per_class)
    n_eoi = eoi_length(tok)
    seqs = []  # (ids, label, prompt_end window, cot window)
    for rows, label in ((ref, 1), (com, 0)):
        for r in rows.itertuples():
            p = tok(format_prompt(tok, r.prompt), add_special_tokens=False).input_ids
            c = tok.encode(r.cot, add_special_tokens=False)[: args.max_cot_tokens]
            seqs.append((torch.tensor([p + c]), label, slice(len(p) - n_eoi, len(p)), slice(len(p), len(p) + len(c))))
    labels = np.array([s[1] for s in seqs])
    print(f"{args.model}: {int(labels.sum())} refusing, {int((1 - labels).sum())} complying CoTs; u = {args.direction}",
          flush=True)

    model = load_model(args.model)
    T, L = n_loops(model.config), len(decoder_layers(model))
    site_idx = {s: i for i, s in enumerate(sites)}
    dev = model.device
    U = torch.stack([u, *own]).to(dev)  # row 0: u; row 1 + i: own direction of site i

    if args.subsets:
        subsets = [tuple(int(x) for x in s.split(",")) for s in args.subsets.split(";")]
    else:
        subsets = [c for k in range(1, T + 1) for c in combinations(range(1, T + 1), k)]
    conditions = [("clean", None)] + [(",".join(map(str, s)), [t - 1 for t in s]) for s in subsets]

    rows = []
    for name, loops in conditions:
        store = {}  # (loop, layer) -> [n_seq, 2 windows, 2 dirs]
        cur = {}

        def capture(layer):
            def hook(module, a, kw):
                t = _loop(kw)
                if (t, layer) not in site_idx:
                    return
                x = _layer_input(a, kw)[0].float()  # [S, D], after any ablation pre-hook
                proj = x @ U[[0, 1 + site_idx[(t, layer)]]].T  # [S, 2]
                store.setdefault((t, layer), []).append(
                    torch.stack([proj[cur["w"][0]].mean(0), proj[cur["w"][1]].mean(0)]).cpu())
            return hook

        abl = ablation_hooks(model, u, loops=loops) if loops is not None else (lambda: [])
        caps = lambda: [b.register_forward_pre_hook(capture(l), with_kwargs=True)
                        for l, b in enumerate(decoder_layers(model))]
        with torch.inference_mode(), hooked(abl), hooked(caps):
            for ids, _, w_p, w_c in seqs:
                cur["w"] = (w_p, w_c)
                model(input_ids=ids.to(dev), use_cache=False)
        for (t, l), vals in store.items():
            v = torch.stack(vals)  # [n_seq, 2, 2]
            for wi, window in enumerate(("prompt_end", "cot")):
                for di, dname in enumerate(("u", "own")):
                    x = v[:, wi, di]
                    pos, neg = x[labels == 1], x[labels == 0]
                    rows.append({"condition": name, "loop": t + 1, "layer": l, "window": window, "dir": dname,
                                 "sep": float(pos.mean() - neg.mean()), "auc": auc(pos, neg),
                                 "mean_ref": float(pos.mean()), "mean_com": float(neg.mean())})
        print(f"  {name:10s} done", flush=True)

    res = pd.DataFrame(rows)
    clean = res[res.condition == "clean"].set_index(["loop", "layer", "window", "dir"])
    res["sep_clean"] = res.set_index(["loop", "layer", "window", "dir"]).index.map(clean.sep)
    res["auc_clean"] = res.set_index(["loop", "layer", "window", "dir"]).index.map(clean.auc)
    (rd / "analysis").mkdir(exist_ok=True)
    res.to_csv(rd / "analysis" / f"loop_rebuild_{tagname}.csv", index=False)

    # summary: signal in the last loop relative to clean (own direction), by condition
    last = res[(res.loop == T) & (res.dir == "own")]
    summ = last.groupby(["condition", "window"]).apply(
        lambda g: pd.Series({"sep_ratio": g.sep.sum() / g.sep_clean.sum(), "auc": g.auc.mean(),
                             "auc_clean": g.auc_clean.mean()}), include_groups=False).reset_index()
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(f"\nlast loop ({T}), own directions: separation relative to clean, mean AUC")
        print(summ.pivot(index="condition", columns="window").round(3).to_string())

    plot(res, rd / "figures" / f"loop_rebuild_{tagname}.png", args.model, args.direction, T, L)
    print(f"\n-> analysis/loop_rebuild_{tagname}.csv, figures/loop_rebuild_{tagname}.png")


if __name__ == "__main__":
    main()
