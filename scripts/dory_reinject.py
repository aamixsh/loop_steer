#!/usr/bin/env python3
"""Does Dory rebuild refusal in later loops from the re-injected prelude output?

    scripts/dory_reinject.py --gpu 0 [--n-loops 4] [--max-per-class 40]

Dory starts every loop with ``h = justnorm(state + x)``, where ``x`` is the output of the 16 input sublayers (the
prelude), computed once and re-injected unchanged in every loop (Ouro and Nanbeige carry only the previous loop's
output). Ablating the refusal direction ``u`` (``hh_eoi`` at ``t0.l26``) in loops 1+2 of a 4-loop run leaves refusal
intact (0.91 vs 0.92 clean), and ``scripts/loop_rebuild.py`` shows that ``u`` is built inside each loop (sublayers
16-28) while the prelude carries almost none of it, but does carry refusal information in other directions.

This script teacher-forces the clean Dory traces of ``loop_rebuild.py`` (refusing vs complying CoTs, train split) under
hook ablation of ``u`` in some loops combined with a change to what is re-injected in other loops (the
``DoryBackbone.reinject_hook`` of ``loop_steer.dory``):

  drop        re-inject nothing: ``h = justnorm(state)``
  proj        re-inject ``x`` with ``d_x`` projected out (``d_x``: the prelude's own refusal direction, ``hh_eoi`` at
              the first recurrent sublayer's input of loop 1, which is ``x`` itself)
  rand        the same with a random unit direction (control)
  mean        re-inject ``x`` with its component along ``d_x`` set to its mean over every token of every trace (removes
              the refusing / complying difference without moving the stream off its usual mean, unlike ``proj``)
  rmean       the same along the random direction (control)
  smean       as ``mean`` but for the span of the prelude's own refusal directions (``hh_eoi`` at sublayers 1-16,
              orthonormalized): every component in that subspace set to its mean
  rsmean      the same for a random subspace of the same dimension (control)
  umean       as ``mean`` but along ``u`` itself (the ablated direction): ``x`` keeps a small refusing / complying
              difference along ``u`` over the CoT tokens (AUC 0.87, none at ``prompt_end``)
  uxmean      both: the components in span(``u``, ``d_x``) set to their means
  flat        re-inject the mean prelude output over every token of every trace (unit norm) at every position: keeps
              the stream's usual scale, removes everything token-specific in ``x``

Conditions are written ``abl<loops>`` / ``drop<loops>`` / ``proj<loops>`` / ``rand<loops>`` joined by ``+`` (loops
1-based, e.g. ``abl12+proj34``). Per condition it records, over the end-of-instruction window (``prompt_end``) and the
CoT window: the refusing - complying separation and AUC of the projection on ``u`` and on each site's own direction at
every sublayer input (loops beyond the 2 trained ones use loop 2's own directions); at each loop's start the separation
of ``x`` (as re-injected) and of the carried state along ``u`` and ``d_x``; and the mean next-token NLL of the CoT
tokens (does the change break the model?). Writes ``analysis/dory_reinject.csv`` (sublayer rows: ``kind=layer``; loop
start rows: ``kind=entry``; NLL: ``kind=nll``). HF backend, one GPU.
"""

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
DEFAULT = ("clean;abl12;abl123;abl1234;drop34;proj34;rand34;abl12+drop34;abl12+proj34;abl12+rand34;"
           "abl12+proj1234;proj1234;mean34;rmean34;abl12+mean34;abl12+rmean34;abl12+mean1234;flat34;abl12+flat34")


def parse_condition(name):
    """'abl12+proj34' -> {'abl': [0, 1], 'proj': [2, 3]} (0-based loops)."""
    out = {}
    if name == "clean":
        return out
    for part in name.split("+"):
        m = re.fullmatch(r"(abl|drop|proj|rand|smean|rsmean|umean|uxmean|mean|rmean|flat)(\d+)", part)
        if not m:
            raise ValueError(f"bad condition part {part!r}")
        out[m.group(1)] = [int(c) - 1 for c in m.group(2)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--n-loops", type=int, default=4)
    ap.add_argument("--direction", default="hh_eoi:t0.l26")
    ap.add_argument("--dx-site", default="t0.l16", help="site of d_x (its input in loop 1 is the prelude output)")
    ap.add_argument("--conditions", default=DEFAULT, help="semicolon-separated")
    ap.add_argument("--max-per-class", type=int, default=40)
    ap.add_argument("--max-cot-tokens", type=int, default=256)
    ap.add_argument("--gen-path", default="train_clean.scored.parquet")
    ap.add_argument("--tag", default="dory_reinject")
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from loop_steer.paths import run_dir, setup_job_env
    setup_job_env()
    import numpy as np
    import pandas as pd
    import torch

    from loop_rebuild import auc, parse_site
    from loop_steer.cot import format_prompt
    from loop_steer.hooks import _layer_input, _loop, ablation_hooks, decoder_layers, hooked
    from loop_steer.models import eoi_length, load_model, load_tokenizer

    rd = run_dir(MODEL)
    dirs = torch.load(rd / "directions" / "train.pt", weights_only=False)
    method, site = args.direction.split(":")
    sites = dirs["sites"]
    site_idx = {s: i for i, s in enumerate(sites)}
    own = dirs[method]["dirs"].float()
    own = own / own.norm(dim=-1, keepdim=True)
    u = own[site_idx[parse_site(site)]]
    dx = own[site_idx[parse_site(args.dx_site)]]
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(u.shape[0], generator=g)
    rnd = rnd / rnd.norm()
    trained = max(t for t, _ in sites) + 1
    S = torch.linalg.qr(torch.stack([own[site_idx[(0, l)]] for l in range(1, 17)]).T)[0].T  # [17, D] prelude span
    RS = torch.linalg.qr(torch.randn(u.shape[0], S.shape[0], generator=g))[0].T  # random subspace, same dim  # loops with their own directions (2)

    tok = load_tokenizer(MODEL)
    df = pd.read_parquet(rd / "generations" / args.gen_path).dropna(subset=["sr_score"])
    cots = df.groupby(["prompt_idx", "cot_rep"]).agg(prompt=("prompt", "first"), cot=("cot", "first"),
                                                     lo=("sr_score", "max"), hi=("sr_score", "min")).reset_index()
    ref = cots[cots.lo < 0.2].head(args.max_per_class)
    com = cots[cots.hi > 0.8].head(args.max_per_class)
    n_eoi = eoi_length(tok)
    seqs = []
    for rows, label in ((ref, 1), (com, 0)):
        for r in rows.itertuples():
            p = tok(format_prompt(tok, r.prompt), add_special_tokens=False).input_ids
            c = tok.encode(r.cot, add_special_tokens=False)[: args.max_cot_tokens]
            seqs.append((torch.tensor([p + c]), label, slice(len(p) - n_eoi, len(p)), slice(len(p), len(p) + len(c))))
    labels = np.array([s[1] for s in seqs])
    print(f"{int(labels.sum())} refusing, {int((1 - labels).sum())} complying CoTs; u = {args.direction}, "
          f"d_x = {method}:{args.dx_site}, cos(u, d_x) = {float(u @ dx):.3f}", flush=True)

    model = load_model(MODEL, n_loops_override=args.n_loops)
    backbone = model.model
    T = backbone.num_loops
    dev = model.device
    W = torch.stack([u, dx]).to(dev)  # entry projections
    Ud, dxd, rndd = u.to(dev), dx.to(dev), rnd.to(dev)
    n_in = model.config.n_input_layers

    def own_row(t, layer):
        key = (t, layer) if (t, layer) in site_idx else (min(t, trained - 1), layer)
        return site_idx.get(key)

    # mean component of x along d_x and the random direction over every token of every trace (for mean / rmean)
    tot = {"dx": 0.0, "rnd": 0.0, "n": 0, "x": 0.0}
    Sd, RSd = S.to(dev), RS.to(dev)

    def record(t, x, state):
        if t == 0:
            xf = x[0].float()
            tot["dx"] += float((xf @ dxd).sum()); tot["rnd"] += float((xf @ rndd).sum()); tot["n"] += xf.shape[0]
            tot["x"] = tot["x"] + xf.sum(0)
        return x

    backbone.reinject_hook = record
    try:
        with torch.inference_mode():
            for ids, *_ in seqs:
                model.model(input_ids=ids.to(dev))
    finally:
        backbone.reinject_hook = None
    mu_dx, mu_rnd = tot["dx"] / tot["n"], tot["rnd"] / tot["n"]
    x_flat = tot["x"] / tot["x"].norm()
    mu_S, mu_RS = Sd @ (tot["x"] / tot["n"]), RSd @ (tot["x"] / tot["n"])  # mean components in each subspace
    UXd = torch.linalg.qr(torch.stack([Ud, dxd]).T.float())[0].T  # [2, D] orthonormal span(u, d_x)
    mu_u, mu_UX = float(Ud @ (tot["x"] / tot["n"])), UXd @ (tot["x"] / tot["n"])
    print(f"mean component of x: d_x {mu_dx:.4f}, u {mu_u:.4f}, random {mu_rnd:.4f}", flush=True)

    def set_comp(x, d, target):
        return x + ((target - x.float() @ d).unsqueeze(-1) * d).to(x)

    def set_span(x, B, target):  # B: [k, D] orthonormal rows; components in span(B) -> target [k]
        return x + ((target - x.float() @ B.T) @ B).to(x)

    rows = []
    for name in args.conditions.split(";"):
        spec = parse_condition(name)
        layer_store, entry_store, nll = {}, {}, []
        cur = {}

        def reinject(t, x, state):
            xt = x
            if t in spec.get("drop", []):
                xt = torch.zeros_like(x)
            elif t in spec.get("proj", []):
                xt = x - (x.float() @ dxd).unsqueeze(-1).to(x) * dxd.to(x)
            elif t in spec.get("rand", []):
                xt = x - (x.float() @ rndd).unsqueeze(-1).to(x) * rndd.to(x)
            elif t in spec.get("mean", []):
                xt = set_comp(x, dxd, mu_dx)
            elif t in spec.get("umean", []):
                xt = set_comp(x, Ud, mu_u)
            elif t in spec.get("uxmean", []):
                xt = set_span(x, UXd, mu_UX)
            elif t in spec.get("rmean", []):
                xt = set_comp(x, rndd, mu_rnd)
            elif t in spec.get("smean", []):
                xt = set_span(x, Sd, mu_S)
            elif t in spec.get("rsmean", []):
                xt = set_span(x, RSd, mu_RS)
            elif t in spec.get("flat", []):
                xt = x_flat.to(x).expand_as(x)
            for what, v in (("x", xt), ("state", state)):
                proj = v[0].float() @ W.T  # [S, 2]
                entry_store.setdefault((t, what), []).append(
                    torch.stack([proj[cur["w"][0]].mean(0), proj[cur["w"][1]].mean(0)]).cpu())
            return xt

        def capture(layer):
            def hook(module, a, kw):
                t = _loop(kw) if layer >= n_in and layer < n_in + model.config.n_recurrent_layers else 0
                i = own_row(t, layer)
                if i is None:
                    return
                x = _layer_input(a, kw)[0].float()
                proj = x @ torch.stack([Ud, own[i].to(dev)]).T
                layer_store.setdefault((t, layer), []).append(
                    torch.stack([proj[cur["w"][0]].mean(0), proj[cur["w"][1]].mean(0)]).cpu())
            return hook

        abl = ablation_hooks(model, u, loops=spec["abl"]) if "abl" in spec else (lambda: [])
        caps = lambda: [b.register_forward_pre_hook(capture(l), with_kwargs=True)  # noqa: E731
                        for l, b in enumerate(decoder_layers(model))]
        backbone.reinject_hook = reinject
        try:
            with torch.inference_mode(), hooked(abl), hooked(caps):
                for ids, _, w_p, w_c in seqs:
                    cur["w"] = (w_p, w_c)
                    ids = ids.to(dev)
                    logits = model(input_ids=ids, use_cache=False).logits[0]
                    lp = torch.log_softmax(logits[w_c.start - 1: w_c.stop - 1].float(), -1)
                    nll.append(float(-lp.gather(-1, ids[0, w_c][:, None]).mean()))
        finally:
            backbone.reinject_hook = None

        def emit(store, kind, dnames):
            for (t, key), vals in store.items():
                v = torch.stack(vals)  # [n_seq, 2 windows, 2 dirs]
                for wi, window in enumerate(("prompt_end", "cot")):
                    for di, dname in enumerate(dnames):
                        x = v[:, wi, di]
                        pos, neg = x[labels == 1], x[labels == 0]
                        rows.append({"condition": name, "kind": kind, "loop": t + 1,
                                     "layer": key if kind == "layer" else -1, "what": "" if kind == "layer" else key,
                                     "window": window, "dir": dname, "sep": float(pos.mean() - neg.mean()),
                                     "auc": auc(pos, neg), "mean_ref": float(pos.mean()), "mean_com": float(neg.mean())})

        emit(layer_store, "layer", ("u", "own"))
        emit(entry_store, "entry", ("u", "dx"))
        nll = np.array(nll)
        rows.append({"condition": name, "kind": "nll", "loop": 0, "layer": -1, "what": "", "window": "cot", "dir": "",
                     "sep": float(nll[labels == 1].mean() - nll[labels == 0].mean()), "auc": float("nan"),
                     "mean_ref": float(nll[labels == 1].mean()), "mean_com": float(nll[labels == 0].mean())})
        print(f"  {name:16s} done; CoT NLL {nll.mean():.3f}", flush=True)

    res = pd.DataFrame(rows)
    out = rd / "analysis" / f"{args.tag}.csv"
    res.to_csv(out, index=False)

    # summary: separation along u at sublayer 26 of each loop (prompt_end), entry decomposition, NLL
    L26 = parse_site(site)[1]
    s = res[(res.kind == "layer") & (res.layer == L26) & (res.dir == "u") & (res.window == "prompt_end")]
    e = res[(res.kind == "entry") & (res.window == "prompt_end")]
    n = res[res.kind == "nll"].set_index("condition")
    order = list(dict.fromkeys(res.condition))
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(f"\nseparation along u at sublayer {L26} (prompt_end), per loop")
        print(s.pivot(index="condition", columns="loop", values="sep").reindex(order).round(4).to_string())
        print("\nAUC along u at the same place")
        print(s.pivot(index="condition", columns="loop", values="auc").reindex(order).round(3).to_string())
        print("\nloop start: separation of the re-injected x and of the carried state along u / d_x (prompt_end)")
        print(e.pivot_table(index="condition", columns=["what", "dir", "loop"], values="sep").reindex(order)
              .round(4).to_string())
        print("\nCoT NLL (refusing / complying)")
        print(n.loc[order, ["mean_ref", "mean_com"]].round(3).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
