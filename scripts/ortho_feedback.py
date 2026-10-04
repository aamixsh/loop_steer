#!/usr/bin/env python3
"""Ouro: how learned norm gains and cross-loop feedback treat a removed direction.

Part 1 (gains): for each of the 48 sublayer norms, how much does the learned
per-dimension gain g rotate a unit direction u?  cos(u, g*u) = 1 for a uniform gain.
Part 2 (feedback): at every layer call, the signed write into the residual stream
along u (attention + MLP outputs after their norms) is compared with the signed
residual component along u that the layer sees.  A negative slope means the model
pushes back against whatever is already along u (a restoring force); ~0 or
positive means nothing opposes the build-up.
Conditions: clean and weight-orthogonalized.  Output JSON -> analysis/ortho_feedback_*.json
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--direction", default="v4_baseline:t3.l16")
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--max-cot-tokens", type=int, default=300)
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from loop_steer.paths import setup_job_env
    setup_job_env()
    import numpy as np
    import pandas as pd
    import torch

    from loop_steer import MODEL_ID
    from loop_steer.hooks import decoder_layers, hooked
    from loop_steer.models import load_model, load_tokenizer
    from loop_steer.ortho import orthogonalize_
    from loop_steer.paths import run_dir

    rd = run_dir(MODEL_ID)
    dirs = torch.load(rd / "directions" / "train.pt", weights_only=False)
    method, site = args.direction.split(":")
    s_loop, s_layer = (int(x) for x in site.replace("t", "").replace("l", "").split("."))
    idx = dirs["sites"].index((s_loop, s_layer))
    u = dirs[method]["dirs"][idx].float()
    u = u / u.norm()
    u_rand = dirs["random"]["dirs"][idx].float()
    u_rand = u_rand / u_rand.norm()

    tok = load_tokenizer(MODEL_ID)
    model = load_model(MODEL_ID)
    layers = decoder_layers(model)
    dev = model.device
    result = {"direction": args.direction}

    # Part 1: gains.
    def gain_stats(direction):
        cos = []
        for layer in layers:
            for norm in (layer.input_layernorm_2, layer.post_attention_layernorm_2):
                g = norm.weight.detach().float().cpu()
                gu = g * direction
                cos.append(float(torch.dot(direction, gu) / (direction.norm() * gu.norm())))
        return {"mean_cos_u_gu": float(np.mean(cos)), "min_cos_u_gu": float(np.min(cos))}

    all_g = torch.cat([n.weight.detach().float().cpu().flatten()
                       for layer in layers for n in (layer.input_layernorm_2, layer.post_attention_layernorm_2)])
    g_abs = all_g.abs()
    result["gain"] = {
        "mean_abs": float(g_abs.mean()), "median_abs": float(g_abs.median()),
        "p99_abs": float(g_abs.quantile(0.99)), "max_abs": float(g_abs.max()),
        "frac_dims_above_5x_median": float((g_abs > 5 * g_abs.median()).float().mean()),
        "refusal_u": gain_stats(u), "random_u": gain_stats(u_rand),
    }

    # Part 2: feedback.
    gen = pd.read_parquet(rd / "generations" / "select_clean.scored.parquet")
    gen = gen.drop_duplicates(["prompt_idx", "cot_rep"]).head(args.n_seqs)
    seqs = []
    for row in gen.itertuples():
        p_ids = tok.apply_chat_template([{"role": "user", "content": row.prompt}], add_generation_prompt=True,
                                        tokenize=True)
        p_ids = p_ids["input_ids"] if hasattr(p_ids, "keys") else p_ids
        seqs.append(torch.tensor([list(p_ids) + tok.encode(row.cot, add_special_tokens=False)[: args.max_cot_tokens]]))

    u_dev = u.to(dev)
    cur = {"loop": 0, "r": None, "a": None}
    rec = {}  # (loop) -> lists of (r, w) arrays

    def layer_pre(module, a_, kw):
        cur["loop"] = kw.get("current_ut", 0)
        x = a_[0] if a_ else kw["hidden_states"]
        cur["r"] = (x.detach().float() @ u_dev).flatten().cpu()
        cur["a"] = None

    def attn_out(module, i, o):
        cur["a"] = (o.detach().float() @ u_dev).flatten().cpu()

    def mlp_out(module, i, o):
        w = cur["a"] + (o.detach().float() @ u_dev).flatten().cpu()
        rec.setdefault(cur["loop"], []).append((cur["r"].numpy(), w.numpy()))

    def install():
        hs = [l.register_forward_pre_hook(layer_pre, with_kwargs=True) for l in layers]
        hs += [l.input_layernorm_2.register_forward_hook(attn_out) for l in layers]
        hs += [l.post_attention_layernorm_2.register_forward_hook(mlp_out) for l in layers]
        return hs

    def run():
        rec.clear()
        with torch.inference_mode(), hooked(install):
            for ids in seqs:
                model(input_ids=ids.to(dev), use_cache=False)
        out = {}
        for loop, items in sorted(rec.items()):
            r = np.concatenate([i[0] for i in items])
            w = np.concatenate([i[1] for i in items])
            slope = float(np.cov(w, r)[0, 1] / np.var(r))
            out[loop] = {"mean_r": float(r.mean()), "rms_r": float((r ** 2).mean() ** 0.5),
                         "mean_write": float(w.mean()), "rms_write": float((w ** 2).mean() ** 0.5),
                         "slope_write_on_resid": slope, "corr": float(np.corrcoef(w, r)[0, 1]),
                         # a subsample for plotting
                         "scatter": [[float(a), float(b)] for a, b in zip(r[::97][:250], w[::97][:250])]}
        return out

    result["clean"] = run()
    orthogonalize_(model, u)
    result["ortho"] = run()
    orthogonalize_(model, None)

    out_path = rd / "analysis" / "ortho_feedback.json"
    json.dump(result, open(out_path, "w"))
    print(json.dumps(result["gain"], indent=1))
    for cond in ("clean", "ortho"):
        print(f"\n[{cond}] per loop: rms resid-along-u | signed mean resid | slope of write on resid | corr")
        for loop, d in result[cond].items():
            print(f"  loop {loop}: rms_r={d['rms_r']:.3f}  mean_r={d['mean_r']:+.3f}  "
                  f"slope={d['slope_write_on_resid']:+.4f}  corr={d['corr']:+.3f}  mean_write={d['mean_write']:+.4f}")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
