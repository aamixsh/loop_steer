#!/usr/bin/env python3
"""Do later loops undo a small disturbance of the residual stream? Impulse response and loop contraction.

    scripts/perturb_response.py --gpu 0 --model ByteDance/Ouro-1.4B-Thinking
    scripts/perturb_response.py --gpu 0 --model Qwen/Qwen3-8B              # dense control: 4 blocks of 9 layers
    scripts/perturb_response.py --gpu 0 --model ByteDance/Ouro-1.4B-Thinking --n-loops 8 --tag loops8
    scripts/perturb_response.py ... --vec caa_sycophancy=caa:sycophancy:t3.l12   # extra directions to inject

Teacher-forced formatted prompts (``--n-prompts`` harmful + as many harmless train prompts, no CoT). Each condition
adds ``eps * |h| * v`` (per token, all positions) to the residual stream at one site, for a unit direction ``v``: the
model's refusal direction ``u``, ``--n-random`` random directions, or any ``--vec``. Sites: every unit (loop; for a dense
model one of ``--blocks`` equal blocks of layers) at fractions ``--fracs`` of its layers.

**Impulse response** (``impulse.parquet``): at every later site and at the final stream, over all tokens and over the
prompt-end (template) tokens, with ``d = h_perturbed - h_clean``:
  rel    mean |d| / |h_clean|              (divide by eps: the share of the disturbance that survives)
  along  mean (d . v) / |h_clean|          (divide by eps: the share still along v)
  cos    mean cos(d, v)
and at the final stream the KL(clean || perturbed) of the next-token distribution and the top-1 flip rate on the
prompt-end tokens.

**Loop contraction** (``contraction.csv``): finite-difference gain of one unit (boundary -> next boundary), relative
to the stream size, for random disturbances and after ``--power-iters`` steps of power iteration (the most amplified
disturbance); cosine of that disturbance with u. **Clean drift** (``drift.csv``): relative change of the clean stream
from one unit boundary to the next (does the state converge with more loops?).

``--kick-mask nosink`` leaves the attention sinks unkicked (the first token and any token with |h| above
``--sink-ratio`` x the sequence median; shares in ``sinks.csv``): a kick scaled by a sink's massive norm breaks the
sink, which is not the propagation we want to measure. Threshold sweep:
``--eps 0.02,0.05,...,2 --n-random 4 --kick-mask nosink --skip-contraction --tag sweep``.

Writes ``analysis/perturb/<tag>/``. HF backend, one GPU.
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-prompts", type=int, default=32, help="harmful and as many harmless train prompts")
    ap.add_argument("--max-len", type=int, default=160, help="skip prompts longer than this (tokens)")
    ap.add_argument("--eps", default="0.05,0.25,1.0")
    ap.add_argument("--fracs", default="0,0.25,0.5,0.75")
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--u", default=None, help="refusal direction file:method:site (default: per model)")
    ap.add_argument("--vec", action="append", default=[], help="name=file:method:site, extra directions")
    ap.add_argument("--blocks", type=int, default=4, help="virtual units of a dense model")
    ap.add_argument("--power-iters", type=int, default=8)
    ap.add_argument("--power-eps", type=float, default=0.02)
    ap.add_argument("--n-loops", type=int, default=None)
    ap.add_argument("--kick-mask", choices=["all", "nosink"], default="all",
                    help="nosink: do not kick the first token or tokens with a massive norm (attention sinks)")
    ap.add_argument("--sink-ratio", type=float, default=10.0, help="nosink: skip tokens with |h| > ratio * median |h|")
    ap.add_argument("--skip-contraction", action="store_true")
    ap.add_argument("--skip-impulse", action="store_true", help="only the loop contraction (and drift)")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from loop_steer.paths import run_dir, setup_job_env
    setup_job_env()
    import pandas as pd
    import torch

    from loop_steer.cot import format_prompt, load_prompts
    from loop_steer.models import eoi_length, load_model, load_tokenizer
    from loop_steer.selfcorr import DEFAULT_U, FINAL, load_dirfile, pad_batch, stream_hooks, structure

    rd = run_dir(args.model)
    out = rd / "analysis" / "perturb" / args.tag
    if args.skip_existing and (out / "impulse.parquet").exists():
        print(f"{out}/impulse.parquet exists; skipping")
        return
    out.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(args.model)
    n_eoi = eoi_length(tok)
    seqs = []
    for kind in ("harmful", "harmless"):
        n = 0
        for p in load_prompts("train", kind):
            ids = tok(format_prompt(tok, p), add_special_tokens=False).input_ids
            if len(ids) <= args.max_len:
                seqs.append(ids); n += 1
            if n == args.n_prompts:
                break
    ids, valid = pad_batch(seqs, tok.pad_token_id if tok.pad_token_id is not None else 0)
    eoi = torch.zeros_like(valid)
    for i, s in enumerate(seqs):
        eoi[i, len(s) - n_eoi: len(s)] = True

    model = load_model(args.model, n_loops_override=args.n_loops)
    st = structure(model, blocks=args.blocks)
    dev = model.device
    ids, valid, eoi = ids.to(dev), valid.to(dev), eoi.to(dev)
    D = model.config.hidden_size
    print(f"{args.model}: {len(seqs)} prompts, {st.T} {'loops' if st.looped else 'blocks'}, {len(st.sites)} sites",
          flush=True)

    # ---- directions
    dirs = {"u": load_dirfile(rd, args.u or DEFAULT_U[rd.name])}
    for spec in args.vec:
        name, ref = spec.split("=", 1)
        dirs[name] = load_dirfile(rd, ref)
    g = torch.Generator().manual_seed(0)
    for k in range(args.n_random):
        dirs[f"random{k}"] = torch.randn(D, generator=g)
    dirs = {k: (v / v.norm()).to(dev, torch.float32) for k, v in dirs.items()}
    u = dirs["u"]

    # ---- clean pass: every site, final stream, prompt-end log-probs
    clean = {}

    def rec(site, h):
        clean[site] = h.detach().clone()

    def rec_final(h):
        clean[FINAL] = h.detach().clone()

    with torch.inference_mode(), stream_hooks(model, rec, rec_final):
        logits = model(input_ids=ids, use_cache=False).logits
    lp_clean = torch.log_softmax(logits[eoi].float(), -1)
    del logits
    norms = {s: h.float().norm(dim=-1) for s, h in clean.items()}  # [B, S]

    # ---- which tokens a kick touches (per site): all, or all but sinks (first token, massive-norm tokens)
    kick_mask, sink_share = {}, []
    pos0 = torch.zeros_like(valid); pos0[:, 0] = True
    for s, n in norms.items():
        if args.kick_mask == "all":
            kick_mask[s] = torch.ones_like(valid)
            continue
        med = torch.stack([n[i][valid[i]].median() for i in range(n.shape[0])])[:, None]
        sink = pos0 | (n > args.sink_ratio * med)
        kick_mask[s] = ~sink
        sink_share.append({"site": str(s), "first_only": float(pos0[valid].float().mean()),
                           "excluded": float(sink[valid].float().mean()),
                           "massive_not_first": float((sink & ~pos0)[valid].float().mean()),
                           "max_ratio": float((n / med)[valid].max())})
    if sink_share:
        pd.DataFrame(sink_share).to_csv(out / "sinks.csv", index=False)

    # ---- clean drift between unit boundaries
    drift = []
    for k in range(st.T):
        a, b = st.boundaries[k], st.boundaries[k + 1]
        da = (clean[b].float() - clean[a].float()).norm(dim=-1) / norms[a]
        cosab = torch.nn.functional.cosine_similarity(clean[b].float(), clean[a].float(), dim=-1)
        drift.append({"unit": k + 1, "from": str(a), "to": str(b), "rel_change": float(da[valid].mean()),
                      "cos": float(cosab[valid].mean())})
    pd.DataFrame(drift).to_csv(out / "drift.csv", index=False)

    def stats(d, h_clean, v, mask):
        d, h_clean = d.float()[mask], h_clean.float()[mask]
        n = h_clean.norm(dim=-1)
        dn = d.norm(dim=-1)
        dv = d @ v
        return float((dn / n).mean()), float((dv / n).mean()), float((dv / dn.clamp_min(1e-12)).mean())

    # ---- impulse response
    rows = []
    eps_list = [float(e) for e in args.eps.split(",")]
    fracs = [float(f) for f in args.fracs.split(",")]
    inj_sites = sorted({st.site_at(k, f) for k in range(st.T) for f in fracs}, key=lambda s: st.order[s])
    for site in ([] if args.skip_impulse else inj_sites):
        for dname, v in dirs.items():
            for eps in eps_list:
                cur = []
                started = {"on": False}

                def inject(s, h, site=site, v=v, eps=eps):
                    if s != site:
                        return None
                    started["on"] = True
                    km = kick_mask[s][..., None]
                    return h + (eps * h.float().norm(dim=-1, keepdim=True) * v * km).to(h.dtype)

                def measure(s, h, v=v):
                    if not started["on"]:
                        return None
                    d = h - clean[s]
                    for wname, m in (("all", valid), ("eoi", eoi)):
                        r, a, c = stats(d, clean[s], v, m)
                        cur.append((s, wname, r, a, c))
                    return None

                last = {}

                def measure_final(h, v=v):  # Ouro unembeds after every loop: keep the last call
                    d = h - clean[FINAL]
                    last["rows"] = [(FINAL, wname, *stats(d, clean[FINAL], v, m))
                                    for wname, m in (("all", valid), ("eoi", eoi))]

                with torch.inference_mode(), stream_hooks(model, inject), stream_hooks(model, measure, measure_final):
                    logits = model(input_ids=ids, use_cache=False).logits
                cur += last["rows"]
                lp = torch.log_softmax(logits[eoi].float(), -1)
                kl = float((lp_clean.exp() * (lp_clean - lp)).sum(-1).mean())
                flip = float((lp.argmax(-1) != lp_clean.argmax(-1)).float().mean())
                del logits, lp
                for s, wname, r, a, c in cur:
                    rows.append({"inj_unit": st.unit[site], "inj_layer": site[1], "inj_depth": st.depth(site),
                                 "dir": dname, "eps": eps, "site": str(s), "unit": st.T if s == FINAL else st.unit[s],
                                 "layer": -1 if s == FINAL else s[1], "depth": st.depth(s), "window": wname,
                                 "rel": r, "along": a, "cos": c,
                                 "kl": kl if s == FINAL else float("nan"),
                                 "flip": flip if s == FINAL else float("nan")})
        print(f"  injected at {site} (unit {st.unit[site]}): {len(rows)} rows", flush=True)
    imp = pd.DataFrame(rows)
    if rows:
        imp.to_parquet(out / "impulse.parquet")

    # ---- loop contraction (power iteration on unit k: boundary k -> boundary k + 1)
    crow = []
    for k in ([] if args.skip_contraction else range(st.T)):
        a, b = st.boundaries[k], st.boundaries[k + 1]
        # nosink: perturb and measure only non-sink tokens (a sink's huge norm would dominate both norms)
        ma, mb = valid & kick_mask[a], valid & kick_mask[b]
        na = clean[a].float()[ma].norm()
        nb = clean[b].float()[mb].norm()
        for start in range(4):
            gen = torch.Generator(device=dev).manual_seed(100 + start)
            delta = torch.randn(clean[a].shape, generator=gen, device=dev) * ma[..., None]
            for it in range(args.power_iters + 1):
                delta = delta * ma[..., None]
                delta = delta / delta[ma].norm() * args.power_eps * na
                got = {}

                def inject(s, h, a=a, delta=delta):
                    return (h.float() + delta).to(h.dtype) if s == a else None

                def grab(s, h, b=b):
                    if s == b:
                        got["d"] = (h.float() - clean[b].float()) * mb[..., None]

                def grab_final(h, b=b):
                    if b == FINAL:
                        got["d"] = (h.float() - clean[FINAL].float()) * mb[..., None]

                with torch.inference_mode(), stream_hooks(model, inject), stream_hooks(model, grab, grab_final):
                    model(input_ids=ids, use_cache=False)
                d = got["d"]
                gain = float((d[mb].norm() / nb) / (delta[ma].norm() / na))
                if it in (0, args.power_iters):
                    tokdir = delta[eoi].mean(0)
                    crow.append({"unit": k + 1, "start": start, "iter": it, "gain": gain,
                                 "gain_per_layer": gain ** (1 / len(st.unit_layers[k])),
                                 "cos_u_eoi": float(tokdir @ u / tokdir.norm())})
                delta = d
        print(f"  contraction unit {k + 1}: gain random {[r['gain'] for r in crow if r['unit'] == k + 1 and r['iter'] == 0]}",
              flush=True)
    if crow:
        pd.DataFrame(crow).to_csv(out / "contraction.csv", index=False)

    if not rows:
        json.dump(vars(args), open(out / "args.json", "w"), indent=1)
        print(f"-> {out}")
        return

    # ---- summary: survival of a disturbance injected early in unit 1 vs in the last unit, at the final stream
    fin = imp[(imp.site == str(FINAL)) & (imp.window == "eoi")].copy()
    fin["survive"] = fin.rel / fin.eps
    fin["along_frac"] = fin.along / fin.eps
    fin["kind"] = fin["dir"].where(~fin["dir"].str.startswith("random"), "random")
    summ = fin.groupby(["kind", "eps", "inj_unit", "inj_layer"])[["survive", "along_frac", "kl", "flip"]].mean()
    with pd.option_context("display.width", 200, "display.max_rows", 400):
        print(summ.round(3).to_string())
    json.dump(vars(args), open(out / "args.json", "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
