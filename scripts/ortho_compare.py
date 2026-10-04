#!/usr/bin/env python3
"""Why does weight orthogonalization work on Nanbeige but break Ouro?  Same measurements on both.

Runs an HF looped model on real prompt+CoT token sequences (its own clean generations) and,
for a refusal direction u and a same-site random direction, compares conditions

  clean        no intervention
  hook         directional ablation with hooks (reference: exact removal)
  ortho        weight orthogonalization: u projected out of embed / o_proj / down_proj
  ortho_na     norm-aware variant (Ouro only): each writer orthogonalized against g*u, g = gain of the
               sandwich norm applied to its output, so the normed write is exactly orthogonal to u

and reports, per loop:
  resid    rms / signed-mean projection of the residual stream (layer inputs) onto u
  mat      rms projection of the raw matrix outputs (o_proj, down_proj) onto u   -> ~0 under any ortho
  write    rms projection of what is actually added to the residual onto u       -> Ouro: after the norm
  Lin/Lout projection entering / leaving the shared norm that runs after the loop
  nll      teacher-forced NLL of the clean CoT tokens from the state after this loop (logit lens;
           the last loop is the model's real output) -- a damage measure, exact for the random direction
  top1     agreement of that loop's argmax with the clean model's final-loop argmax

Also: gain statistics of every norm that sits between a writer and the residual (how far g*u
rotates away from u), optional reduced-loop-count runs (``--loops``), and short greedy generations.
"""

import argparse
import json
import os
from collections import defaultdict


def parse_site(site):
    loop, layer = site.replace("t", "").replace("l", "").split(".")
    return int(loop), int(layer)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--direction", required=True, help="method:site key into directions/train.pt, e.g. v4_baseline:t3.l16")
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--max-cot-tokens", type=int, default=300)
    ap.add_argument("--loops", default="", help="comma-separated reduced loop counts to also test, e.g. 1,2")
    ap.add_argument("--gen-tokens", type=int, default=60)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from loop_steer.paths import setup_job_env
    setup_job_env()
    import pandas as pd
    import torch
    import torch.nn.functional as F

    from loop_steer.cot import format_prompt
    from loop_steer.hooks import _loop, ablation_hooks, decoder_layers, hooked
    from loop_steer.models import is_ouro, load_model, load_tokenizer, n_loops
    from loop_steer.ortho import loop_span, orthogonalize_
    from loop_steer.paths import run_dir

    rd = run_dir(args.model)
    dirs = torch.load(rd / "directions" / "train.pt", weights_only=False)
    method, site = args.direction.split(":")
    idx = dirs["sites"].index(parse_site(site))
    u_ref = dirs[method]["dirs"][idx].float()
    u_ref = u_ref / u_ref.norm()
    u_rnd = dirs["random"]["dirs"][idx].float()
    u_rnd = u_rnd / u_rnd.norm()
    directions = {"refusal": u_ref, "random": u_rnd}

    tok = load_tokenizer(args.model)
    ouro = is_ouro(args.model)

    # ---- sequences: this model's own clean prompt+CoT traces
    gen_path = (rd / "generations" / "select_clean.scored.parquet") if ouro \
        else (rd / "generations" / "clean_select" / "none.scored.parquet")
    gen = pd.read_parquet(gen_path).drop_duplicates(["prompt_idx", "cot_rep"]).head(args.n_seqs)
    seqs, prompts = [], []
    for row in gen.itertuples():
        p_ids = tok(format_prompt(tok, row.prompt), add_special_tokens=False).input_ids
        c_ids = tok.encode(row.cot, add_special_tokens=False)[: args.max_cot_tokens]
        seqs.append(torch.tensor([list(p_ids) + c_ids]))
        prompts.append(row.prompt)
    n_tokens = sum(s.shape[1] for s in seqs)
    print(f"{args.model}: direction {args.direction}; {len(seqs)} sequences, {n_tokens} tokens", flush=True)

    result = {"model": args.model, "direction": args.direction, "n_seqs": len(seqs), "n_tokens": n_tokens}

    # ---- Part 1: norm gains between writers and the residual
    def gain_report(model):
        layers = decoder_layers(model)
        norms = {"inter_loop": [model.model.norm.weight]}
        if ouro:
            norms["sandwich_attn"] = [l.input_layernorm_2.weight for l in layers]
            norms["sandwich_mlp"] = [l.post_attention_layernorm_2.weight for l in layers]
        out = {}
        for name, ws in norms.items():
            g_all = torch.cat([w.detach().float().cpu().flatten() for w in ws])
            ga = g_all.abs()
            thr = 5 * ga.median()
            rep = {"n_norms": len(ws), "median_abs": float(ga.median()), "p99_abs": float(ga.quantile(0.99)),
                   "max_abs": float(ga.max()), "frac_dims_gt_5x_median": float((ga > thr).float().mean())}
            for dname, u in directions.items():
                cos, amp, mass = [], [], []
                for w in ws:
                    g = w.detach().float().cpu()
                    gu = g * u
                    cos.append(float(torch.dot(u, gu) / gu.norm()))
                    amp.append(float(gu.norm() / g.pow(2).mean().sqrt()))  # |g*u| relative to rms gain
                    mass.append(float((u[g.abs() > 5 * g.abs().median()] ** 2).sum()))
                rep[dname] = {"cos_u_gu_mean": sum(cos) / len(cos), "cos_u_gu_min": min(cos),
                              "amp_mean": sum(amp) / len(amp), "u_mass_on_outlier_dims_mean": sum(mass) / len(mass)}
            out[name] = rep
        return out

    # ---- Part 2: per-loop tracking under each condition
    def run_tracking(model, u, T, final_argmax_ref=None):
        """Return per-loop stats dict and the final-loop argmax per sequence (for top1 agreement)."""
        layers = decoder_layers(model)
        dev = model.device
        u_dev = u.to(dev)
        acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # (stage, loop) -> [sum proj, sum proj^2, sum |x|^2, n]
        state = {"loop": 0, "final_calls": 0}
        lens = defaultdict(lambda: [0.0, 0, 0])  # loop -> [sum nll, n agree, n]
        cur = {"ids": None, "argmax": {}}

        def rec(stage, loop, x):
            x = x.detach().float().reshape(-1, x.shape[-1])
            p = x @ u_dev
            a = acc[(stage, loop)]
            a[0] += float(p.sum()); a[1] += float((p ** 2).sum()); a[2] += float((x ** 2).sum()); a[3] += x.shape[0]

        def layer_pre(module, a_, kw):
            state["loop"] = _loop(kw)
            rec("resid", state["loop"], a_[0] if a_ else kw["hidden_states"])

        def mat_hook(module, inputs, output):  # raw matrix output (Ouro: input of the sandwich norm)
            rec("mat", state["loop"], inputs[0] if ouro else (output[0] if isinstance(output, tuple) else output))

        def write_hook(module, inputs, output):  # what gets added to the residual
            rec("write", state["loop"], output[0] if isinstance(output, tuple) else output)

        def final_hook(module, inputs, output):
            loop = state["final_calls"] % T
            state["final_calls"] += 1
            rec("Lin", loop, inputs[0]); rec("Lout", loop, output)
            # logit lens from the state after this loop
            logits = model.lm_head(output).float()[0]  # [S, V]
            ids = cur["ids"][0]
            nll = F.cross_entropy(logits[:-1], ids[1:], reduction="sum")
            am = logits.argmax(-1).cpu()
            l = lens[loop]
            l[0] += float(nll); l[2] += ids.shape[0] - 1
            if final_argmax_ref is not None:
                l[1] += int((am[:-1] == final_argmax_ref[cur["i"]][:-1]).sum())
            cur["argmax"][loop] = am

        def install():
            hs = [l.register_forward_pre_hook(layer_pre, with_kwargs=True) for l in layers]
            for l in layers:
                if ouro:
                    hs += [l.input_layernorm_2.register_forward_hook(mat_hook),
                           l.post_attention_layernorm_2.register_forward_hook(mat_hook),
                           l.input_layernorm_2.register_forward_hook(write_hook),
                           l.post_attention_layernorm_2.register_forward_hook(write_hook)]
                else:
                    hs += [l.self_attn.register_forward_hook(mat_hook), l.mlp.register_forward_hook(mat_hook),
                           l.self_attn.register_forward_hook(write_hook), l.mlp.register_forward_hook(write_hook)]
            hs.append(model.model.norm.register_forward_hook(final_hook))
            return hs

        final_argmax = []
        with torch.inference_mode(), hooked(install):
            for i, ids in enumerate(seqs):
                cur["ids"], cur["i"] = ids.to(dev), i
                state["final_calls"] = 0
                model(input_ids=cur["ids"], use_cache=False)
                final_argmax.append(cur["argmax"][T - 1])
        rows = {}
        for loop in range(T):
            r = {}
            for stage in ("resid", "mat", "write", "Lin", "Lout"):
                s, s2, x2, n = acc[(stage, loop)]
                r[f"{stage}_rms"] = (s2 / n) ** 0.5
                r[f"{stage}_mean"] = s / n
                r[f"{stage}_len"] = (x2 / n) ** 0.5
            nll, agree, n = lens[loop]
            r["nll"] = nll / n
            r["top1"] = agree / n if final_argmax_ref is not None else None
            rows[loop] = r
        return rows, final_argmax

    def fmt_rows(rows, T):
        cols = ["resid_rms", "resid_mean", "mat_rms", "write_rms", "Lin_rms", "Lout_rms", "Lout_len", "nll", "top1"]
        head = "  loop " + " ".join(f"{c:>10s}" for c in cols)
        lines = [head]
        for loop in range(T):
            r = rows[loop]
            lines.append(f"  {loop:4d} " + " ".join(
                f"{'-' if r[c] is None else r[c]:>10.3f}" if isinstance(r[c], float) else f"{'-':>10s}" for c in cols))
        return "\n".join(lines)

    def full_run(T_override=None, label="full"):
        model = load_model(args.model, n_loops_override=T_override)
        T = n_loops(model.config)
        print(f"\n######## {label}: {T} loop(s)", flush=True)
        out = {"T": T}
        if T_override is None:
            out["gains"] = gain_report(model)
        clean_rows, clean_argmax = run_tracking(model, u_ref, T)
        # top1 agreement of clean intermediate loops with the clean final loop
        clean_rows, _ = run_tracking(model, u_ref, T, final_argmax_ref=clean_argmax)
        out["clean"] = {"refusal": clean_rows}
        print("\n[clean] (projections onto the refusal direction)")
        print(fmt_rows(clean_rows, T))
        # consecutive-loop state similarity (clean): cosine of Lout between loops t and t+1
        for dname, u in directions.items():
            out.setdefault("clean", {})[dname] = clean_rows if dname == "refusal" else run_tracking(model, u, T, clean_argmax)[0]
            # hook
            with hooked(ablation_hooks(model, u)):
                rows, _ = run_tracking(model, u, T, clean_argmax)
            out.setdefault("hook", {})[dname] = rows
            print(f"\n[hook  / {dname}]"); print(fmt_rows(rows, T))
            # plain ortho
            orthogonalize_(model, u)
            rows, _ = run_tracking(model, u, T, clean_argmax)
            out.setdefault("ortho", {})[dname] = rows
            print(f"\n[ortho / {dname}]"); print(fmt_rows(rows, T))
            if ouro:
                orthogonalize_(model, u, norm_aware=True)
                rows, _ = run_tracking(model, u, T, clean_argmax)
                out.setdefault("ortho_na", {})[dname] = rows
                print(f"\n[ortho_na / {dname}]"); print(fmt_rows(rows, T))
            # span edit: also remove g*u, g^2*u, ... so the inter-loop norm cannot re-introduce u
            if T > 1:
                orthogonalize_(model, loop_span(model, u, T), norm_aware=ouro)
                rows, _ = run_tracking(model, u, T, clean_argmax)
                out.setdefault("ortho_span", {})[dname] = rows
                print(f"\n[ortho_span (rank {T}) / {dname}]"); print(fmt_rows(rows, T))
            orthogonalize_(model, None)
        # generations
        if args.gen_tokens > 0:
            gens = {}
            text = format_prompt(tok, prompts[0])
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)

            def greedy():
                with torch.inference_mode():
                    o = model.generate(**enc, max_new_tokens=args.gen_tokens, do_sample=False,
                                       pad_token_id=tok.pad_token_id or tok.eos_token_id)
                return tok.decode(o[0, enc.input_ids.shape[1]:], skip_special_tokens=False)

            gens["clean"] = greedy()
            for dname, u in directions.items():
                with hooked(ablation_hooks(model, u)):
                    gens[f"hook/{dname}"] = greedy()
                orthogonalize_(model, u); gens[f"ortho/{dname}"] = greedy()
                if ouro:
                    orthogonalize_(model, u, norm_aware=True); gens[f"ortho_na/{dname}"] = greedy()
                if T > 1:
                    orthogonalize_(model, loop_span(model, u, T), norm_aware=ouro); gens[f"ortho_span/{dname}"] = greedy()
                orthogonalize_(model, None)
            out["generations"] = {"prompt": prompts[0], **gens}
            print(f"\n[generations] prompt: {prompts[0][:100]!r}")
            for k, v in gens.items():
                print(f"  {k:16s} {v[:200]!r}")
        del model
        torch.cuda.empty_cache()
        return out

    result["full"] = full_run()
    print("\n[gains]"); print(json.dumps(result["full"]["gains"], indent=1))
    for T in [int(x) for x in args.loops.split(",") if x]:
        result[f"loops{T}"] = full_run(T_override=T, label=f"reduced to {T} loop(s)")

    out = rd / "analysis" / f"ortho_compare_{method}_{site.replace('.', '')}{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    json.dump(result, open(out, "w"), indent=1, default=str)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
