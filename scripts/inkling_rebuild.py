#!/usr/bin/env python3
"""Inkling: where does pass 3 rebuild the refusal signal from, after it is ablated in passes 1-2?

    scripts/inkling_rebuild.py --gpu 0 [--model thinkingmachines/Inkling-2-Mini] [--direction v4_cot:t2.l11]

Every Inkling core pass restarts from the prelude's state and reads the previous pass only through gated cross-pass
messages. With the refusal direction ``u`` ablated in passes 1-2, pass 3 can rebuild it from two channels: the prelude
state it restarts from, and the messages from pass 2. This script teacher-forces the model's own clean train traces
(refusing: all 3 answers score < 0.2; complying: all > 0.8; formatted prompt + first ``--max-cot-tokens`` CoT tokens)
through vLLM (``loop_steer.vllm_inkling``, NVFP4 as served) under:

  clean               no intervention
  abl12               u ablated in passes 1-2 (every input and write of those passes)
  abl12+pre           ... and in the prelude (the state every pass restarts from)
  abl12+msgabl        ... u projected out of the messages into pass 3
  abl12+msgzero       ... the messages into pass 3 zeroed
  abl12+pre+msgabl    both channels cleaned of u
  abl12+pre+msgzero   prelude cleaned of u, messages into pass 3 zeroed
  msgzero             messages into pass 3 zeroed, nothing ablated (how much zeroing alone disturbs the model)
  abl123              u ablated in all three passes (reference: the full ablation's effect inside the core)

Re-injection controls (as ``scripts/dory_reinject.py``; Inkling's analogue of Dory's re-injected input is the
prelude state, embedding and prelude block, that every core pass restarts from and the coda reads): ``<mode><passes>``
edits that state for passes ``passes`` (a run to the last pass, e.g. ``3`` or ``45``), restored before the coda unless
the part ends in ``c``:

  drop     zero it                         proj    project ``d_x`` out (``d_x``: ``--dx``, the prelude's own refusal
  rand     ``proj`` along a random unit         direction, ``hh_eoi`` at the prelude block t0.l6)
           direction (control)             mean    set its component along ``d_x`` to its mean over every token of
  rmean    ``mean``, random direction              every trace (clean run)
  smean    ``mean`` for the span of the prelude's own refusal directions (``hh_eoi`` at t0.l1-t0.l6, orthonormalized)
  rsmean   ``smean`` for a random subspace of the same dimension
  umean    ``mean`` along ``u``              uxmean  ``mean`` in span(u, d_x)
  flat     replace it with its mean vector over every token (keeps the scale, removes everything token-specific)

Conditions join parts with ``+``: ``abl<passes>`` (u ablated in those passes), ``pre`` (and in the prelude), ``msgabl``
/ ``msgzero`` (the messages into ``--msg-pass``), and the re-injection parts, e.g. ``abl12+uxmean3``, ``abl123+mean45``.

and records, at every site's resid_pre, each sequence's mean projection on unit ``u`` over the end-of-instruction
token (``eoi``) and the CoT tokens (``cot``): the refusing - complying **separation** and its AUC; and the mean
next-token NLL of the CoT tokens (does a condition break the model?). Writes analysis/inkling_rebuild[_<tag>].csv.
"""

import argparse
import functools
import json
import os


def auc(pos, neg):
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).float().mean() + 0.5 * (diff == 0).float().mean())


import re

REINJECT = ("drop", "proj", "rand", "mean", "rmean", "smean", "rsmean", "umean", "uxmean", "flat")


def parse_condition(name):
    """'abl12+pre+uxmean3c' -> (ablate loops or None, msg modes, reinject (mode, from_pass, coda) or None)."""
    if name in CONDITIONS:
        return (*CONDITIONS[name], None)
    loops, msgs, rj = None, [], None
    for part in name.split("+"):
        if part == "clean":
            continue
        if part == "pre":
            loops = (loops or []) + [-2]
        elif part in ("msgabl", "msgzero"):
            msgs.append(part[3:].replace("abl", "ablate"))
        elif (m := re.fullmatch(r"abl(\d+)", part)):
            loops = (loops or []) + [int(c) - 1 for c in m.group(1)]
        elif (m := re.fullmatch(r"(%s)(\d+)(c?)" % "|".join(sorted(REINJECT, key=len, reverse=True)), part)):
            rj = (m.group(1), int(m.group(2)[0]) - 1, m.group(3) == "c")
        else:
            raise ValueError(f"bad condition part {part!r}")
    return loops, msgs, rj


CONDITIONS = {  # name: (ablate loops or None, msg specs)
    "clean": (None, []),
    "abl12": ([0, 1], []),
    "abl12+pre": ([0, 1, -2], []),
    "abl12+msgabl": ([0, 1], ["ablate"]),
    "abl12+msgzero": ([0, 1], ["zero"]),
    "abl12+pre+msgabl": ([0, 1, -2], ["ablate"]),
    "abl12+pre+msgzero": ([0, 1, -2], ["zero"]),
    "msgzero": (None, ["zero"]),
    "abl123": ([0, 1, 2], []),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", default="thinkingmachines/Inkling-2-Mini")
    ap.add_argument("--direction", default=None, help="method:site (default: analysis/inkling_site.json)")
    ap.add_argument("--max-per-class", type=int, default=40)
    ap.add_argument("--max-cot-tokens", type=int, default=256)
    ap.add_argument("--msg-pass", type=int, default=2, help="0-based pass whose incoming messages are changed")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-loops", type=int, default=None, help="core passes to run (default: the trained count)")
    ap.add_argument("--conditions", default=None, help="semicolon-separated (default: the msg / prelude set above)")
    ap.add_argument("--dx", default="hh_eoi:t0.l6", help="d_x: the prelude's own refusal direction (method:site)")
    ap.add_argument("--span-sites", default="t0.l1,t0.l2,t0.l3,t0.l4,t0.l5,t0.l6", help="smean: hh_eoi at these sites")
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from loop_steer.paths import run_dir, setup_job_env
    setup_job_env()
    import pandas as pd
    import torch
    from vllm import LLM, SamplingParams

    from loop_steer import vllm_inkling
    from loop_steer.cot import format_prompt
    from loop_steer.models import (eoi_length, load_tokenizer, loop_overrides, model_sites, prepare_vllm, vllm_kwargs,
                                   vllm_model)

    rd = run_dir(args.model)
    direction = args.direction or "{method}:{site}".format(**json.load(open(rd / "analysis" / "inkling_site.json")))
    method, site = direction.split(":")
    dirs = torch.load(rd / "directions" / "train.pt", weights_only=False)
    t, l = (int(x) for x in site.replace("t", "").replace("l", "").split("."))
    u = dirs[method]["dirs"][dirs["sites"].index((t, l))].float()
    u = u / u.norm()

    def site_dir(spec):
        m_, st = spec.split(":")
        tt, ll = (int(x) for x in st.replace("t", "").replace("l", "").split("."))
        v = dirs[m_]["dirs"][dirs["sites"].index((tt, ll))].float()
        return v / v.norm()

    def orthonormal(vs):
        q, _ = torch.linalg.qr(torch.stack(vs).T)
        return q.T.contiguous()

    dx = site_dir(args.dx)
    span = orthonormal([site_dir(f"hh_eoi:{st}") for st in args.span_sites.split(",")])
    g_ = torch.Generator().manual_seed(0)
    rand1 = orthonormal([torch.randn(u.shape[0], generator=g_)])
    rspan = orthonormal([torch.randn(u.shape[0], generator=g_) for _ in range(len(span))])
    rj_dirs = {"proj": orthonormal([dx]), "rand": rand1, "mean": orthonormal([dx]), "rmean": rand1, "smean": span,
               "rsmean": rspan, "umean": orthonormal([u]), "uxmean": orthonormal([u, dx])}
    print(f"cos(u, d_x) = {float(u @ dx):.3f}; prelude span dim {len(span)}", flush=True)

    tok = load_tokenizer(args.model)
    k_eoi = eoi_length(tok)
    df = pd.read_parquet(rd / "generations" / "train_clean.scored.parquet").dropna(subset=["sr_score"])
    cots = df.groupby(["prompt_idx", "cot_rep"]).agg(prompt=("prompt", "first"), cot=("cot", "first"),
                                                     lo=("sr_score", "max"), hi=("sr_score", "min")).reset_index()
    seqs = []  # (ids, label, prompt length, n CoT tokens)
    for rows, label in ((cots[cots.lo < 0.2].head(args.max_per_class), 1),
                        (cots[cots.hi > 0.8].head(args.max_per_class), 0)):
        for r in rows.itertuples():
            p = tok(format_prompt(tok, r.prompt), add_special_tokens=False).input_ids
            c = tok.encode(r.cot, add_special_tokens=False)[: args.max_cot_tokens]
            seqs.append((p + c, label, len(p), len(c)))
    labels = torch.tensor([s[1] for s in seqs])
    print(f"{int(labels.sum())} refusing, {int((1 - labels).sum())} complying CoTs; u = {direction}", flush=True)

    prepare_vllm(args.model)
    llm = LLM(model=vllm_model(args.model), enforce_eager=True, enable_prefix_caching=False, seed=0,
              max_model_len=max(len(s[0]) for s in seqs) + 8, gpu_memory_utilization=0.6,
              hf_overrides=loop_overrides(args.model, args.n_loops), **vllm_kwargs(args.model))
    sites = model_sites(llm.llm_engine.model_config.hf_config)
    prompts = [{"prompt_token_ids": s[0]} for s in seqs]

    # the prelude state's mean over every token of every trace (but position 0), clean: embedding = resid_pre of
    # layer 0, prelude block = resid_pre of layer 6 in pass 1
    i_emb, i_pre = sites.index((0, 0)), sites.index((0, 6))
    llm.apply_model(functools.partial(vllm_inkling.install, specs=[{"kind": "capture", "n_req": len(seqs), "n_windows": 1}]))
    llm.generate(prompts, [SamplingParams(max_tokens=1, temperature=0.0, extra_args={"ls_cap": (k, [(1, len(s[0]))])})
                           for k, s in enumerate(seqs)], use_tqdm=False)
    sums, counts = llm.apply_model(vllm_inkling.fetch_capture)[0]
    flat = torch.stack([sums[:, 0, i_emb].sum(0), sums[:, 0, i_pre].sum(0)]) / counts[:, 0].sum().float()  # [2, D]
    params = [SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0,
                             extra_args={"ls_cap": (k, [(P - k_eoi, P), (P, P + n)])})
              for k, (_, _, P, n) in enumerate(seqs)]

    rows = []
    names = args.conditions.split(";") if args.conditions else list(CONDITIONS)
    for name in names:
        loops, msgs, rj = parse_condition(name)
        specs = [{"kind": "capture", "n_req": len(seqs), "n_windows": 2}]
        if loops is not None:
            specs.append({"kind": "ablate", "dir": u, "loops": loops})
        specs += [{"kind": "msg", "mode": m, "passes": [args.msg_pass], "dir": u} for m in msgs]
        if rj is not None:
            mode, p0, coda = rj
            spec = {"kind": "reinject", "from_pass": p0, "coda": coda,
                    "mode": {"rand": "proj", "drop": "drop", "flat": "flat", "proj": "proj"}.get(mode, "mean")}
            if mode not in ("drop", "flat"):
                d_ = rj_dirs[mode]
                spec.update({"dirs": d_, "means": flat @ d_.T})
            spec["flat"] = flat
            specs.append(spec)
        llm.apply_model(functools.partial(vllm_inkling.install, specs=specs))
        outs = llm.generate(prompts, params, use_tqdm=False)
        sums, counts = llm.apply_model(vllm_inkling.fetch_capture)[0]
        means = sums / counts.clamp(min=1)[..., None, None].float()  # [R, 2, S, D]
        nll = torch.tensor([-sum(list(d.values())[0].logprob for d in o.prompt_logprobs[P:P + n]) / n
                            for o, (_, _, P, n) in zip(outs, seqs)])
        for dname, dvec in (("u", u), ("dx", dx)):
            proj = means @ dvec  # [R, 2, S]
            for w, window in enumerate(("eoi", "cot")):
                for s_i, (tt, ll) in enumerate(sites):
                    pos, neg = proj[labels == 1, w, s_i], proj[labels == 0, w, s_i]
                    rows.append({"condition": name, "dir": dname, "window": window, "loop": tt, "layer": ll,
                                 "sep": float(pos.mean() - neg.mean()), "auc": auc(pos, neg),
                                 "mean_ref": float(pos.mean()), "mean_com": float(neg.mean())})
        rows.append({"condition": name, "dir": "", "window": "nll", "loop": -1, "layer": -1, "sep": float("nan"),
                     "auc": float("nan"), "mean_ref": float(nll[labels == 1].mean()),
                     "mean_com": float(nll[labels == 0].mean())})
        print(f"  {name:18s} done; CoT NLL {nll.mean():.3f}", flush=True)
    llm.apply_model(vllm_inkling.clear)

    res = pd.DataFrame(rows)
    out = rd / "analysis" / f"inkling_rebuild{'_' + args.tag if args.tag else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    order = names
    show = [(tt, ll) for tt, ll in sites if ll in (l, 17) or ll >= 19]
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        for window in ("eoi", "cot"):
            s = res[(res.window == window) & (res.dir == "u")].copy()
            s["site"] = [f"t{a}.l{b}" for a, b in zip(s.loop, s.layer)]
            s = s[[(a, b) in show for a, b in zip(s.loop, s.layer)]]
            print(f"\nseparation along u ({window}), refusing - complying")
            print(s.pivot(index="condition", columns="site", values="sep").reindex(order).round(4).to_string())
        n = res[res.window == "nll"].set_index("condition").reindex(order)
        print("\nCoT NLL (refusing / complying)")
        print(n[["mean_ref", "mean_com"]].round(3).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
