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
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from loop_steer.paths import run_dir, setup_job_env
    setup_job_env()
    import pandas as pd
    import torch
    from vllm import LLM, SamplingParams

    from loop_steer import vllm_inkling
    from loop_steer.cot import format_prompt
    from loop_steer.models import eoi_length, load_tokenizer, model_sites, prepare_vllm, vllm_kwargs, vllm_model

    rd = run_dir(args.model)
    direction = args.direction or "{method}:{site}".format(**json.load(open(rd / "analysis" / "inkling_site.json")))
    method, site = direction.split(":")
    dirs = torch.load(rd / "directions" / "train.pt", weights_only=False)
    t, l = (int(x) for x in site.replace("t", "").replace("l", "").split("."))
    u = dirs[method]["dirs"][dirs["sites"].index((t, l))].float()
    u = u / u.norm()

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
              max_model_len=max(len(s[0]) for s in seqs) + 8, gpu_memory_utilization=0.6, **vllm_kwargs(args.model))
    sites = model_sites(llm.llm_engine.model_config.hf_config)
    prompts = [{"prompt_token_ids": s[0]} for s in seqs]
    params = [SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0,
                             extra_args={"ls_cap": (k, [(P - k_eoi, P), (P, P + n)])})
              for k, (_, _, P, n) in enumerate(seqs)]

    rows = []
    for name, (loops, msgs) in CONDITIONS.items():
        specs = [{"kind": "capture", "n_req": len(seqs), "n_windows": 2}]
        if loops is not None:
            specs.append({"kind": "ablate", "dir": u, "loops": loops})
        specs += [{"kind": "msg", "mode": m, "passes": [args.msg_pass], "dir": u} for m in msgs]
        llm.apply_model(functools.partial(vllm_inkling.install, specs=specs))
        outs = llm.generate(prompts, params, use_tqdm=False)
        sums, counts = llm.apply_model(vllm_inkling.fetch_capture)[0]
        means = sums / counts.clamp(min=1)[..., None, None].float()  # [R, 2, S, D]
        proj = means @ u  # [R, 2, S]
        nll = torch.tensor([-sum(list(d.values())[0].logprob for d in o.prompt_logprobs[P:P + n]) / n
                            for o, (_, _, P, n) in zip(outs, seqs)])
        for w, window in enumerate(("eoi", "cot")):
            for s_i, (tt, ll) in enumerate(sites):
                pos, neg = proj[labels == 1, w, s_i], proj[labels == 0, w, s_i]
                rows.append({"condition": name, "window": window, "loop": tt, "layer": ll,
                             "sep": float(pos.mean() - neg.mean()), "auc": auc(pos, neg),
                             "mean_ref": float(pos.mean()), "mean_com": float(neg.mean())})
        rows.append({"condition": name, "window": "nll", "loop": -1, "layer": -1, "sep": float("nan"),
                     "auc": float("nan"), "mean_ref": float(nll[labels == 1].mean()),
                     "mean_com": float(nll[labels == 0].mean())})
        print(f"  {name:18s} done; CoT NLL {nll.mean():.3f}", flush=True)
    llm.apply_model(vllm_inkling.clear)

    res = pd.DataFrame(rows)
    out = rd / "analysis" / f"inkling_rebuild{'_' + args.tag if args.tag else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    order = list(CONDITIONS)
    show = [(tt, ll) for tt, ll in sites if ll in (l, 17) or ll >= 19]
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        for window in ("eoi", "cot"):
            s = res[res.window == window].copy()
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
