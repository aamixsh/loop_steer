#!/usr/bin/env python3
"""Contrastive activation addition (CAA) on behaviour A/B sets, injected in one loop or all: which steering survives?

    scripts/caa_loops.py --gpu 0 --model ByteDance/Ouro-1.4B-Thinking
    scripts/caa_loops.py --gpu 0 --model Qwen/Qwen3-8B         # dense reference: every 2nd layer

Data: the A/B behaviour sets of Rimsky et al. 2023 (github.com/nrimsky/CAA, cloned to ``data/external/caa_repo``):
coordinate-other-ais, corrigible-neutral-HHH, hallucination, myopic-reward, survival-instinct, sycophancy, refusal.
Prompts use the chat template with an empty thinking block (the model answers directly).

1. **Vectors**: for ``--n-gen`` items of each generate set, the residual stream at the answer-letter token of
   ``prompt + "(A"`` vs ``"(B"`` at every site; vector = mean(matching behaviour) - mean(not matching). Saved as
   ``directions/caa.pt`` (directions-file format: usable by ``perturb_response.py --vec``).
2. **Steering**: on the 50 test items, P(matching answer) = p(match) / (p(A) + p(B)) from the log-probs of the two
   answer continuations, with ``c * vector`` added at layer l (vector from the last loop's site at l) in the first
   loop only, the last loop only, or every loop; plus a random vector of the same norm. Looped models: l at fractions
   ``--fracs`` of a loop; dense models: every ``--dense-step``-th layer.
``--vec-source own``: in each loop, the vector of that loop's own site at l (first and every loop; the last loop is the
default run) -> ``analysis/caa_loops_own.csv``. Writes ``analysis/caa_loops.csv``. HF backend, one GPU.
"""

import argparse
import json
import os

BEHAVIOURS = ["coordinate-other-ais", "corrigible-neutral-HHH", "hallucination", "myopic-reward", "survival-instinct",
              "sycophancy", "refusal"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-gen", type=int, default=300)
    ap.add_argument("--coeffs", default="-4,-2,-1,1,2,4")
    ap.add_argument("--fracs", default="0.25,0.5,0.75")
    ap.add_argument("--dense-step", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--behaviours", default=",".join(BEHAVIOURS))
    ap.add_argument("--vec-source", choices=["last", "own"], default="last",
                    help="vector from the last loop's site at the layer, or from each loop's own site (own: first "
                         "and every loop only; tag caa_loops_own)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from loop_steer.paths import DATA_ROOT, run_dir, setup_job_env
    setup_job_env()
    import pandas as pd
    import torch

    from loop_steer.hooks import actadd_hooks, hooked
    from loop_steer.models import load_model, load_tokenizer, model_sites, n_loops
    from loop_steer.selfcorr import no_think_prompt, pad_batch, stream_hooks, structure

    rd = run_dir(args.model)
    own = args.vec_source == "own"
    args.tag = args.tag or ("caa_loops_own" if own else "caa_loops")
    src = DATA_ROOT / "external" / "caa_repo" / "datasets"
    tok = load_tokenizer(args.model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    model = load_model(args.model)
    dev = model.device
    st = structure(model)
    sites = model_sites(model.config)
    T = n_loops(model.config)
    coeffs = [float(c) for c in args.coeffs.split(",")]

    def encode(question, answer):
        p = tok(no_think_prompt(tok, [{"role": "user", "content": question}]), add_special_tokens=False).input_ids
        full = tok(no_think_prompt(tok, [{"role": "user", "content": question}]) + answer[:2],
                   add_special_tokens=False).input_ids
        return p, full

    vec_path = rd / "directions" / "caa.pt"
    vecs = torch.load(vec_path, weights_only=False) if vec_path.exists() else {"sites": sites}
    rows = []
    out = rd / "analysis" / f"{args.tag}.csv"
    out.parent.mkdir(exist_ok=True)
    for beh in args.behaviours.split(","):
        gen = json.load(open(src / "generate" / beh / "generate_dataset.json"))[: args.n_gen]
        test = json.load(open(src / "test" / beh / "test_dataset_ab.json"))
        # ---- vectors
        if beh not in vecs:
            sums = {}
            items = [(encode(it["question"], it[key])[1], sign) for it in gen
                     for key, sign in (("answer_matching_behavior", 1), ("answer_not_matching_behavior", -1))]
            for b0 in range(0, len(items), args.batch):
                chunk = items[b0: b0 + args.batch]
                ids, valid = pad_batch([c[0] for c in chunk], pad)
                ids, lens = ids.to(dev), valid.sum(1).to(dev)
                sign = torch.tensor([c[1] for c in chunk], dtype=torch.float32, device=dev)

                def rec(s, h):
                    x = h[torch.arange(len(lens), device=dev), lens - 1].float()
                    sums[s] = sums.get(s, 0) + (sign[:, None] * x).sum(0)

                with torch.inference_mode(), stream_hooks(model, rec):
                    model(input_ids=ids, use_cache=False)
            vecs[beh] = {"dirs": torch.stack([sums[s] / len(gen) for s in sites]).cpu()}
            torch.save(vecs, vec_path)
        table = vecs[beh]["dirs"]

        # ---- steering eval
        enc = []
        for it in test:
            for key in ("answer_matching_behavior", "answer_not_matching_behavior"):
                p, full = encode(it["question"], it[key])
                enc.append((p, full))
        ids_all, valid_all = pad_batch([e[1] for e in enc], pad)

        def p_match():
            lps = []
            for b0 in range(0, len(enc), args.batch):
                ids = ids_all[b0: b0 + args.batch].to(dev)
                with torch.inference_mode():
                    lp = torch.log_softmax(model(input_ids=ids, use_cache=False).logits.float(), -1)
                for i in range(ids.shape[0]):
                    p, full = enc[b0 + i]
                    tgt = torch.tensor(full[len(p):], device=dev)
                    pos = torch.arange(len(p) - 1, len(full) - 1, device=dev)
                    lps.append(float(lp[i, pos, tgt].sum()))
            lps = torch.tensor(lps).view(-1, 2)
            return float(torch.softmax(lps, -1)[:, 0].mean())

        base = p_match()
        if st.looped:
            layers = sorted({st.site_at(T - 1, f)[1] for f in (float(x) for x in args.fracs.split(","))})
            loopsets = {"first": [0], "all": None} if own else {"first": [0], "last": [T - 1], "all": None}
            src_site = lambda l: (T - 1, l)  # noqa: E731
        else:
            assert not own, "--vec-source own is for looped models"
            layers = list(range(0, len(sites), args.dense_step))
            loopsets = {"all": None}
            src_site = lambda l: (0, l)  # noqa: E731
        gen_r = torch.Generator().manual_seed(0)
        for l in layers:
            v = table[sites.index(src_site(l))]
            r = torch.randn(v.shape, generator=gen_r)
            r = r / r.norm() * v.norm()
            for lname, loops in loopsets.items():
                for kind, vec in (("caa", v), ("random", r)):
                    for c in (coeffs if kind == "caa" else [min(coeffs), max(coeffs)]):
                        if own and kind == "caa":  # each loop gets the vector of its own site at layer l
                            regs = [actadd_hooks(model, table[sites.index((t, l))].to(dev), l, coeff=c, loops=[t])
                                    for t in (loops if loops is not None else range(T))]
                        else:
                            regs = [actadd_hooks(model, vec.to(dev), l, coeff=c, loops=loops)]
                        with hooked(lambda: [h for reg in regs for h in reg()]):
                            pm = p_match()
                        rows.append({"behaviour": beh, "layer": l, "depth": st.depth(src_site(l)), "loops": lname,
                                     "kind": kind, "c": c, "p_match": pm, "p_match_clean": base,
                                     "vec_norm": float(table[sites.index((0, l))].norm() if own and lname == "first" and kind == "caa"
                                                       else v.norm())})
            print(f"  {beh} layer {l}: " + ", ".join(f"{r['loops']} c={r['c']:+g} {r['p_match']:.2f}"
                                                    for r in rows[-len(loopsets) * (len(coeffs) + 2):]
                                                    if r["kind"] == "caa" and abs(r["c"]) == max(coeffs)), flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}, {vec_path}")


if __name__ == "__main__":
    main()
