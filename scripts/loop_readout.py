#!/usr/bin/env python3
"""How final is each loop's state? Next-token readout after loop k of the model's own clean traces.

    scripts/loop_readout.py --gpu 0 --model ByteDance/Ouro-1.4B-Thinking
    scripts/loop_readout.py --gpu 0 --model nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 --max-loops 4 \
        --gen-path train_clean.scored.parquet
    scripts/loop_readout.py --gpu 0 --model Qwen/Qwen3-8B --blocks 4

Tests whether a model trained to exit early (Ouro: next-token loss after every loop) keeps near-final states in its
intermediate loops, unlike models trained at a fixed depth (Nanbeige, Dory). Teacher-forces the same refusing and
complying CoTs as ``scripts/loop_rebuild.py`` (formatted prompt + first ``--max-cot-tokens`` CoT tokens) and reads the
next-token distribution over the CoT positions after loop k = 1..T with the model's own readout. Running loops 1..k is
the same computation as the first k loops of a T-loop run, so the readout after loop k is the model run with k loops
(``load_model(n_loops_override=k)``): Ouro's early exit (shared norm + LM head), Nanbeige's final norm + LM head, Dory's
output stack + LM head. ``--blocks N`` on a standard model reads out after each of N equal layer blocks with the final
norm + LM head (logit lens) as the dense counterpart.

Per sequence and k: ``nll`` of the actual CoT tokens, ``agree`` = top-1 agreement with the readout after loop T, and
``kl`` = KL(readout T || readout k) over readout T's top-64 tokens (``top_mass`` = their total probability under T).
Writes analysis/loop_readout[_<tag>].csv. HF backend, one GPU.
"""

import argparse
import gc
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-loops", type=int, default=None, help="T (default: the trained number of loops)")
    ap.add_argument("--blocks", type=int, default=None, help="standard model: read out after N equal layer blocks")
    ap.add_argument("--max-per-class", type=int, default=40)
    ap.add_argument("--max-cot-tokens", type=int, default=256)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--gen-path", default=None, help="clean traces, relative to generations/")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from loop_steer.paths import run_dir, setup_job_env
    setup_job_env()
    import pandas as pd
    import torch
    from transformers import AutoConfig

    from loop_steer.cot import format_prompt
    from loop_steer.models import _remote_kwargs, is_dory, is_ouro, load_model, load_tokenizer, n_loops

    rd = run_dir(args.model)
    tok = load_tokenizer(args.model)
    gen_path = (rd / "generations" / "select_clean.scored.parquet") if is_ouro(args.model) \
        else (rd / "generations" / "clean_select" / "none.scored.parquet")
    if args.gen_path:
        gen_path = rd / "generations" / args.gen_path
    elif not gen_path.exists():
        gen_path = rd / "generations" / "select_clean.scored.parquet"
    df = pd.read_parquet(gen_path).dropna(subset=["sr_score"])
    cots = df.groupby(["prompt_idx", "cot_rep"]).agg(prompt=("prompt", "first"), cot=("cot", "first"),
                                                     lo=("sr_score", "max"), hi=("sr_score", "min")).reset_index()
    seqs = []  # (ids [1, S], label, first CoT position, n CoT tokens)
    for rows, label in ((cots[cots.lo < 0.2].head(args.max_per_class), 1),
                        (cots[cots.hi > 0.8].head(args.max_per_class), 0)):
        for r in rows.itertuples():
            p = tok(format_prompt(tok, r.prompt), add_special_tokens=False).input_ids
            c = tok.encode(r.cot, add_special_tokens=False)[: args.max_cot_tokens]
            seqs.append((torch.tensor([p + c]), label, len(p), len(c)))
    print(f"{args.model}: {sum(s[1] for s in seqs)} refusing, {sum(1 - s[1] for s in seqs)} complying CoTs "
          f"from {gen_path.name}", flush=True)

    if is_dory(args.model):  # registers the Dory config and model classes
        from loop_steer.dory import register
        register()
    cfg = AutoConfig.from_pretrained(args.model, **_remote_kwargs(args.model))
    T = args.blocks or args.max_loops or n_loops(cfg)

    rows, ref = [], {}

    def record(k, i, lp):
        """Metrics of sequence i's readout after loop k ([n_cot, V] log-probs); k == T must come first."""
        ids, label, start, n = seqs[i]
        if k == T:
            ref[i] = (lp.topk(args.topk, -1), lp.argmax(-1))
        (top_lp, top_idx), arg = ref[i]
        p_ref = top_lp.exp()
        rows.append({"k": k, "seq": i, "label": label,
                     "nll": float(-lp.gather(-1, ids[0, start: start + n].to(lp.device)[:, None]).mean()),
                     "agree": float((lp.argmax(-1) == arg).float().mean()),
                     "kl": float((p_ref * (top_lp - lp.gather(-1, top_idx))).sum(-1).mean()),
                     "top_mass": float(p_ref.sum(-1).mean())})

    def report(k):
        r = pd.DataFrame([x for x in rows if x["k"] == k])
        print(f"  k={k}: nll {r.nll.mean():.3f}  agree {r.agree.mean():.3f}  kl {r.kl.mean():.3f}", flush=True)

    if args.blocks:  # one forward per sequence, every block end read out with the final norm + LM head
        model = load_model(args.model)
        n_layers = model.config.num_hidden_layers
        ends = [round(n_layers * (b + 1) / T) for b in range(T)]
        for i, (ids, _, start, n) in enumerate(seqs):
            with torch.inference_mode():
                hs = model(input_ids=ids.to(model.device), output_hidden_states=True, use_cache=False).hidden_states
                for k in [T, *range(1, T)]:
                    h = hs[ends[k - 1]][0, start - 1: start - 1 + n]
                    if k < T:  # the last entry already has the final norm applied
                        h = model.model.norm(h)
                    record(k, i, torch.log_softmax(model.lm_head(h).float(), -1))
        for k in [T, *range(1, T)]:
            report(k)
    else:  # the model run with k loops = the readout after loop k
        for k in [T, *range(1, T)]:
            model = load_model(args.model, n_loops_override=k)
            for i, (ids, _, start, n) in enumerate(seqs):
                with torch.inference_mode():
                    logits = model(input_ids=ids.to(model.device), use_cache=False).logits[0, start - 1: start - 1 + n]
                    record(k, i, torch.log_softmax(logits.float(), -1))
            del model
            gc.collect()
            torch.cuda.empty_cache()
            report(k)

    res = pd.DataFrame(rows)
    res["model"] = rd.name
    res["T"] = T
    res["unit"] = "block" if args.blocks else "loop"
    name = "loop_readout" + (f"_{args.tag}" if args.tag else (f"_blocks{T}" if args.blocks else f"_loops{T}"))
    (rd / "analysis").mkdir(parents=True, exist_ok=True)
    res.to_csv(rd / "analysis" / f"{name}.csv", index=False)
    summ = res.groupby(["k", "label"])[["nll", "agree", "kl"]].mean().unstack("label").round(3)
    print(summ.to_string())
    print(f"-> {rd / 'analysis' / name}.csv")


if __name__ == "__main__":
    main()
