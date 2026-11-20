#!/usr/bin/env python3
"""Median residual-stream norm at every site, for norm-matched activation addition (``intervene.py ... rel=1``).

    scripts/stream_norms.py --gpu 0 --model Qwen/Qwen3-8B
    scripts/stream_norms.py --gpu 0 --model ByteDance/Ouro-1.4B-Thinking --n-loops 6

Teacher-forces ``--n-seqs`` of the model's own clean harmful-prompt traces (formatted prompt + first
``--max-cot-tokens`` CoT tokens) and records the per-token norm of the stream entering every decoder layer (every
loop), position 0 excluded. Writes ``directions/stream_norms.pt`` (``stream_norms_loops<N>.pt`` with ``--n-loops``):
``{"sites": [(loop, layer), ...], "norms": [S] median norms, "n_loops": T}``. HF backend, one GPU, about a minute.
"""

import argparse
import os

TRACES = ["train_clean.scored.parquet", "clean_train/none.scored.parquet", "clean_train/none.parquet",
          "select_clean.scored.parquet", "clean_select/none.scored.parquet"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-loops", type=int, default=None)
    ap.add_argument("--n-seqs", type=int, default=32)
    ap.add_argument("--max-cot-tokens", type=int, default=256)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from loop_steer.paths import run_dir, setup_job_env
    setup_job_env()
    import pandas as pd
    import torch

    from loop_steer.cot import format_prompt
    from loop_steer.models import load_model, load_tokenizer
    from loop_steer.selfcorr import stream_hooks, structure

    rd = run_dir(args.model)
    out = rd / "directions" / (f"stream_norms_loops{args.n_loops}.pt" if args.n_loops else "stream_norms.pt")
    if args.skip_existing and out.exists():
        print(f"{out} exists; skipping")
        return
    path = next((rd / "generations" / p for p in TRACES if (rd / "generations" / p).exists()), None)
    if path is None:
        raise FileNotFoundError(f"no clean traces under {rd / 'generations'} (looked for {TRACES})")
    df = pd.read_parquet(path).drop_duplicates(["prompt_idx", "cot_rep"]).head(args.n_seqs)
    tok = load_tokenizer(args.model)
    from loop_steer.models import is_inkling

    if is_inkling(args.model):  # vLLM as served (loop_steer.vllm_inkling: resid_pre = the running partial block)
        return vllm_norms(args, df, tok, path, out)
    model = load_model(args.model, n_loops_override=args.n_loops)
    st = structure(model)
    acc = {}

    def rec(s, h):
        acc.setdefault(s, []).append(h[0, 1:].float().norm(dim=-1).cpu())

    for r in df.itertuples():
        ids = tok(format_prompt(tok, r.prompt), add_special_tokens=False).input_ids
        ids += tok.encode(r.cot, add_special_tokens=False)[: args.max_cot_tokens]
        with torch.inference_mode(), stream_hooks(model, rec):
            model(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
    sites = [s for s in st.sites if s in acc]
    norms = torch.tensor([float(torch.cat(acc[s]).median()) for s in sites])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sites": sites, "norms": norms, "n_loops": st.T if st.looped else 1}, out)
    print(f"{len(df)} traces from {path.name}; {len(sites)} sites; norms {norms.min():.1f}-{norms.max():.1f} -> {out}")


def vllm_norms(args, df, tok, path, out):
    import functools

    import torch
    from vllm import LLM, SamplingParams

    from loop_steer import vllm_inkling
    from loop_steer.cot import format_prompt
    from loop_steer.models import loop_overrides, model_sites, prepare_vllm, vllm_kwargs, vllm_model

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    prepare_vllm(args.model)
    seqs = []
    for r in df.itertuples():
        ids = tok(format_prompt(tok, r.prompt), add_special_tokens=False).input_ids
        seqs.append(ids + tok.encode(r.cot, add_special_tokens=False)[: args.max_cot_tokens])
    llm = LLM(model=vllm_model(args.model), enforce_eager=True, enable_prefix_caching=False, seed=0,
              max_model_len=max(map(len, seqs)) + 8, gpu_memory_utilization=0.6,
              hf_overrides=loop_overrides(args.model, args.n_loops), **vllm_kwargs(args.model))
    sites = model_sites(llm.llm_engine.model_config.hf_config)
    llm.apply_model(functools.partial(vllm_inkling.install, specs=[{"kind": "norms"}]))
    llm.generate([{"prompt_token_ids": s} for s in seqs], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
    med = llm.apply_model(vllm_inkling.fetch_norms)[0]
    norms = torch.tensor([med[v] for v in range(len(sites))])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sites": sites, "norms": norms, "n_loops": max(t for t, _ in sites) + 1}, out)
    print(f"{len(df)} traces from {path.name}; {len(sites)} sites; norms {norms.min():.1f}-{norms.max():.1f} -> {out}")


if __name__ == "__main__":
    main()
