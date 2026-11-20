#!/usr/bin/env python3
"""Cache per-CoT window means of resid_pre at every (loop, layer) site.

For each unique (prompt, CoT) in a labels parquet, runs prompt + CoT tokens
(CoT includes ``<think>`` ... ``</think>``) through the HF model and stores
the token-mean of the residual stream entering each decoder layer over:
  cot          all CoT tokens (reasoning-manipulation v4 'cot')
  cot_first150 first 150 CoT tokens (v1/v2)
  cot_last150  last 150 CoT tokens (decision-adjacent)
  eoi          end-of-instruction template tokens (v4 'baseline' / Arditi)
  prompt       all prompt tokens
Output: ``activations/<name>.pt`` = {meta, windows, sites, acts[N, W, S, D] fp16}.

Inkling 2 Mini (no HF path in this environment) runs through vLLM as served, NVFP4 / MXFP8 experts included
(``loop_steer.vllm_inkling`` capture; its resid_pre is the running partial block, see there), in ``.venv-inkling``.
"""

import argparse
import os
from pathlib import Path

WINDOWS = ("cot", "cot_first150", "cot_last150", "eoi", "prompt")


def window_bounds(n_prompt, n_cot, k_eoi):
    p, e = n_prompt, n_prompt + n_cot
    return {
        "cot": (p, e), "cot_first150": (p, min(p + 150, e)), "cot_last150": (max(p, e - 150), e),
        "eoi": (p - k_eoi, p), "prompt": (0, p),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--name", required=True, help="Output name under the run's activations/ dir")
    ap.add_argument("--labeled-only", action="store_true", help="Skip CoTs with no cot- or prompt-level label")
    ap.add_argument("--max-batch-tokens", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=512, help="vLLM path: requests per capture call")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from loop_steer.paths import run_dir, setup_job_env
    setup_job_env()
    import pandas as pd
    import torch
    from tqdm import tqdm

    from loop_steer.cot import format_prompt
    from loop_steer.models import eoi_length, is_gpt_oss, is_inkling, load_tokenizer
    from loop_steer.models import model_sites

    tok = load_tokenizer(args.model)
    vllm_path = is_inkling(args.model)
    if not vllm_path:
        from loop_steer.hooks import capture_window_means
        from loop_steer.models import load_model
        model = load_model(args.model)
    k_eoi = eoi_length(tok)  # default template; Ouro and Qwen3 both open <think> themselves

    df = pd.read_parquet(args.labels)
    if args.labeled_only:
        df = df[df.label_cot.notna() | df.label_prompt.notna()]
    df = df.reset_index(drop=True)

    seqs = []
    for row in df.itertuples():
        if is_gpt_oss(args.model):  # the generation prompt: date-pinned template text (see loop_steer.cot)
            p_ids = tok(format_prompt(tok, row.prompt), add_special_tokens=False).input_ids
        else:
            p_ids = tok.apply_chat_template([{"role": "user", "content": row.prompt}],
                                            add_generation_prompt=True, tokenize=True)
            if isinstance(p_ids, dict) or hasattr(p_ids, "input_ids"):
                p_ids = p_ids["input_ids"]
        c_ids = tok.encode(row.cot, add_special_tokens=False)
        seqs.append((list(p_ids), c_ids))
    df["n_prompt_tokens"] = [len(p) for p, _ in seqs]
    df["n_cot_tokens"] = [len(c) for _, c in seqs]

    if vllm_path:
        acts, sites = capture_vllm(args, tok, seqs, k_eoi)
        save(args, df, sites, k_eoi, acts)
        return
    sites = model_sites(model.config)
    D = model.config.hidden_size
    acts = torch.zeros(len(df), len(WINDOWS), len(sites), D, dtype=torch.float16)

    # Longest first, so each batch's first sequence sets its padded length.
    length = lambda i: len(seqs[i][0]) + len(seqs[i][1])
    batches, cur = [], []
    for i in sorted(range(len(seqs)), key=length, reverse=True):
        if cur and (len(cur) + 1) * length(cur[0]) > args.max_batch_tokens:
            batches.append(cur)
            cur = []
        cur.append(i)
    batches.append(cur)

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for batch in tqdm(batches, desc="activations"):
        T = max(len(seqs[i][0]) + len(seqs[i][1]) for i in batch)
        ids = torch.full((len(batch), T), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), T), dtype=torch.long)
        weights = torch.zeros(len(WINDOWS), len(batch), T)
        for b, i in enumerate(batch):
            p_ids, c_ids = seqs[i]
            full = p_ids + c_ids
            ids[b, : len(full)] = torch.tensor(full)
            mask[b, : len(full)] = 1
            for w, (s, e) in enumerate(window_bounds(len(p_ids), len(c_ids), k_eoi)[n] for n in WINDOWS):
                weights[w, b, s:e] = 1.0 / max(e - s, 1)
        with torch.inference_mode(), capture_window_means(model, weights) as out:
            model.model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device), use_cache=False)  # skip lm_head
        for s_idx, site in enumerate(sites):
            acts[batch, :, s_idx] = out[site].transpose(0, 1).to(torch.float16)  # [B, W, D]

    save(args, df, sites, k_eoi, acts)


def save(args, df, sites, k_eoi, acts):
    import torch

    from loop_steer.paths import run_dir

    path = run_dir(args.model) / "activations" / f"{args.name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = df.drop(columns=["cot", "scores"], errors="ignore")
    torch.save({"meta": meta, "windows": WINDOWS, "sites": sites, "k_eoi": k_eoi, "acts": acts}, path)
    print(f"Saved {tuple(acts.shape)} (k_eoi={k_eoi}) -> {path}")


def capture_vllm(args, tok, seqs, k_eoi):
    """Window means of resid_pre at every site through vLLM (``loop_steer.vllm_inkling`` capture)."""
    import functools

    import torch
    from tqdm import tqdm
    from vllm import LLM, SamplingParams

    from loop_steer import vllm_inkling
    from loop_steer.models import model_sites, prepare_vllm, vllm_kwargs, vllm_model

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # capture buffers live in this process
    prepare_vllm(args.model)
    longest = max(len(p) + len(c) for p, c in seqs)
    llm = LLM(model=vllm_model(args.model), enforce_eager=True, enable_prefix_caching=False, seed=0,
              max_model_len=longest + 8, gpu_memory_utilization=args.gpu_memory_utilization,
              **{"tokenizer": args.model, **vllm_kwargs(args.model)})
    config = llm.llm_engine.model_config.hf_config
    sites = model_sites(config)
    W, D = len(WINDOWS), config.text_config.hidden_size
    acts = torch.zeros(len(seqs), W, len(sites), D, dtype=torch.float16)
    chunk = args.chunk
    llm.apply_model(functools.partial(vllm_inkling.install,
                                      specs=[{"kind": "capture", "n_req": chunk, "n_windows": W}]))
    for start in tqdm(range(0, len(seqs), chunk), desc="activations"):
        idx = list(range(start, min(start + chunk, len(seqs))))
        prompts, params = [], []
        for k, i in enumerate(idx):
            p_ids, c_ids = seqs[i]
            b = window_bounds(len(p_ids), len(c_ids), k_eoi)
            prompts.append({"prompt_token_ids": p_ids + c_ids})
            params.append(SamplingParams(max_tokens=1, temperature=0.0,
                                         extra_args={"ls_cap": (k, [b[n] for n in WINDOWS])}))
        llm.generate(prompts, params, use_tqdm=False)
        sums, counts = llm.apply_model(vllm_inkling.fetch_capture)[0]
        n = len(idx)
        expect = torch.tensor([[max(e - s, 0) for s, e in (window_bounds(len(seqs[i][0]), len(seqs[i][1]), k_eoi)[w]
                                                            for w in WINDOWS)] for i in idx], dtype=torch.float64)
        if not torch.equal(counts[:n], expect):
            bad = (counts[:n] != expect).any(1).nonzero().flatten().tolist()[:5]
            raise RuntimeError(f"captured token counts differ from the windows (rows {bad}); prefix caching on?")
        acts[idx] = (sums[:n] / counts[:n].clamp(min=1)[..., None, None].float()).to(torch.float16)
    llm.apply_model(vllm_inkling.clear)
    return acts, sites


if __name__ == "__main__":
    main()
