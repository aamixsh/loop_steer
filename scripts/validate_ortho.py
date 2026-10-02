#!/usr/bin/env python3
"""Check that three ways of ablating a direction agree (greedy decoding, bf16).

  A. HF + ablation hooks (resid_pre, attn out, mlp out at every layer)
  B. HF + in-place weight orthogonalization (loop_steer.ortho)
  C. vLLM + in-place weight orthogonalization via LLM.apply_model
plus the unmodified model, so you can see the direction actually changes outputs.
Reports per-prompt agreement of the first N generated tokens.
"""

import argparse
import functools
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--direction", required=True, help=".pt file with a [D] tensor")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--vllm-mem", type=float, default=0.35)
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from loop_steer.paths import setup_job_env
    setup_job_env()
    import torch
    from vllm import LLM, SamplingParams

    from loop_steer.cot import format_prompt, load_prompts
    from loop_steer.hooks import ablation_hooks, hooked
    from loop_steer.models import load_model, load_tokenizer
    from loop_steer.ortho import orthogonalize_

    direction = torch.load(args.direction, weights_only=False).float().cpu()
    tok = load_tokenizer(args.model)
    tok.padding_side = "left"
    prompts = [format_prompt(tok, p) for p in load_prompts("test")[: args.n_prompts]]

    def hf_generate(model):
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_tokens, do_sample=False)
        return [o[enc.input_ids.shape[1]:].tolist() for o in out]

    model = load_model(args.model)
    results = {"clean_hf": hf_generate(model)}
    with hooked(ablation_hooks(model, direction)):
        results["A_hooks_hf"] = hf_generate(model)
    orthogonalize_(model, direction)
    results["B_ortho_hf"] = hf_generate(model)
    del model
    torch.cuda.empty_cache()

    llm = LLM(model=args.model, gpu_memory_utilization=args.vllm_mem, max_model_len=4096, seed=0)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0)
    results["clean_vllm"] = [o.outputs[0].token_ids for o in llm.generate(prompts, sp)]
    llm.apply_model(functools.partial(orthogonalize_, direction=direction))
    llm.reset_prefix_cache()
    results["C_ortho_vllm"] = [o.outputs[0].token_ids for o in llm.generate(prompts, sp)]

    def agree(a, b):  # length of common prefix / max_tokens, averaged over prompts
        tot = 0
        for x, y in zip(a, b):
            n = 0
            for u, v in zip(x, y):
                if u != v:
                    break
                n += 1
            tot += n / args.max_tokens
        return tot / len(a)

    names = list(results)
    print("common-prefix agreement (fraction of first tokens identical):")
    print(" " * 14 + "".join(f"{n:>14s}" for n in names))
    for a in names:
        print(f"{a:>14s}" + "".join(f"{agree(results[a], results[b]):14.2f}" for b in names))
    for i in range(min(3, len(prompts))):
        for n in ("clean_hf", "A_hooks_hf", "C_ortho_vllm"):
            print(f"[{i}] {n}: {tok.decode(results[n][i])[:200]!r}")


if __name__ == "__main__":
    main()
