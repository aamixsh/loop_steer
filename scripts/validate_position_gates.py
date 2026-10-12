#!/usr/bin/env python3
"""Check position-restricted interventions in vLLM (``pos=`` in intervene.py, ``loop_steer.vllm_hooks``).

    scripts/validate_position_gates.py --gpu 0 [--model ByteDance/Ouro-1.4B-Thinking] [--direction v4_baseline:t3.l16]

1. Class counts: with a ``record`` spec, every token the engine processes is counted per request and class. Stage 1
   (prompt -> CoT) must show P prompt tokens and n_generated - 1 CoT tokens; stage 2 (prompt + CoT -> answer) P prompt,
   C - P CoT and n_generated - 1 answer tokens (the last sampled token is never fed back). Prefix caching is off here
   so every prompt token is counted.
2. Boundaries (teacher-forced, deterministic): prompt log-probs of a fixed prompt + CoT + answer sequence. Gating on
   all classes equals plain ablation; gating on a set of classes leaves every token before the first gated position
   bitwise identical to the matching reference (clean or full ablation) and changes the tokens after it.
Exit code 1 if a check fails. (Greedy generations are not compared: vLLM decoding is not bitwise reproducible.)
"""

import argparse
import functools
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", default="ByteDance/Ouro-1.4B-Thinking")
    ap.add_argument("--direction", default="v4_baseline:t3.l16", help="method:site in directions/train.pt")
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--vllm-mem", type=float, default=0.3)
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ["VLLM_BATCH_INVARIANT"] = "1"  # deterministic kernels, so the teacher-forced checks can be exact
    from loop_steer.paths import run_dir, setup_job_env
    setup_job_env()
    import torch
    from vllm import LLM, SamplingParams

    from loop_steer import vllm_hooks
    from loop_steer.cot import format_prompt, load_prompts
    from loop_steer.models import load_tokenizer, prepare_vllm, vllm_kwargs, vllm_model
    from loop_steer.sampling import vllm_generator

    tok = load_tokenizer(args.model)
    dirs = torch.load(run_dir(args.model) / "directions" / "train.pt", weights_only=False)
    method, site = args.direction.split(":")
    t, l = site.replace("t", "").replace("l", "").split(".")
    u = dirs[method]["dirs"][dirs["sites"].index((int(t), int(l)))].float()
    prompts = [format_prompt(tok, p) for p in load_prompts("train")[500: 500 + args.n_prompts]]

    prepare_vllm(args.model)
    llm = LLM(model=vllm_model(args.model), gpu_memory_utilization=args.vllm_mem, max_model_len=4096, enforce_eager=True,
              enable_prefix_caching=False, attention_config={"backend": "TRITON_ATTN"}, **vllm_kwargs(args.model))
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0, skip_special_tokens=False)
    gen = vllm_generator(llm, sp, tok, gated=True)
    n_ids = lambda text: len(tok(text, add_special_tokens=False).input_ids)
    failures = []

    def install(specs):
        llm.apply_model(functools.partial(vllm_hooks.install, specs=specs))
        llm.reset_prefix_cache()

    def records():
        return sorted(tuple(r) for r in llm.apply_model(vllm_hooks.fetch_records)[0].values())

    # ---- 1. class counts
    install([{"kind": "record"}])
    llm.apply_model(vllm_hooks.fetch_records)  # drop counts from warm-up
    s1 = gen(prompts, prompt_texts=prompts)
    want = sorted((0, n_ids(p), n - 1, 0) for p, (_, n, _) in zip(prompts, s1))
    got = records()
    print(f"stage 1 counts [none, prompt, cot, answer]: want {want}\n                                           got  {got}")
    if want != got:
        failures.append("stage-1 class counts")
    texts = [p + text[: len(text) // 2] + "</think>\n\n" for p, (text, _, _) in zip(prompts, s1)]  # a stand-in CoT
    s2 = gen(texts, prompt_texts=prompts, answer=True)
    want = sorted((0, n_ids(p), n_ids(x) - n_ids(p), n - 1) for p, x, (_, n, _) in zip(prompts, texts, s2))
    got = records()
    print(f"stage 2 counts [none, prompt, cot, answer]: want {want}\n                                           got  {got}")
    if want != got:
        failures.append("stage-2 class counts")

    # ---- 2. teacher-forced boundaries: log-probs of a fixed prompt + CoT + answer sequence under each gate. The
    # log-prob of token j depends on positions < j only, so a gate must leave tokens up to its first gated position
    # bitwise unchanged and change the ones after it.
    seqs = [x + a for x, (a, _, _) in zip(texts, s2)]
    P = [n_ids(p) for p in prompts]
    C = [n_ids(x) for x in texts]

    def logprobs(specs):
        install(specs)
        ids = [tok(x, add_special_tokens=False).input_ids for x in seqs]
        params = [SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0, extra_args={"ls_bounds": (p, c)})
                  for p, c in zip(P, C)]
        outs = llm.generate([{"prompt_token_ids": i} for i in ids], params, use_tqdm=False)
        return [torch.tensor([d[t].logprob for d, t in zip(o.prompt_logprobs[1:], i[1:])]) for o, i in zip(outs, ids)]

    abl = lambda pos: [{"kind": "ablate", "dir": u, "loops": None, "positions": pos}]
    ref = {"clean": logprobs([]), "full": logprobs(abl(None))}
    # (gate, reference it must equal, last token index (exclusive, per sequence) where it must equal it)
    cases = [(["prompt", "cot", "answer"], "full", lambda i: len(ref["full"][i]) + 1),
             (["answer"], "clean", lambda i: C[i]), (["cot", "answer"], "clean", lambda i: P[i]),
             (["prompt"], "full", lambda i: P[i]), (["prompt", "cot"], "full", lambda i: C[i])]
    for pos, name, end in cases:
        out = logprobs(abl(pos))
        same = max(float((o[: end(i) - 1] - r[: end(i) - 1]).abs().max()) for i, (o, r) in enumerate(zip(out, ref[name])))
        after = [float((o[end(i) - 1:] - r[end(i) - 1:]).abs().mean()) for i, (o, r) in enumerate(zip(out, ref[name]))
                 if end(i) - 1 < len(o)]
        msg = f"pos={'+'.join(pos):20s} vs {name:5s}: max |diff| before the boundary {same:.2e}"
        if after:
            msg += f", mean |diff| after it {sum(after) / len(after):.3f}"
        ok = same == 0.0 and (not after or sum(after) > 0)
        print(msg + ("" if ok else "   <-- FAIL"))
        if not ok:
            failures.append(f"teacher-forced pos={'+'.join(pos)}")
    print(f"clean vs full ablation: mean |diff| {sum(float((a - b).abs().mean()) for a, b in zip(ref['clean'], ref['full'])) / len(seqs):.3f}")

    install([])
    print("\nFAILED: " + ", ".join(failures) if failures else "\nall checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
