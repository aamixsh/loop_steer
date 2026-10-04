#!/usr/bin/env python3
"""Check vLLM-engine hooks (loop_steer.vllm_hooks) against HF hooks (loop_steer.hooks).

Greedy-decodes test prompts under: clean, ablation of a direction, and
activation addition of a scaled direction at one (loop, layer); reports the
fraction of leading tokens shared between HF and vLLM for each condition.
The direction defaults to a random unit vector; actadd uses ``--coeff`` times
the mean resid_pre norm at that layer so the effect is visible.
"""

import argparse
import functools
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--model", default="ByteDance/Ouro-1.4B-Thinking")
    ap.add_argument("--direction", default=None, help=".pt [D] tensor; random if omitted")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--loops", default=None, help="Comma list of loops for actadd/ablation (default: all)")
    ap.add_argument("--coeff", type=float, default=1.0)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--vllm-mem", type=float, default=0.1)
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from loop_steer.paths import setup_job_env
    setup_job_env()
    import torch
    from vllm import LLM, SamplingParams

    from loop_steer import vllm_hooks
    from loop_steer.cot import format_prompt, load_prompts
    from loop_steer.hooks import ablation_hooks, actadd_hooks, hooked
    from loop_steer.models import load_model, load_tokenizer, prepare_vllm, vllm_kwargs

    loops = None if args.loops is None else [int(x) for x in args.loops.split(",")]
    tok = load_tokenizer(args.model)
    tok.padding_side = "left"
    prompts = [format_prompt(tok, p) for p in load_prompts("test")[: args.n_prompts]]
    model = load_model(args.model)
    D = model.config.hidden_size
    if args.direction:
        u = torch.load(args.direction, weights_only=False).float().cpu()
    else:
        torch.manual_seed(0)
        u = torch.randn(D)
    u = u / u.norm()

    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    # Mean per-token resid_pre norm at the actadd site sets the vector's scale.
    with torch.inference_mode():
        eye_norms = []
        def norm_hook(m, a, kw):
            if loops is None or kw.get("current_ut", 0) in loops:
                h = a[0] if a else kw["hidden_states"]
                eye_norms.append(float(h[enc.attention_mask.bool()].float().norm(dim=-1).mean()))
        with hooked(lambda: [model.model.layers[args.layer].register_forward_pre_hook(norm_hook, with_kwargs=True)]):
            model(**enc)
    scale = args.coeff * sum(eye_norms) / len(eye_norms)
    print(f"resid_pre norm at layer {args.layer} (loops {loops}): {scale / args.coeff:.2f}; actadd |vec|={scale:.2f}")
    vec = scale * u

    def hf_gen():
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=args.max_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        return [x[enc.input_ids.shape[1]:].tolist() for x in o]

    hf = {"clean": hf_gen()}
    with hooked(ablation_hooks(model, u, loops=loops)):
        hf["ablate"] = hf_gen()
    with hooked(actadd_hooks(model, vec, args.layer, loops=loops)):
        hf["actadd"] = hf_gen()
    del model
    torch.cuda.empty_cache()

    as_ids = prepare_vllm(args.model)
    llm = LLM(model=args.model, gpu_memory_utilization=args.vllm_mem, max_model_len=4096, enforce_eager=True,
              **vllm_kwargs(args.model))
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0)
    specs = {
        "clean": [],
        "ablate": [{"kind": "ablate", "dir": u, "loops": loops}],
        "actadd": [{"kind": "actadd", "vec": vec, "layer": args.layer, "coeff": 1.0, "loops": loops}],
    }
    vl = {}
    for name, spec in specs.items():
        llm.apply_model(functools.partial(vllm_hooks.install, specs=spec))
        llm.reset_prefix_cache()
        inputs = [{"prompt_token_ids": tok(t, add_special_tokens=False).input_ids} for t in prompts] if as_ids else prompts
        vl[name] = [list(o.outputs[0].token_ids) for o in llm.generate(inputs, sp)]
    llm.apply_model(vllm_hooks.clear)

    def prefix(a, b):
        return sum(next((i for i, (x, y) in enumerate(zip(p, q)) if x != y), min(len(p), len(q)))
                   for p, q in zip(a, b)) / (len(a) * args.max_tokens)

    print(f"{'condition':>8s} {'HF~vLLM':>8s} {'HF~HFclean':>11s} {'vLLM~vLLMclean':>15s}")
    for name in specs:
        print(f"{name:>8s} {prefix(hf[name], vl[name]):8.2f} {prefix(hf[name], hf['clean']):11.2f} "
              f"{prefix(vl[name], vl['clean']):15.2f}")
    print("actadd HF  :", repr(tok.decode(hf["actadd"][0])[:160]))
    print("actadd vLLM:", repr(tok.decode(vl["actadd"][0])[:160]))


if __name__ == "__main__":
    main()
