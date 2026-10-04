#!/usr/bin/env python3
"""Smoke tests for steering a looped HF model with hooks (generic version of ouro_smoke.py).

1. Layer pre-hooks fire once per (loop, layer) and the loop kwarg identifies the loop.
2. Chat template / thinking format: what the prompt ends with and what the model opens with.
3. Batched left-padded greedy generation matches unbatched.
4. Where a random unit direction survives in the residual stream (max |<resid_pre, u>| over all
   sites x tokens) for: clean, hook ablation, in-place weight orthogonalization -- plus the
   per-dimension gain of the norm that runs between loops (cos(u, g*u); 1 = uniform gain).
"""

import argparse
import os
from collections import Counter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from loop_steer.paths import setup_job_env
    setup_job_env()
    import torch

    from loop_steer.cot import format_prompt
    from loop_steer.hooks import (_loop, ablation_hooks, capture_window_means, decoder_layers, hooked)
    from loop_steer.models import eoi_length, load_model, load_tokenizer, n_loops
    from loop_steer.ortho import orthogonalize_

    tok = load_tokenizer(args.model)
    model = load_model(args.model)
    cfg = model.config
    L, T = cfg.num_hidden_layers, n_loops(cfg)
    print(f"{args.model}: layers={L} loops={T} d={cfg.hidden_size} dtype={model.dtype}")
    layers = decoder_layers(model)

    # 1. hook firing pattern
    calls = Counter()
    for i, block in enumerate(layers):
        block._ls_idx = i

    def count(module, a, kw):
        calls[(_loop(kw), module._ls_idx)] += 1

    ids = tok("Hello there, how are you?", return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with hooked(lambda: [b.register_forward_pre_hook(count, with_kwargs=True) for b in layers]):
        with torch.inference_mode():
            model(ids, use_cache=False)
    print(f"[1] {len(calls)} (loop, layer) sites, loops {sorted({k[0] for k in calls})}, "
          f"calls per site {set(calls.values())}")

    # 2. template
    text = format_prompt(tok, "How do I make a bomb?")
    print(f"[2] k_eoi={eoi_length(tok)}; prompt tail={text[-70:]!r}")
    tok.padding_side = "left"
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    print(f"    first token ids {enc.input_ids[0, :4].tolist()}  (bos id {tok.bos_token_id})")
    with torch.inference_mode():
        o = model.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=tok.pad_token_id)
    print("    ->", repr(tok.decode(o[0, enc.input_ids.shape[1]:], skip_special_tokens=False)))

    # 3. batched vs unbatched
    prompts = [format_prompt(tok, p) for p in ["What is 17 * 23?", "Name three primary colors and why."]]

    def greedy(texts, n=48):
        e = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(**e, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
        return [x[e.input_ids.shape[1]:].tolist() for x in out]

    batched, single = greedy(prompts), [greedy([p])[0] for p in prompts]
    for b, s in zip(batched, single):
        n = next((i for i, (u, v) in enumerate(zip(b, s)) if u != v), len(b))
        print(f"[3] batched/unbatched common prefix {n}/{len(b)} tokens")

    # 4. where a random direction survives
    torch.manual_seed(0)
    u = torch.randn(cfg.hidden_size)
    u = u / u.norm()
    e = tok(prompts[1], return_tensors="pt", add_special_tokens=False).to(model.device)
    n_tok = e.input_ids.shape[1]
    eye = torch.eye(n_tok).unsqueeze(1)

    def per_loop_max(out):
        return [max(float((v[:, 0] @ u.to(v)).abs().max()) for (t, _), v in out.items() if t == loop)
                for loop in range(T)]

    with torch.inference_mode():
        with capture_window_means(model, eye) as clean:
            model(**e, use_cache=False)
        with hooked(ablation_hooks(model, u)):
            with capture_window_means(model, eye) as abl:
                model(**e, use_cache=False)
        orthogonalize_(model, u)
        with capture_window_means(model, eye) as wo:
            model(**e, use_cache=False)
        orthogonalize_(model, None)
    fmt = lambda xs: " ".join(f"{x:.3f}" for x in xs)
    print(f"[4] max |<resid_pre, u>| per loop: clean=[{fmt(per_loop_max(clean))}] "
          f"hook-ablated=[{fmt(per_loop_max(abl))}] weight-orthogonalized=[{fmt(per_loop_max(wo))}]")
    g = model.model.norm.weight.detach().float().cpu()
    gu = g * u
    print(f"    inter-loop norm gain: median|g|={g.abs().median():.3f} p99={g.abs().quantile(0.99):.3f} "
          f"max={g.abs().max():.3f}; cos(u, g*u)={float(torch.dot(u, gu) / gu.norm()):.3f}")


if __name__ == "__main__":
    main()
