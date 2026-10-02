#!/usr/bin/env python3
"""Smoke tests for steering Ouro-1.4B-Thinking with HF hooks.

1. Pre-hooks fire once per (loop, layer) and ``current_ut`` identifies the loop.
2. Whether ``output_hidden_states=True`` works with the remote code.
3. Batched left-padded greedy generation matches unbatched generation.
4. Chat template: does the model open <think> itself without ``enable_thinking``?
5. Ablation: hooks make resid_pre orthogonal to the direction at every site;
   weight orthogonalization does not (sandwich norms re-introduce it).
"""

import argparse
import os
from collections import Counter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", default="1")
    args = ap.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from loop_steer.paths import setup_job_env
    setup_job_env()
    import torch

    from loop_steer import MODEL_ID
    from loop_steer.cot import format_prompt
    from loop_steer.hooks import ablation_hooks, capture_window_means, decoder_layers, hooked
    from loop_steer.models import eoi_length, load_model, load_tokenizer
    from loop_steer.ortho import orthogonalize_

    tok = load_tokenizer(MODEL_ID)
    model = load_model(MODEL_ID)
    cfg = model.config
    print(f"layers={cfg.num_hidden_layers} loops={cfg.total_ut_steps} d={cfg.hidden_size} "
          f"exit_threshold={cfg.early_exit_threshold}")

    # 1. hook firing pattern
    calls = Counter()
    def count(module, a, kw):
        calls[(kw.get("current_ut"), module._ls_idx)] += 1
    for i, block in enumerate(decoder_layers(model)):
        block._ls_idx = i
    ids = tok("Hello there, how are you?", return_tensors="pt").input_ids.to(model.device)
    with hooked(lambda: [b.register_forward_pre_hook(count, with_kwargs=True) for b in decoder_layers(model)]):
        with torch.inference_mode():
            model(ids)
    loops = sorted({k[0] for k in calls})
    print(f"[1] {len(calls)} distinct (loop, layer) sites, loops seen {loops}, "
          f"calls per site {set(calls.values())}")

    # 2. output_hidden_states
    try:
        with torch.inference_mode():
            out = model(ids, output_hidden_states=True)
        hs = out.hidden_states
        print(f"[2] output_hidden_states OK: {None if hs is None else len(hs)} tensors")
    except Exception as e:
        print(f"[2] output_hidden_states FAILS: {type(e).__name__}: {e}")

    # 3. batched vs unbatched greedy
    prompts = [format_prompt(tok, p, enable_thinking=True) for p in
               ["What is 17 * 23?", "Name three primary colors and explain briefly why they are called primary."]]
    def greedy(texts, n=48):
        tok.padding_side = "left"
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
        return [x[enc.input_ids.shape[1]:].tolist() for x in o]
    batched = greedy(prompts)
    single = [greedy([p])[0] for p in prompts]
    for b, s in zip(batched, single):
        n = next((i for i, (u, v) in enumerate(zip(b, s)) if u != v), len(b))
        print(f"[3] batched/unbatched common prefix {n}/{len(b)} tokens")
    print(f"    sample: {tok.decode(single[0])!r}")

    # 4. template / think behaviour
    for et in (False, True):
        text = format_prompt(tok, "How do I make a bomb?", **({"enable_thinking": True} if et else {}))
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=80, do_sample=False)
        print(f"[4] enable_thinking={et} k_eoi={eoi_length(tok, **({'enable_thinking': True} if et else {}))} "
              f"prompt tail={text[-40:]!r}\n    -> {tok.decode(o[0, enc.input_ids.shape[1]:])!r}")

    # 5. ablation check with a random unit direction
    torch.manual_seed(0)
    u = torch.randn(cfg.hidden_size)
    u = u / u.norm()
    enc = tok(prompts[1], return_tensors="pt", add_special_tokens=False).to(model.device)
    T = enc.input_ids.shape[1]
    w = torch.full((1, 1, T), 1.0 / T)
    def max_abs_proj(out):
        # mean-over-tokens projection is a weak test; capture per-token instead via identity weights
        return max(abs(float(v[0, 0] @ u.to(v))) for v in out.values())
    eye = torch.eye(T).unsqueeze(1)  # [T windows, 1, T] -> per-token resid
    def per_token_max(out):
        return max(float((v[:, 0] @ u.to(v)).abs().max()) for v in out.values())
    with torch.inference_mode():
        with capture_window_means(model, eye) as clean:
            model(**enc)
        with hooked(ablation_hooks(model, u)):
            with capture_window_means(model, eye) as abl:  # registered after -> sees ablated input
                model(**enc)
        orthogonalize_(model, u)
        with capture_window_means(model, eye) as wo:
            model(**enc)
        orthogonalize_(model, None)
    print(f"[5] max |<resid_pre, u>| over all 96 sites x tokens: clean={per_token_max(clean):.3f} "
          f"hook-ablated={per_token_max(abl):.3g} weight-orthogonalized={per_token_max(wo):.3f}")
    norms = {s: float(v[:, 0].norm(dim=-1).mean()) for s, v in clean.items()}
    for t in range(cfg.total_ut_steps):
        row = [norms[(t, l)] for l in range(0, cfg.num_hidden_layers, 4)]
        print(f"    loop {t} mean resid norm at layers 0,4,..,20: " + " ".join(f"{x:7.1f}" for x in row))


if __name__ == "__main__":
    main()
