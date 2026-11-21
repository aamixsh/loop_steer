#!/usr/bin/env python3
"""Reference means for persona **mean ablation** (``persona_steer.py --mode ablate --ablate-ref noprompt``).

Projection ablation sets a trait's component to 0, which is "no trait" only if the model's own states sit near 0 along
the vector. On Inkling they often do not (``hallucinating``: about -32 unprompted, -15 when a system prompt elicits the
trait), so zeroing pushes toward the trait. Mean ablation instead sets the component to the model's unprompted mean:
one fixed value per site, the same for every evaluation question.

The means come from **held-out questions**: the model's own unprompted answers (no system prompt, the sampling of
``persona_steer.py``: T = 1.0, ``--n-samples`` per question, ``--max-tokens``) to the persona-vectors repo's 20
*extraction* questions per trait (``trait_data_extract``; disjoint from the 20 evaluation questions the ablation is
scored on), generated here and kept in ``persona/calib_noprompt[_loops<N>].parquet``. They are teacher-forced (eager)
and the mean projection on the trait's unit vector (the one ``persona_steer.py`` ablates: the last trained loop's site
at the tuned layer of ``persona_tune.json``, mid-loop without one) and on the seeded random control direction is
recorded over every token of prompt and response but position 0: Inkling (``loop_steer.vllm_inkling``) at every
visit and ablation hook point, the residual-stream models (``loop_steer.vllm_hooks``) at every layer input per loop and
Dory's final stream.

    scripts/persona_ablation_means.py --gpu 0 [--model thinkingmachines/Inkling-2-Mini] [--n-loops N]
-> directions/persona_ablation_means[_loops<N>].pt  {"means": {trait | "random:<trait>": Inkling [V, N_HOOK, MAX_ROWS],
   others [n_loops, n_layers]}, "finals" (Dory), ...}
"""
import argparse
import functools
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", default="thinkingmachines/Inkling-2-Mini")
    ap.add_argument("--n-loops", type=int, default=None, help="loops / passes to run (default: the trained count)")
    ap.add_argument("--ref", default=None, help="ignored (older chains pass it): the means come from held-out questions")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = ap.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"  # apply_model ships functions to the worker
    from loop_steer.paths import run_dir, setup_job_env
    setup_job_env()
    import pandas as pd
    import torch
    from vllm import LLM, SamplingParams

    from loop_steer.models import (DORY_ID, hf_config, is_inkling, load_tokenizer, loop_overrides, loop_unit_layers,
                                   n_loops, prepare_vllm, vllm_kwargs, vllm_model)
    from loop_steer.selfcorr import no_think_prompt

    inkling = is_inkling(args.model)
    if args.model == DORY_ID:
        from loop_steer.dory import register
        register()
    rd = run_dir(args.model)
    vecs = torch.load(rd / "directions" / "persona.pt", weights_only=False)
    tp = rd / "analysis" / "persona_tune.json"
    tuned = json.load(open(tp)) if tp.exists() else {}
    cfg = hf_config(args.model)
    T = n_loops(cfg)
    looped = T > 1
    unit = loop_unit_layers(cfg)
    sites = [tuple(s) for s in vecs["sites"]]
    traits = [t for t in vecs if t not in ("sites", "norms", "counts")]
    # as persona_steer.ablate_specs: the trait's vector at (last trained loop, tuned layer) (dense: (0, layer)); the
    # seeded random control; the default layer mid-loop (dense: DENSE_BLOCK * 2 + DENSE_BLOCK // 2 = 22)
    D = vecs[traits[0]]["dirs"].shape[-1]
    rand = torch.randn(D, generator=torch.Generator().manual_seed(0))
    mid = unit[len(unit) // 2] if looped else 22
    dirs = {}
    for t in traits:
        layer = (tuned.get(t) or {}).get("layer", mid)
        dirs[t] = (layer, vecs[t]["dirs"][sites.index((T - 1, layer) if looped else (0, layer))].float())

    from persona_vectors import keep_special

    from loop_steer.paths import DATA_ROOT

    src = DATA_ROOT / "external" / "pv_repo" / "data_generation"
    sfx = f"_loops{args.n_loops}" if args.n_loops else ""
    tok = load_tokenizer(args.model)
    prepare_vllm(args.model)
    llm = LLM(model=vllm_model(args.model), enforce_eager=True, enable_prefix_caching=False, seed=0,
              max_model_len=4096, gpu_memory_utilization=args.gpu_memory_utilization,
              hf_overrides=loop_overrides(args.model, args.n_loops), **vllm_kwargs(args.model))
    gen = rd / "persona" / f"calib_noprompt{sfx}.parquet"
    if gen.exists():
        df = pd.read_parquet(gen)
    else:  # unprompted answers to the held-out extraction questions, sampled as persona_steer.py
        sp = SamplingParams(temperature=1.0, max_tokens=args.max_tokens, n=args.n_samples, seed=0,
                            skip_special_tokens=not keep_special(args.model))
        rows = []
        for t in traits:
            qs = json.load(open(src / "trait_data_extract" / f"{t}.json"))["questions"]
            ids = [{"prompt_token_ids": tok(no_think_prompt(tok, [{"role": "user", "content": q}]),
                                            add_special_tokens=False).input_ids} for q in qs]
            for qi, (q, r) in enumerate(zip(qs, llm.generate(ids, sp, use_tqdm=False))):
                rows += [{"trait": t, "q": qi, "question": q, "sample": k, "response": o.text}
                         for k, o in enumerate(r.outputs)]
            print(f"  {t}: {len(qs)} held-out extraction questions x {args.n_samples} unprompted answers", flush=True)
        df = pd.DataFrame(rows)
        df.to_parquet(gen)
        print(f"-> {gen}", flush=True)
    if inkling:
        from loop_steer import vllm_inkling as hooks
    else:
        from loop_steer import vllm_hooks as hooks
    means, finals = {}, {}
    for t in traits:
        sub = df[df.trait == t]
        if sub.empty:
            print(f"  {t}: no held-out answers; skipped", flush=True)
            continue
        ids = [tok(no_think_prompt(tok, [{"role": "user", "content": r.question}]) + r.response,
                   add_special_tokens=False).input_ids[:4000] for r in sub.itertuples()]
        layer, v = dirs[t]
        llm.apply_model(functools.partial(hooks.install, specs=[{"kind": "calib", "dirs": torch.stack([v, rand])}]))
        llm.generate([{"prompt_token_ids": x} for x in ids], SamplingParams(max_tokens=1, temperature=0.0),
                     use_tqdm=False)
        if inkling:
            m = llm.apply_model(hooks.fetch_calib)[0]  # [2, V, N_HOOK, MAX_ROWS]
            means[t], means[f"random:{t}"] = m[0], m[1]
            blk, n_core, c0 = hooks.H["block"], len(unit), unit[0]  # core visits: c0 + pass * n_core + offset
            show = [float(m[0, c0 + p * n_core: c0 + (p + 1) * n_core, blk, 0].mean()) for p in range(args.n_loops or T)]
        else:
            m = llm.apply_model(hooks.fetch_calib)[0]  # ref [2, n_loops, n_layers]
            means[t], means[f"random:{t}"] = m["ref"][0], m["ref"][1]
            if m["ref_final"] is not None:
                finals[t], finals[f"random:{t}"] = float(m["ref_final"][0]), float(m["ref_final"][1])
            show = [float(m["ref"][0, p, layer]) for p in range(m["ref"].shape[1])]
        print(f"  {t} (layer {layer}): {len(ids)} responses; mean projection at layer {layer}'s input per loop "
              f"(Inkling: core block writes per pass): " + ", ".join(f"{x:.2f}" for x in show), flush=True)
    llm.apply_model(hooks.clear)
    out = rd / "directions" / f"persona_ablation_means{sfx}.pt"
    torch.save({"means": means, "finals": finals, "backend": "vllm_inkling" if inkling else "vllm_hooks",
                "ref": f"held-out extraction questions, unprompted ({gen.name})",
                "layers": {t: dirs[t][0] for t in traits}, "n_loops": args.n_loops or T}, out)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
