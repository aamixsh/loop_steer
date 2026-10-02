#!/usr/bin/env python3
"""Two-stage generation under residual-stream interventions, many candidates per vLLM engine.

Candidate syntax: ``kind:method:site[:opt=val...]`` or ``none``.
  kind    ortho  (project direction out of weights; pre-norm models only, e.g. Qwen)
          ablate (hooks on every residual write; works for Ouro)
          actadd (add c * raw mean-diff vector at resid_pre of the site's layer)
  method  key in directions/<file>.pt (v4_cot, v12_cot150, v4_baseline, paired_cot, ...)
  site    l<layer> or t<loop>.l<layer> -- where the direction was extracted
  opts    c=<coeff>      actadd coefficient (default -1.0: subtract the refusal direction)
          apply=<t,t..>  loops to intervene in (default: all loops)
          layer=<l>      actadd at a different layer than the extraction site
Examples: ortho:v4_cot:l23   actadd:v12_cot150:l17:c=-1.5   ablate:v4_cot:t3.l12:apply=3

Weights are restored / hooks removed between candidates.
"""

import argparse
import functools
import os
import re


def parse_candidate(text):
    if text == "none":
        return {"name": "none", "kind": "none"}
    kind, method, site, *opts = text.split(":")
    m = re.fullmatch(r"(?:t(\d+)\.)?l(\d+)", site)
    if m is None:
        raise ValueError(f"bad site {site!r}")
    cand = {"name": text.replace(":", "_").replace("=", "").replace(",", "-"), "kind": kind, "method": method,
            "site": (int(m.group(1) or 0), int(m.group(2))), "coeff": -1.0, "apply": None, "layer": None}
    for opt in opts:
        key, val = opt.split("=")
        if key == "c":
            cand["coeff"] = float(val)
        elif key == "apply":
            cand["apply"] = [int(x) for x in val.split(",")]
        elif key == "layer":
            cand["layer"] = int(val)
        else:
            raise ValueError(f"unknown option {key!r}")
    return cand


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", required=True)
    ap.add_argument("--directions", required=True)
    ap.add_argument("--candidates", required=True, nargs="+", help="Space-separated candidate specs")
    ap.add_argument("--split", default="train")
    ap.add_argument("--kind", default="harmful")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--cot-reps", type=int, default=3)
    ap.add_argument("--out-reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--tag", required=True, help="Output subdir under generations/")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    cands = [parse_candidate(c) for c in args.candidates]
    uses_hooks = any(c["kind"] in ("ablate", "actadd") for c in cands)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"  # apply_model ships functions to the worker
    from loop_steer.paths import run_dir, setup_job_env
    setup_job_env()
    import torch
    from vllm import LLM, SamplingParams

    from loop_steer import vllm_hooks
    from loop_steer.cot import load_prompts
    from loop_steer.models import load_tokenizer, vllm_kwargs
    from loop_steer.ortho import orthogonalize_
    from loop_steer.sampling import two_stage

    rd = run_dir(args.model)
    dirs = torch.load(rd / "directions" / f"{args.directions}.pt", weights_only=False)
    site_index = {site: i for i, site in enumerate(dirs["sites"])}
    out_dir = rd / "generations" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.split, args.kind)[args.offset:]
    if args.n_prompts is not None:
        prompts = prompts[: args.n_prompts]

    tok = load_tokenizer(args.model)
    llm = LLM(model=args.model, seed=args.seed, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=8192, enforce_eager=uses_hooks, **vllm_kwargs(args.model))
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, skip_special_tokens=False)
    weights_dirty = False

    for c in cands:
        out = out_dir / f"{c['name']}.parquet"
        if args.skip_existing and out.exists():
            continue
        vec = None if c["kind"] == "none" else dirs[c["method"]]["dirs"][site_index[c["site"]]]
        if weights_dirty or c["kind"] == "ortho":
            llm.apply_model(functools.partial(orthogonalize_, direction=vec if c["kind"] == "ortho" else None))
            weights_dirty = c["kind"] == "ortho"
        specs = []
        if c["kind"] == "ablate":
            specs = [{"kind": "ablate", "dir": vec, "loops": c["apply"]}]
        elif c["kind"] == "actadd":
            specs = [{"kind": "actadd", "vec": vec, "coeff": c["coeff"], "loops": c["apply"],
                      "layer": c["layer"] if c["layer"] is not None else c["site"][1]}]
        if uses_hooks:
            llm.apply_model(functools.partial(vllm_hooks.install, specs=specs))
        llm.reset_prefix_cache()
        df = two_stage(llm, tok, prompts, cot_reps=args.cot_reps, out_reps=args.out_reps,
                       sampling=sampling, prompt_offset=args.offset)
        df["candidate"] = c["name"]
        df["n_cot_total"], df["n_invalid_cot"] = df.attrs["n_cot_total"], df.attrs["n_invalid_cot"]
        invalid = df.attrs.pop("invalid")
        if len(invalid):  # kept outside the scored dir's *.parquet glob
            (out.parent / "invalid").mkdir(exist_ok=True)
            invalid.to_parquet(out.parent / "invalid" / out.name)
        df.to_parquet(out)
        print(f"[{c['name']}] {len(df)} rows, invalid CoTs {df.attrs['n_invalid_cot']}/{df.attrs['n_cot_total']}"
              f" -> {out}", flush=True)


if __name__ == "__main__":
    main()
