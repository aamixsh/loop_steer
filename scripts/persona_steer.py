#!/usr/bin/env python3
"""Persona-vector steering restricted to loops (dense: quarters), with a coherence-preserving strength per trait.

    # 1. tune: every loop (dense: every quarter) at once, strength x site grid -> analysis/persona_tune.{csv,json}
    scripts/persona_steer.py --gpu 0 --model ByteDance/Ouro-1.4B-Thinking --mode tune
    # 2. loops: the tuned (site, strength) per trait, in single loops / sets of loops
    scripts/persona_steer.py --gpu 0 --model ByteDance/Ouro-1.4B-Thinking --mode loops \\
        --sets 1=0 2=1 3=2 4=3 all=0,1,2,3
    scripts/persona_steer.py --gpu 0 --model nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 --mode loops --n-loops 4 \\
        --sets 1=0 2=1 3=2 4=3 12=0,1 all=0,1,2,3
    scripts/persona_steer.py --gpu 0 --model Qwen/Qwen3-8B --mode loops --sets q1=0 q2=1 q3=2 q4=3 all=0,1,2,3
    # 3. ablate: elicit the trait with its system prompts, project its vector out of single loops / sets of loops
    scripts/persona_steer.py --gpu 0 --model Nanbeige/Nanbeige4.2-3B --mode ablate --sets 1=0 2=1 all=0,1

Vectors are those of ``scripts/persona_vectors.py --stage extract`` (``directions/persona.pt``: mean answer-token
activation with the trait minus without it, at every site). One vector per trait, from ONE site, is used everywhere:
looped models take the last trained loop's vector at layer l and add it at layer l in each selected loop; dense models
(4 quarters of 9 layers as "loops") take the vector at layer l and add it at the same offset in each selected quarter
(layers l % 9 + 9 q). The added vector is unit(v) * c * (median stream norm where it is added) (norms from persona.pt;
a loop beyond the trained ones uses the last trained loop's norm), i.e. the same fraction c of the stream everywhere.

tune   clean + every loop (quarter) at once, for each candidate site layer (looped: 0.25 / 0.5 / 0.75 of a loop;
       dense: ``--dense-layers``) x c in ``--coeffs``; ``--n-eval`` samples x 20 evaluation questions per trait.
       Selection per trait: the (layer, c) with the largest trait gain over clean among those whose mean coherence
       stays >= clean coherence - ``--max-coh-drop``; ties go to the smaller c; ``steerable`` = gain >= 10 points.
loops  the tuned (layer, c) of each steerable trait, added in each set of ``--sets`` (name=loop,loop,..; 0-based;
       dense: quarters), plus clean. ``--n-loops`` runs a looped model with more loops (same vectors and c).
ablate the trait elicited by the persona-vectors repo's trait-eliciting system prompts (``trait_data_extract``: the
       ``pos`` instructions the vectors were extracted with; evaluation question i gets instruction i mod 5), every
       trait with a vector (steerable or not). Conditions: ``noprompt`` (no system prompt, no hook), ``prompted``
       (system prompt, no hook), ``ablate_<set>`` (system prompt; the trait's unit vector, from the last trained
       loop's site at the tuned layer of ``persona_tune.json`` or mid-loop without one, projected out of every layer
       input and residual write in the set's loops, as the refusal ablation; dense: every layer of the set's quarters;
       a set of every loop also covers Dory's input / output sublayers) and ``random_all`` (a seeded random unit
       direction ablated in every loop: control). Summary: ``removed`` = (prompted - cond) / (prompted - noprompt)
       trait score, ``coh_drop`` against prompted.
Every mode generates (vLLM, thinking off as persona_vectors.py), judges trait and coherence with gpt-oss-20b (the
repo's prompts), and summarises: ``persona/steer_<name>.judged.parquet``, ``analysis/persona_<name>.csv``; name =
``tune``, ``loops`` or ``ablate`` (+ ``_loops<N>`` with ``--n-loops``). Skip-existing per stage.
"""

import argparse
import functools
import json
import os

DENSE_BLOCK = 9  # Qwen3-8B: 36 layers = 4 quarters


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["tune", "loops", "ablate"], required=True)
    ap.add_argument("--traits", default=None, help="comma-separated (default: every trait with a vector)")
    ap.add_argument("--coeffs", default="0.01,0.025,0.05,0.1,0.15,0.2,0.3")
    ap.add_argument("--fracs", default="0.25,0.5,0.75", help="tune, looped: candidate layers as fractions of a loop")
    ap.add_argument("--dense-layers", default="12,16,21", help="tune, dense: candidate site layers")
    ap.add_argument("--max-coh-drop", type=float, default=10.0)
    ap.add_argument("--sets", nargs="+", default=None, help="loops, ablate: name=t,t,.. (0-based loops or quarters)")
    ap.add_argument("--n-loops", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=None, help="samples per question (default tune 3, loops 5)")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--stages", default="steer,judge,summary")
    ap.add_argument("--tag", default=None, help="suffix of every output name (and of the tuning file loops reads)")
    args = ap.parse_args()
    if "," in args.stages:  # one process per stage: the generation engine must release the GPU before the judge loads
        import subprocess
        import sys

        argv, skip = [], False
        for x in sys.argv[1:]:
            if skip:
                skip = False
            elif x == "--stages":
                skip = True
            elif not x.startswith("--stages="):
                argv.append(x)
        for st in args.stages.split(","):
            subprocess.run([sys.executable, os.path.abspath(__file__), *argv, "--stages", st], check=True)
        return

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from loop_steer.paths import DATA_ROOT, run_dir, setup_job_env
    setup_job_env()
    import pandas as pd
    import torch
    from persona_vectors import COHERENCE, answer_of, parse_score

    rd = run_dir(args.model)
    wd = rd / "persona"
    sfx = f"_{args.tag}" if args.tag else ""
    name = args.mode + (f"_loops{args.n_loops}" if args.n_loops else "") + sfx
    gen_path, judged_path = wd / f"steer_{name}.parquet", wd / f"steer_{name}.judged.parquet"
    src = DATA_ROOT / "external" / "pv_repo" / "data_generation"
    vecs = torch.load(rd / "directions" / "persona.pt", weights_only=False)
    traits = args.traits.split(",") if args.traits else [t for t in vecs if t not in ("sites", "norms", "counts")]
    traits = [t for t in traits if t in vecs]
    evalq = {t: json.load(open(src / "trait_data_eval" / f"{t}.json")) for t in traits}
    ablate = args.mode == "ablate"
    ref_cond = "prompted" if ablate else "clean"
    stages = args.stages.split(",")
    n_eval = args.n_eval or (3 if args.mode == "tune" else 5)

    if "steer" in stages and not gen_path.exists():
        from transformers import AutoConfig
        from vllm import LLM, SamplingParams

        from loop_steer import vllm_hooks
        from loop_steer.models import (DORY_ID, _remote_kwargs, load_tokenizer, loop_overrides, n_loops,
                                       prepare_vllm, vllm_kwargs, vllm_model)
        from loop_steer.selfcorr import no_think_prompt

        if args.model == DORY_ID:
            from loop_steer.dory import register
            register()
        cfg = AutoConfig.from_pretrained(args.model, **_remote_kwargs(args.model))
        T = n_loops(cfg)  # trained loops: the sites of persona.pt
        looped = T > 1
        T_run = (args.n_loops or T) if looped else 4
        sites = vecs["sites"]
        norm = {tuple(s): float(x) for s, x in zip(sites, vecs["norms"])}
        if getattr(cfg, "model_type", None) == "dory":
            unit = list(range(cfg.n_input_layers, cfg.n_input_layers + cfg.n_recurrent_layers))
        else:
            unit = list(range(cfg.num_hidden_layers))

        def specs_for(trait, layer, c, units):
            """Hook specs: trait vector from the site at ``layer`` added in loops / quarters ``units``."""
            v = vecs[trait]["dirs"][sites.index((T - 1, layer) if looped else (0, layer))].float()
            v = v / v.norm()
            out = []
            for u in units:
                if looped:
                    t, l, loops = u, layer, [u]
                    m = norm[(min(t, T - 1), l)]
                else:
                    t, l, loops = 0, layer % DENSE_BLOCK + DENSE_BLOCK * u, None
                    m = norm[(0, l)]
                out.append({"kind": "actadd", "vec": v * m, "coeff": c, "loops": loops, "layer": l, "positions": None})
            return out

        def ablate_specs(trait, layer, units):
            """Hook specs: the trait's vector (``trait="random"``: a seeded random direction) projected out in
            ``units`` (loops; dense: quarters); every loop -> loops=None, which adds Dory's input / output sublayers."""
            v = vecs[next(iter(traits))]["dirs"][0] if trait == "random" else \
                vecs[trait]["dirs"][sites.index((T - 1, layer) if looped else (0, layer))]
            if trait == "random":
                v = torch.randn(v.shape[-1], generator=torch.Generator().manual_seed(0))
            spec = {"kind": "ablate", "dir": v.float(), "positions": None, "loops": None}
            if looped:
                if sorted(units) != list(range(T_run)):
                    spec["loops"] = list(units)
            elif sorted(units) != list(range(4)):
                spec["layers"] = [l for u in units for l in range(DENSE_BLOCK * u, DENSE_BLOCK * (u + 1))]
            return [spec]

        conds = [("clean", -1, 0.0, [])]
        if args.mode == "tune":
            if looped:
                layers = sorted({unit[min(int(round(float(f) * len(unit))), len(unit) - 1)]
                                 for f in args.fracs.split(",")})
            else:
                layers = [int(x) for x in args.dense_layers.split(",")]
            conds += [("all", l, float(c), list(range(T_run))) for l in layers for c in args.coeffs.split(",")]
            per_trait = {t: conds for t in traits}
        elif ablate:
            tp = rd / "analysis" / f"persona_tune{sfx}.json"
            tuned = json.load(open(tp)) if tp.exists() else {}
            mid = unit[len(unit) // 2] if looped else DENSE_BLOCK * 2 + DENSE_BLOCK // 2
            sets = []
            for item in args.sets or [f"all={','.join(map(str, range(T_run)))}"]:
                nm, _, ts = item.partition("=")
                sets.append((nm, sorted({int(x) % T_run for x in ts.split(",")})))
            per_trait = {}
            for t in traits:
                layer = (tuned.get(t) or {}).get("layer", mid)
                per_trait[t] = ([("noprompt", -1, 0.0, []), ("prompted", -1, 0.0, [])]
                                + [(f"ablate_{nm}", layer, 0.0, us) for nm, us in sets]
                                + [("random_all", layer, 0.0, list(range(T_run)))])
        else:
            tuned = json.load(open(rd / "analysis" / f"persona_tune{sfx}.json"))
            sets = []
            for item in args.sets or []:
                nm, _, ts = item.partition("=")
                sets.append((nm, sorted({int(x) % T_run for x in ts.split(",")})))
            per_trait = {}
            for t in traits:
                sel = tuned.get(t)
                if not sel or not sel["steerable"]:
                    print(f"  {t}: not steerable in the tuning run; skipped", flush=True)
                    continue
                per_trait[t] = conds + [(nm, sel["layer"], sel["c"], us) for nm, us in sets]

        if not per_trait:
            raise SystemExit("no trait to steer (none steerable in the tuning file?)")
        tok = load_tokenizer(args.model)
        prepare_vllm(args.model)
        llm = LLM(model=vllm_model(args.model), seed=0, gpu_memory_utilization=args.gpu_memory_utilization,
                  max_model_len=4096, enforce_eager=True, hf_overrides=loop_overrides(args.model, args.n_loops),
                  **vllm_kwargs(args.model))
        sp = SamplingParams(temperature=1.0, max_tokens=args.max_tokens, n=n_eval, skip_special_tokens=True, seed=0)
        rows = []
        for t, tconds in per_trait.items():
            qs = evalq[t]["questions"]
            ids = [{"prompt_token_ids": tok(no_think_prompt(tok, [{"role": "user", "content": q}]),
                                            add_special_tokens=False).input_ids} for q in qs]
            if ablate:  # question i under the trait's i-th (mod 5) eliciting system prompt
                inst = json.load(open(src / "trait_data_extract" / f"{t}.json"))["instruction"]
                sys_of = [inst[qi % len(inst)]["pos"] for qi in range(len(qs))]
                ids_sys = [{"prompt_token_ids": tok(no_think_prompt(tok, [{"role": "system", "content": sy},
                                                                          {"role": "user", "content": q}]),
                                                    add_special_tokens=False).input_ids} for q, sy in zip(qs, sys_of)]
            for cname, layer, c, units in tconds:
                if ablate:
                    specs = (ablate_specs("random" if cname.startswith("random") else t, layer, units)
                             if units else [])
                else:
                    specs = specs_for(t, layer, c, units) if units else []
                system = ablate and cname != "noprompt"
                llm.apply_model(functools.partial(vllm_hooks.install, specs=specs))
                llm.reset_prefix_cache()
                outs = llm.generate(ids_sys if system else ids, sp, use_tqdm=False)
                for qi, (q, r) in enumerate(zip(qs, outs)):
                    for s, o in enumerate(r.outputs):
                        rows.append({"trait": t, "cond": cname, "units": ",".join(map(str, units)), "layer": layer,
                                     "c": c, "q": qi, "question": q, "sample": s, "response": o.text,
                                     "system": sys_of[qi] if system else ""})
                print(f"  {t} {cname} layer {layer} c={c}: done", flush=True)
            pd.DataFrame(rows).to_parquet(gen_path.with_suffix(".partial.parquet"))
        llm.apply_model(vllm_hooks.clear)
        del llm
        pd.DataFrame(rows).to_parquet(gen_path)
        print(f"-> {gen_path}", flush=True)

    if "judge" in stages and not judged_path.exists():
        from loop_steer.judge import chat_offline, load_offline_judge

        df = pd.read_parquet(gen_path)
        llm = load_offline_judge(gpu_memory_utilization=0.85, max_model_len=8192)
        ans = [answer_of(r)[:6000] for r in df.response]
        tp = [evalq[t]["eval_prompt"].format(question=q, answer=a) for t, q, a in zip(df.trait, df.question, ans)]
        cp = [COHERENCE.format(question=q, answer=a) for q, a in zip(df.question, ans)]
        scores = chat_offline(tp + cp, llm, max_tokens=1024, parse=parse_score)
        df["trait_score"], df["coherence"] = scores[: len(df)], scores[len(df):]
        df.to_parquet(judged_path)
        print(f"-> {judged_path}", flush=True)

    if "summary" in stages:
        df = pd.read_parquet(judged_path)
        g = df.groupby(["trait", "cond", "units", "layer", "c"]).agg(
            trait_score=("trait_score", "mean"), coherence=("coherence", "mean"),
            unscored=("trait_score", lambda s: s.isna().mean()), n=("trait_score", "size")).reset_index()
        clean = g[g.cond == ref_cond].set_index("trait")
        g["trait_clean"], g["coherence_clean"] = g.trait.map(clean.trait_score), g.trait.map(clean.coherence)
        g["gain"], g["coh_drop"] = g.trait_score - g.trait_clean, g.coherence_clean - g.coherence
        if ablate:  # trait_clean / coherence_clean are the prompted (unablated) run's
            base = g[g.cond == "noprompt"].set_index("trait").trait_score
            g["trait_noprompt"] = g.trait.map(base)
            g["removed"] = (g.trait_clean - g.trait_score) / (g.trait_clean - g.trait_noprompt)
        (rd / "analysis").mkdir(exist_ok=True)
        g.to_csv(rd / "analysis" / f"persona_{name}.csv", index=False)
        with pd.option_context("display.width", 220, "display.max_rows", 500):
            print(g.round(1).to_string())
        if args.mode == "tune":
            sel = {}
            for t, gt in g[g.cond != "clean"].groupby("trait"):
                ok = gt[gt.coh_drop <= args.max_coh_drop].sort_values(["gain", "c"], ascending=[False, True])
                if len(ok):
                    b = ok.iloc[0]
                    sel[t] = {"layer": int(b.layer), "c": float(b.c), "gain": float(b.gain),
                              "coh_drop": float(b.coh_drop), "trait_clean": float(b.trait_clean),
                              "coherence_clean": float(b.coherence_clean), "steerable": bool(b.gain >= 10)}
                else:
                    sel[t] = {"steerable": False}
            path = rd / "analysis" / f"persona_tune{sfx}.json"
            json.dump({"rule": f"max gain with coherence drop <= {args.max_coh_drop}; steerable = gain >= 10",
                       **sel}, open(path, "w"), indent=1)
            print(json.dumps(sel, indent=1))
            print(f"-> {path}")


if __name__ == "__main__":
    main()
