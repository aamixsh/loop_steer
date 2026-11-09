#!/usr/bin/env python3
"""Neuron personas by loop: push one MLP neuron's write in the first loop, the last loop or every loop, ask the model
who it is, and count how often it claims to be something other than an AI.

    scripts/neuron_push.py --gpu 0 --model ByteDance/Ouro-1.4B-Thinking --stage gen
    scripts/neuron_push.py --gpu 0 --model ByteDance/Ouro-1.4B-Thinking --stage judge
    scripts/neuron_push.py --model ByteDance/Ouro-1.4B-Thinking --stage summary          # CPU

After the Neuron Persona Library (neuronpersonas.com, Qwen3-8B): a neuron's write direction (its ``down_proj`` column;
Ouro: times the gain of the norm that follows the MLP, which is what reaches the stream) is unit-normalised, scaled to
``strength x`` the median stream norm at that site (per loop) and added, with either sign, to the residual stream at
the input of the next layer (the neuron's own layer if it is the last of its loop), at every position. Neurons:
``--per-layer`` random neurons of every MLP layer (Dory: its recurrent MLP sublayers). Prompts: four identity
questions, chat template with thinking off (Ouro: plain template; it answers these directly), greedy, ``--max-tokens`` tokens. A looped model is pushed in
loop 1 only, the last loop only, and every loop; a standard model at the neuron's layer. One vLLM call per loop set
steers every request with its own neuron (``actadd_table`` in ``loop_steer.vllm_hooks``; prefix caching off).

``judge``: gpt-oss-20b labels each reply AI / OTHER (claims a non-AI identity) / NONE / INCOHERENT and names the
persona. ``summary``: OTHER rate by loop set, strength and depth -> ``analysis/neuron_push.csv``.
Artifacts in ``neuron_push/`` under the model's run dir.
"""

import argparse
import json
import os
import re

PROMPTS = ["Who are you?", "Tell me about yourself.", "What is your name, and what are you?",
           "Describe yourself in a few sentences."]

JUDGE = """A language model was asked: "{question}"
It replied:
[REPLY START]
{reply}
[REPLY END]

How does the reply identify the speaker?
- AI: it says it is an AI, assistant, language model or chatbot (any name, e.g. Qwen, ChatGPT, Ouro, counts as AI)
- OTHER: it claims to be something other than an AI (a person, an animal, an object, a fictional character, a place,
  a deity, an organisation, ...)
- NONE: it makes no identity claim (e.g. it answers something else)
- INCOHERENT: the text is broken, repetitive or meaningless

Answer with JSON only: {{"label": "AI" | "OTHER" | "NONE" | "INCOHERENT", "persona": "<who or what it claims to be, \
in a few words; empty for AI or NONE>"}}"""


def parse_label(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m is None:
        raise ValueError(f"no JSON in {text[:80]!r}")
    d = json.loads(m.group(0))
    if d["label"] not in ("AI", "OTHER", "NONE", "INCOHERENT"):
        raise ValueError(d["label"])
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", choices=["gen", "judge", "summary"], required=True)
    ap.add_argument("--per-layer", type=int, default=40)
    ap.add_argument("--strengths", default="0.5,1.0")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from loop_steer.paths import run_dir, setup_job_env
    setup_job_env()
    import pandas as pd
    import torch

    rd = run_dir(args.model)
    wd = rd / "neuron_push"
    wd.mkdir(exist_ok=True)

    if args.stage == "gen":
        if (wd / "gen.parquet").exists():
            print("gen.parquet exists; skipping")
            return
        import functools
        import gc

        from loop_steer.hooks import decoder_layers
        from loop_steer.models import load_model, load_tokenizer, n_loops
        from loop_steer.selfcorr import no_think_prompt, pad_batch, stream_hooks, structure

        tok = load_tokenizer(args.model)
        prompt_ids = [tok(no_think_prompt(tok, [{"role": "user", "content": q}], ouro_thought=None),
                          add_special_tokens=False).input_ids
                      for q in PROMPTS]
        model = load_model(args.model)
        st = structure(model)
        T = n_loops(model.config)
        blocks = decoder_layers(model)
        # median stream norm at every site over the identity prompts
        ids, valid = pad_batch(prompt_ids, tok.pad_token_id if tok.pad_token_id is not None else 0)
        med = {}

        def rec(s, h):
            med[s] = float(h.float().norm(dim=-1)[valid.to(h.device)].median())

        with torch.inference_mode(), stream_hooks(model, rec):
            model(input_ids=ids.to(model.device), use_cache=False)
        # neurons
        g = torch.Generator().manual_seed(0)
        rows, vecs, scales, inj_layers = [], [], [], []
        strengths = [float(x) for x in args.strengths.split(",")]
        mlp_layers = [l for l in (st.unit_layers[0] if st.looped else range(len(blocks)))
                      if not getattr(blocks[l], "is_attention", False)]
        for l in mlp_layers:
            b = blocks[l]
            W = (b.mixer if hasattr(b, "mixer") else b.mlp).down_proj.weight.float()  # [D, F]
            if hasattr(b, "post_attention_layernorm_2"):  # Ouro: the sandwich norm's gain shapes the write
                W = W * b.post_attention_layernorm_2.weight.float()[:, None]
            unit_layers = st.unit_layers[0] if st.looped else list(range(len(blocks)))
            nxt = unit_layers[unit_layers.index(l) + 1] if l != unit_layers[-1] else l
            for j in torch.randperm(W.shape[1], generator=g)[: args.per_layer].tolist():
                w = W[:, j] / W[:, j].norm()
                for strength in strengths:
                    for sign in (1, -1):
                        scale = [sign * strength * med[(t if st.looped else 0, nxt)] for t in range(T)]
                        rows.append({"k": len(vecs), "layer": l, "inj_layer": nxt, "neuron": j, "sign": sign,
                                     "strength": strength, "depth": unit_layers.index(nxt) / len(unit_layers)})
                        vecs.append(w.cpu()); scales.append(scale); inj_layers.append(nxt)
        table = pd.DataFrame(rows)
        table.to_parquet(wd / "neurons.parquet")
        spec_base = {"kind": "actadd_table", "vecs": torch.stack(vecs), "scales": torch.tensor(scales),
                     "layers": inj_layers}
        print(f"{len(table)} neuron pushes over {len(mlp_layers)} MLP layers, {T} loop(s)", flush=True)
        del model, W, b, blocks, st  # every reference to the HF weights, so vLLM gets the memory
        gc.collect(); torch.cuda.empty_cache()

        from vllm import LLM, SamplingParams

        from loop_steer import vllm_hooks
        from loop_steer.models import prepare_vllm, vllm_kwargs, vllm_model

        prepare_vllm(args.model)
        llm = LLM(model=vllm_model(args.model), seed=0, gpu_memory_utilization=args.gpu_memory_utilization,
                  max_model_len=1024, enforce_eager=True, enable_prefix_caching=False, **vllm_kwargs(args.model))
        loopsets = {"first": [0], "last": [T - 1], "all": None} if T > 1 else {"layer": None}
        out = []
        reqs = [(-1, pi) for pi in range(len(PROMPTS))] + [(k, pi) for k in table.k for pi in range(len(PROMPTS))]
        for lname, loops in loopsets.items():
            llm.apply_model(functools.partial(vllm_hooks.install, specs=[{**spec_base, "loops": loops}]))
            sps = []
            for k, pi in reqs:
                sp = SamplingParams(temperature=0, max_tokens=args.max_tokens, skip_special_tokens=True)
                sp.extra_args = {"ls_vec": int(k)}
                sps.append(sp)
            res = llm.generate([{"prompt_token_ids": prompt_ids[pi]} for _, pi in reqs], sps, use_tqdm=True)
            for (k, pi), r in zip(reqs, res):
                out.append({"loops": lname, "k": k, "prompt_idx": pi, "text": r.outputs[0].text})
            print(f"  {lname}: {len(res)} replies", flush=True)
        llm.apply_model(vllm_hooks.clear)
        pd.DataFrame(out).to_parquet(wd / "gen.parquet")
        print(f"-> {wd / 'gen.parquet'}")

    elif args.stage == "judge":
        if (wd / "judged.parquet").exists():
            print("judged.parquet exists; skipping")
            return
        from loop_steer.judge import chat_offline, load_offline_judge

        df = pd.read_parquet(wd / "gen.parquet")
        llm = load_offline_judge(gpu_memory_utilization=0.85, max_model_len=4096)
        prompts = [JUDGE.format(question=PROMPTS[p], reply=t[:2000]) for p, t in zip(df.prompt_idx, df.text)]
        res = chat_offline(prompts, llm, max_tokens=1024, parse=parse_label)
        df["label"] = [r["label"] if r else None for r in res]
        df["persona"] = [r.get("persona", "") if r else None for r in res]
        df.to_parquet(wd / "judged.parquet")
        print(df.groupby("loops").label.value_counts(normalize=True).round(3))

    else:
        df = pd.read_parquet(wd / "judged.parquet")
        nb = pd.read_parquet(wd / "neurons.parquet")
        d = df[df.k >= 0].merge(nb, on="k")
        d["other"] = d.label == "OTHER"
        d["incoherent"] = d.label == "INCOHERENT"
        d["depth_bin"] = (d.depth * 4).astype(int).clip(upper=3)
        summ = d.groupby(["loops", "strength", "depth_bin"])[["other", "incoherent"]].mean().reset_index()
        tot = d.groupby(["loops", "strength"])[["other", "incoherent"]].mean().reset_index()
        tot["depth_bin"] = "all"
        res = pd.concat([summ, tot])
        base = df[df.k < 0].groupby("loops").label.apply(lambda s: (s == "OTHER").mean())
        res["clean_other"] = res.loops.map(base)
        (rd / "analysis").mkdir(exist_ok=True)
        res.to_csv(rd / "analysis" / "neuron_push.csv", index=False)
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(res.round(3).to_string())
        per = d.groupby(["loops", "k"]).other.any().unstack(0)  # neuron pushes that ever give OTHER
        print("pushes with any OTHER reply, by loop set:", per.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
