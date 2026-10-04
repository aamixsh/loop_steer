#!/usr/bin/env python3
"""Why does weight orthogonalization fail to remove a direction from Ouro's residual stream?

Runs Ouro (HF) on real prompt+CoT token sequences under three conditions --
clean, hook ablation, in-place weight orthogonalization -- and measures how
much of a unit direction u is present at each stage of a decoder layer:

  resid_pre     residual stream entering a layer (what the refusal direction lives in)
  attn_in/out   input / output of input_layernorm_2   (norm applied to the attention output)
  mlp_in/out    input / output of post_attention_layernorm_2 (norm applied to the MLP output)
  loop_in/out   input / output of the final norm that runs after every loop

For each stage reports rms(<x, u>) (projection size) and rms(|x|) (vector length),
averaged over tokens and (loop, layer) call sites. If weight orthogonalization
worked as for a pre-norm model, attn_in/mlp_in projections would be ~0 (the
matrices cannot write along u) and stay ~0 after the norm. A non-zero *_out
projection when *_in is ~0 means the norm's per-dimension gain re-introduces u;
a smaller *_in length means the norm rescales the rest up.
"""

import argparse
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--direction", default="v4_baseline:t3.l16", help="method:site key into directions/train.pt")
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--max-cot-tokens", type=int, default=300)
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from loop_steer.paths import setup_job_env
    setup_job_env()
    import pandas as pd
    import torch

    from loop_steer import MODEL_ID
    from loop_steer.hooks import decoder_layers, hooked, ouro_ablation_hooks
    from loop_steer.models import load_model, load_tokenizer
    from loop_steer.ortho import orthogonalize_
    from loop_steer.paths import run_dir

    rd = run_dir(MODEL_ID)
    dirs = torch.load(rd / "directions" / "train.pt", weights_only=False)
    method, site = args.direction.split(":")
    loop, layer = (int(x) for x in site.replace("t", "").replace("l", "").split("."))
    raw = dirs[method]["dirs"][dirs["sites"].index((loop, layer))].float()
    u = raw / raw.norm()

    tok = load_tokenizer(MODEL_ID)
    model = load_model(MODEL_ID)
    u_dev = u.to(model.device)

    gen = pd.read_parquet(rd / "generations" / "select_clean.scored.parquet")
    gen = gen.drop_duplicates(["prompt_idx", "cot_rep"]).head(args.n_seqs)
    seqs = []
    for row in gen.itertuples():
        p_ids = tok.apply_chat_template([{"role": "user", "content": row.prompt}], add_generation_prompt=True,
                                        tokenize=True)
        p_ids = p_ids["input_ids"] if hasattr(p_ids, "keys") else p_ids
        c_ids = tok.encode(row.cot, add_special_tokens=False)[: args.max_cot_tokens]
        seqs.append(torch.tensor([list(p_ids) + c_ids]))

    layers = decoder_layers(model)
    state = {"loop": 0}
    stats = defaultdict(lambda: [0.0, 0.0, 0])  # (stage, loop) -> [sum proj^2, sum |x|^2, n_tokens]

    def record(stage, loop, x):
        x = x.detach().float()
        p = x @ u_dev
        s = stats[(stage, loop)]
        s[0] += float((p ** 2).sum())
        s[1] += float((x ** 2).sum(-1).sum())
        s[2] += x.shape[-2] * x.shape[0] if x.dim() == 3 else x.shape[0]

    def layer_pre(module, args_, kwargs):
        state["loop"] = kwargs.get("current_ut", 0)
        record("resid_pre", state["loop"], args_[0] if args_ else kwargs["hidden_states"])

    def norm_hook(name):
        def hook(module, inputs, output):
            record(f"{name}_in", state["loop"], inputs[0])
            record(f"{name}_out", state["loop"], output)
        return hook

    final_calls = {"n": 0}

    def final_norm_hook(module, inputs, output):
        loop = final_calls["n"] % model.config.total_ut_steps
        final_calls["n"] += 1
        record("loop_in", loop, inputs[0])
        record("loop_out", loop, output)

    def install():
        handles = [layer.register_forward_pre_hook(layer_pre, with_kwargs=True) for layer in layers]
        for layer in layers:
            handles.append(layer.input_layernorm_2.register_forward_hook(norm_hook("attn")))
            handles.append(layer.post_attention_layernorm_2.register_forward_hook(norm_hook("mlp")))
        handles.append(model.model.norm.register_forward_hook(final_norm_hook))
        return handles

    def run(label):
        stats.clear()
        final_calls["n"] = 0
        with torch.inference_mode(), hooked(install):
            for ids in seqs:
                model(input_ids=ids.to(model.device), use_cache=False)
        rows = []
        for (stage, loop), (sp, sx, n) in sorted(stats.items()):
            rows.append({"condition": label, "stage": stage, "loop": loop,
                         "rms_proj": (sp / n) ** 0.5, "rms_len": (sx / n) ** 0.5})
        return rows

    rows = run("clean")
    with hooked(ouro_ablation_hooks(model, u)):
        rows += run("hook")
    orthogonalize_(model, u)
    rows += run("ortho")
    orthogonalize_(model, None)

    df = pd.DataFrame(rows)
    df["proj_frac"] = df.rms_proj / df.rms_len  # projection as a fraction of vector length
    out = rd / "analysis" / f"ortho_diagnostic_{method}_t{loop}l{layer}.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"direction {method} at loop {loop} layer {layer}; {len(seqs)} sequences, "
          f"{sum(s.shape[1] for s in seqs)} tokens")
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        for stage in ("resid_pre", "attn_in", "attn_out", "mlp_in", "mlp_out", "loop_in", "loop_out"):
            sub = df[df.stage == stage]
            print(f"\n== {stage}: rms projection onto u  |  vector length  (per loop)")
            piv_p = sub.pivot(index="condition", columns="loop", values="rms_proj").reindex(["clean", "hook", "ortho"])
            piv_l = sub.pivot(index="condition", columns="loop", values="rms_len").reindex(["clean", "hook", "ortho"])
            print(pd.concat({"proj": piv_p, "len": piv_l}, axis=1).round(3).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
