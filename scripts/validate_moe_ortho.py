#!/usr/bin/env python3
"""Weight orthogonalization on an MoE model (gpt-oss-20b): is it an exact ablation?

    scripts/validate_moe_ortho.py --gpu 0 [--direction v4_baseline:l15]   (random direction if omitted)

gpt-oss is pre-norm and adds every write to the residual unnormalized, so projecting ``u`` out of all writers
(embedding, every ``o_proj`` and every expert's down projection, *and their biases*) should equal hook ablation for
any routing: the MoE output is a router-weighted sum of expert outputs, each orthogonal to ``u``. Checks:

1. HF, teacher-forced log-probs of a few prompt + response sequences: hook ablation vs weight edit with biases vs the
   reasoning-manipulation edit (biases left as they are) vs clean; and the projection of the residual stream on ``u``
   (layer inputs) under each edit.
2. How much each kind of bias writes along ``u`` (``|b . u|`` vs the rms write of the clean model).
3. MXFP4: re-quantizing the edited expert down projections (blocks of 32 along the input dimension, E2M1 values,
   power-of-two scales) -- what fraction of the original write along ``u`` comes back.
4. vLLM (bf16 copy): the same log-prob comparison with the in-place edit applied through ``LLM.apply_model``, so the
   fused-MoE weights are known to be edited.
"""

import argparse
import functools
import os


def mxfp4_roundtrip(w, axis):
    """Quantize-dequantize ``w`` to MXFP4 with blocks of 32 along ``axis`` (OCP MX: E2M1 values, E8M0 scales)."""
    import torch

    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=w.device)
    x = w.float().movedim(axis, -1)
    shape = x.shape
    x = x.reshape(*shape[:-1], shape[-1] // 32, 32)
    amax = x.abs().amax(-1, keepdim=True).clamp_min(2 ** -126)
    scale = torch.exp2(torch.floor(torch.log2(amax)) - 2)  # shared exponent; E2M1 max exponent is 2
    y = (x / scale).clamp(-6, 6)
    q = grid[(y.abs().unsqueeze(-1) - grid).abs().argmin(-1)] * y.sign()
    return (q * scale).reshape(shape).movedim(-1, axis).to(w.dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--direction", default=None, help="method:site in directions/train.pt (default: random)")
    ap.add_argument("--n-seqs", type=int, default=4)
    ap.add_argument("--skip-vllm", action="store_true")
    ap.add_argument("--skip-hf", action="store_true", help="vLLM part only (run the two parts as separate processes: "
                                                          "the HF model is not reliably freed before vLLM starts)")
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from loop_steer.paths import run_dir, setup_job_env
    setup_job_env()
    import torch
    from transformers import AutoConfig

    from loop_steer.cot import format_prompt, load_prompts
    from loop_steer.models import _remote_kwargs, load_tokenizer

    tok = load_tokenizer(args.model)
    D = AutoConfig.from_pretrained(args.model, **_remote_kwargs(args.model)).hidden_size
    if args.direction:
        dirs = torch.load(run_dir(args.model) / "directions" / "train.pt", weights_only=False)
        method, site = args.direction.split(":")
        u = dirs[method]["dirs"][dirs["sites"].index((0, int(site.lstrip("l"))))].float()
    else:
        torch.manual_seed(0)
        u = torch.randn(D)
    u = u / u.norm()
    texts = [format_prompt(tok, p) + "<|channel|>analysis<|message|>The user asks for something. I need to think about"
             " whether this is allowed and how to answer it.<|end|><|start|>assistant<|channel|>final<|message|>"
             "Here is my answer:" for p in load_prompts("train")[500: 500 + args.n_seqs]]
    seqs = [tok(t, add_special_tokens=False).input_ids for t in texts]
    print(f"{args.model}: {len(seqs)} sequences, direction {args.direction or 'random'}", flush=True)

    if not args.skip_hf:
        hf_checks(args, tok, u, seqs)
    if not args.skip_vllm:
        vllm_checks(args, u, seqs)


def hf_checks(args, tok, u, seqs):
    import torch

    from loop_steer.hooks import ablation_hooks, decoder_layers, hooked
    from loop_steer.models import load_model
    from loop_steer.ortho import _residual_writers, orthogonalize_

    model = load_model(args.model)
    # ---- 1. HF log-probs and residual projections
    dev, uu = model.device, u.to(model.device)

    def run():
        proj, lps = [], []
        hs = [b.register_forward_pre_hook(lambda m, a, kw: proj.append(
            float(((a[0] if a else kw["hidden_states"]).float() @ uu).pow(2).mean().sqrt())), with_kwargs=True)
            for b in decoder_layers(model)]
        try:
            with torch.inference_mode():
                for s in seqs:
                    ids = torch.tensor([s], device=dev)
                    lp = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)[:-1]
                    lps.append(lp.gather(1, ids[0, 1:, None])[:, 0].cpu())
        finally:
            for h in hs:
                h.remove()
        return torch.cat(lps), sum(proj) / len(proj)

    res = {"clean": run()}
    with hooked(ablation_hooks(model, u)):
        res["hooks"] = run()
    orthogonalize_(model, u, biases=True)
    res["ortho (+biases)"] = run()
    orthogonalize_(model, u, biases=False)
    res["ortho, biases kept (upstream)"] = run()
    orthogonalize_(model, None)
    ref = res["hooks"][0]
    print("\n[HF] mean |dlogp| vs hook ablation, rms projection of layer inputs on u")
    for k, (lp, pr) in res.items():
        print(f"  {k:32s} {float((lp - ref).abs().mean()):.4f}   {pr:.4f}")

    # ---- 2. bias writes along u
    write = []
    hs = [getattr(b, "self_attn").register_forward_hook(lambda m, a, o: write.append(float((o[0].float() @ uu).pow(2).mean().sqrt())))
          for b in decoder_layers(model)]
    with torch.inference_mode():
        model(input_ids=torch.tensor([seqs[0]], device=dev))
    for h in hs:
        h.remove()
    ob = torch.stack([b.self_attn.o_proj.bias.float() @ uu for b in decoder_layers(model)]).abs()
    eb = torch.stack([b.mlp.experts.down_proj_bias.float() @ uu for b in decoder_layers(model)]).abs()  # [L, E]
    print(f"\n[biases] |o_proj bias . u| mean {float(ob.mean()):.4f} (rms attention write along u {sum(write) / len(write):.4f});"
          f" |expert bias . u| mean {float(eb.mean()):.4f}, max {float(eb.max()):.4f}")

    # ---- 3. MXFP4 round trip of the edited experts
    orthogonalize_(model, u, biases=True)
    num = den = 0.0
    for name, w, kind, _, is_bias in _residual_writers(model):
        if not name.startswith("experts_down.") or is_bias:
            continue
        orig = model._ls_original[name].to(dev).float()  # [E, I, D]
        q = mxfp4_roundtrip(w.data, axis=-2)
        num += float((q.float() @ uu).pow(2).sum())
        den += float((orig @ uu).pow(2).sum())
    print(f"\n[MXFP4] write along u of the edited experts after re-quantization: {(num / den) ** 0.5:.1%} of the original"
          " (0% in bf16)")
    orthogonalize_(model, None)


def vllm_checks(args, u, seqs):
    # ---- 4. vLLM (bf16 copy): hooks vs in-place edit
    import torch
    from vllm import LLM, SamplingParams

    from loop_steer import vllm_hooks
    from loop_steer.models import vllm_kwargs, vllm_model
    from loop_steer.ortho import orthogonalize_

    llm = LLM(model=vllm_model(args.model), gpu_memory_utilization=0.8, max_model_len=4096, enforce_eager=True,
              **vllm_kwargs(args.model))
    sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0)

    def vlp():
        outs = llm.generate([{"prompt_token_ids": s} for s in seqs], sp, use_tqdm=False)
        return torch.cat([torch.tensor([d[t].logprob for d, t in zip(o.prompt_logprobs[1:], s[1:])])
                          for o, s in zip(outs, seqs)])

    out = {"clean": vlp()}
    llm.apply_model(functools.partial(vllm_hooks.install, specs=[{"kind": "ablate", "dir": u, "loops": None}]))
    out["hooks"] = vlp()
    llm.apply_model(vllm_hooks.clear)
    for name, b in (("ortho (+biases)", True), ("ortho, biases kept (upstream)", False)):
        llm.apply_model(functools.partial(orthogonalize_, direction=u, biases=b))
        out[name] = vlp()
    llm.apply_model(functools.partial(orthogonalize_, direction=None))
    print("\n[vLLM] mean |dlogp| vs vLLM hook ablation")
    for k, lp in out.items():
        print(f"  {k:32s} {float((lp - out['hooks']).abs().mean()):.4f}")


if __name__ == "__main__":
    main()
