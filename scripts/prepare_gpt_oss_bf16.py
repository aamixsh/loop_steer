#!/usr/bin/env python3
"""Write a bf16 copy of gpt-oss-20b (MXFP4 experts dequantized) for vLLM runs that edit or hook the model.

    scripts/prepare_gpt_oss_bf16.py            # -> data/models/gpt-oss-20b-bf16 (about 42 GB), CPU only

vLLM runs the MXFP4 experts with fused kernels on packed 4-bit weights, which in-place weight edits cannot touch.
The bf16 copy holds exactly the dequantized MXFP4 values (same function up to kernel numerics), so clean runs, hooks
and weight orthogonalization all use one engine. The reasoning-manipulation code also edits gpt-oss in bf16.
Skips if the copy exists. The tokenizer stays the hub snapshot (``loop_steer.models.vllm_kwargs``).
"""

import argparse
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from loop_steer.paths import setup_job_env
    setup_job_env()
    import torch
    from transformers import AutoModelForCausalLM, Mxfp4Config

    from loop_steer.models import GPT_OSS_ID, GPT_OSS_REVISION, gpt_oss_bf16_dir, load_tokenizer

    out = gpt_oss_bf16_dir()
    if (out / "config.json").exists() and not args.force:
        print(f"{out} exists; skipping")
        return
    model = AutoModelForCausalLM.from_pretrained(GPT_OSS_ID, revision=GPT_OSS_REVISION, dtype=torch.bfloat16,
                                                 quantization_config=Mxfp4Config(dequantize=True), device_map="cpu")
    experts = model.model.layers[0].mlp.experts
    print(f"loaded; experts down_proj {tuple(experts.down_proj.shape)} {experts.down_proj.dtype}", flush=True)
    if hasattr(model.config, "quantization_config"):
        del model.config.quantization_config  # the copy is plain bf16
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    load_tokenizer(GPT_OSS_ID).save_pretrained(out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
