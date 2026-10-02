"""Load HF models/tokenizers, with the pinned Ouro revision when applicable."""

import torch

from loop_steer import MODEL_ID as OURO_ID
from loop_steer import MODEL_REVISION as OURO_REVISION


def is_ouro(model_name: str) -> bool:
    return "ouro" in model_name.lower()


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    kwargs = {"revision": OURO_REVISION, "trust_remote_code": True} if model_name == OURO_ID else {}
    return AutoTokenizer.from_pretrained(model_name, **kwargs)


def load_model(model_name: str, *, device="cuda:0", dtype=torch.bfloat16, total_ut_steps=None):
    from transformers import AutoConfig, AutoModelForCausalLM

    kwargs = {}
    if is_ouro(model_name):
        kwargs = {"trust_remote_code": True}
        if model_name == OURO_ID:
            kwargs["revision"] = OURO_REVISION
        config = AutoConfig.from_pretrained(model_name, **kwargs)
        config.early_exit_threshold = 1.0  # always unembed the final loop
        if total_ut_steps is not None:
            config.total_ut_steps = total_ut_steps
        kwargs["config"] = config
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map=device, attn_implementation="sdpa", **kwargs,
    )
    return model.eval()


def eoi_length(tokenizer, **template_kwargs) -> int:
    """Number of template tokens after the user message (Arditi's end-of-instruction tokens)."""
    marker = "XYZZY_MARKER"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}], add_generation_prompt=True, tokenize=False, **template_kwargs,
    )
    suffix = text.split(marker, 1)[1]
    return len(tokenizer.encode(suffix, add_special_tokens=False))


def vllm_kwargs(model_name: str) -> dict:
    """Extra ``vllm.LLM`` kwargs (pinned revision + remote code for Ouro)."""
    if model_name == OURO_ID:
        return {"revision": OURO_REVISION, "code_revision": OURO_REVISION,
                "tokenizer_revision": OURO_REVISION, "trust_remote_code": True}
    return {}
