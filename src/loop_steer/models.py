"""Load HF models/tokenizers, with pinned revisions and remote code for the looped models."""

import torch

from loop_steer import MODEL_ID as OURO_ID
from loop_steer import MODEL_REVISION as OURO_REVISION

NANBEIGE_ID = "Nanbeige/Nanbeige4.2-3B"
NANBEIGE_REVISION = "b82e54bd609793562a75cbf9337970a93369eab5"

# Looped models: pinned (revision, loop-count config attribute). Custom code => trust_remote_code.
LOOPED = {
    OURO_ID: (OURO_REVISION, "total_ut_steps"),
    NANBEIGE_ID: (NANBEIGE_REVISION, "num_loops"),
}


def is_ouro(model_name: str) -> bool:
    return "ouro" in model_name.lower()


def is_looped(model_name: str) -> bool:
    return model_name in LOOPED or is_ouro(model_name)


def n_loops(config) -> int:
    """Number of times the layer stack is applied (1 for ordinary models)."""
    return getattr(config, "total_ut_steps", None) or getattr(config, "num_loops", None) or 1


def _remote_kwargs(model_name: str) -> dict:
    if model_name in LOOPED:
        return {"revision": LOOPED[model_name][0], "trust_remote_code": True}
    return {"trust_remote_code": True} if is_ouro(model_name) else {}


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    kwargs = _remote_kwargs(model_name)
    if model_name == NANBEIGE_ID:
        kwargs["use_fast"] = False  # model card recommends the slow SentencePiece tokenizer
    return AutoTokenizer.from_pretrained(model_name, **kwargs)


def load_model(model_name: str, *, device="cuda:0", dtype=torch.bfloat16, n_loops_override=None):
    from transformers import AutoConfig, AutoModelForCausalLM

    kwargs = _remote_kwargs(model_name)
    if model_name == NANBEIGE_ID:
        # Its remote code (written for transformers ~4.45) calls DynamicCache.get_max_length(),
        # which no longer exists in 4.57; None means "unbounded", matching the old behaviour.
        from transformers import DynamicCache

        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = lambda self: None
    if model_name in LOOPED or is_ouro(model_name):
        config = AutoConfig.from_pretrained(model_name, **kwargs)
        if is_ouro(model_name):
            config.early_exit_threshold = 1.0  # always unembed the final loop
        if n_loops_override is not None:
            attr = LOOPED[model_name][1] if model_name in LOOPED else "total_ut_steps"
            setattr(config, attr, n_loops_override)
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
    """Extra ``vllm.LLM`` kwargs (pinned revision + remote code for the looped models)."""
    if model_name == OURO_ID:
        return {"revision": OURO_REVISION, "code_revision": OURO_REVISION,
                "tokenizer_revision": OURO_REVISION, "trust_remote_code": True}
    if model_name == NANBEIGE_ID:
        return {"revision": NANBEIGE_REVISION, "code_revision": NANBEIGE_REVISION,
                "tokenizer_revision": NANBEIGE_REVISION, "trust_remote_code": True,
                "tokenizer_mode": "slow"}  # same SentencePiece tokenizer as the HF path
    return {}


def prepare_vllm(model_name: str) -> bool:
    """Call before building ``vllm.LLM``. Registers out-of-tree architectures.

    Returns True when prompts should be passed as token ids (tokenized with the HF tokenizer and
    ``add_special_tokens=False``) so vLLM cannot add a second BOS. Nanbeige has no native vLLM
    support; our port (``loop_steer.vllm_nanbeige``) is registered in-process, which needs the
    engine core to run in this process too.
    """
    import os

    if model_name == NANBEIGE_ID:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from loop_steer.vllm_nanbeige import register

        register()
        return True
    return False
