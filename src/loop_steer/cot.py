"""Chat formatting and reasoning-trace parsing for <think>-style models."""

import re

import pandas as pd

from loop_steer.paths import dataset_dir

THINK_END = "</think>"
# Everything up to and including </think> is the CoT; the remainder is the answer.
_THINK_SPLIT = re.compile(r"(.*?</think>)(.*)", re.DOTALL)
_END_TOKENS = re.compile(r"(.*?)(?:<\|im_end\|>|<\|endoftext\|>|<｜end▁of▁sentence｜>|$)", re.DOTALL)


def load_prompts(split: str, kind: str = "harmful") -> list[str]:
    """Prompts from the reasoning-manipulation dataset copy, e.g. split='train'."""
    return pd.read_csv(dataset_dir() / f"{split}_{kind}_prompts.csv")["prompt"].tolist()


def format_prompt(tokenizer, prompt: str, **template_kwargs) -> str:
    """User-only chat turn with the generation prompt, as in reasoning-manipulation."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False, **template_kwargs,
    )


def split_cot(response: str) -> tuple[str, str, bool]:
    """Return (cot including </think>, answer, has_cot)."""
    match = _THINK_SPLIT.search(response)
    if match is None:
        return response, "", False
    return match.group(1), match.group(2), True


def strip_end(text: str) -> str:
    """Cut a decoded continuation at the first end-of-turn token."""
    return _END_TOKENS.search(text).group(1)
