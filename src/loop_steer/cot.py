"""Chat formatting and reasoning-trace parsing for <think>-style models, gpt-oss (harmony format) and Inkling.

Inkling writes ``<|message_model|><|content_thinking|>...<|end_message|><|message_model|><|content_text|>answer
<|end_message|><|content_model_end_sampling|>`` (its template has no generation prefix: the model emits its own role
header); the CoT is everything up to and including the text-message header.

gpt-oss writes its reasoning in the ``analysis`` channel and its answer in the ``final`` channel:
``<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>answer<|return|>``. The CoT is
everything up to and including the ``final`` header (the counterpart of ``</think>``), the answer is the rest.
"""

import re

import pandas as pd

from loop_steer.paths import dataset_dir

THINK_END = "</think>"
HARMONY_FINAL = "<|channel|>final<|message|>"
INKLING_TEXT = "<|message_model|><|content_text|>"
# Everything up to and including </think> (gpt-oss: the final-channel header; Inkling: the model's first text-message
# header, after its <|content_thinking|> message) is the CoT; the remainder is the answer.
_THINK_SPLIT = re.compile(
    r"(.*?(?:</think>|<\|channel\|>final<\|message\|>|<\|message_model\|>[^<]*<\|content_text\|>))(.*)", re.DOTALL)
_END_TOKENS = re.compile(r"(.*?)(?:<\|im_end\|>|<\|endoftext\|>|<｜end▁of▁sentence｜>|<\|return\|>|<\|end\|>"
                         r"|<\|end_message\|>|<\|content_model_end_sampling\|>|$)", re.DOTALL)
# gpt-oss's template stamps today's date into the system message; pin it so prompts do not change from day to day
_HARMONY_DATE = re.compile(r"Current date: \d{4}-\d{2}-\d{2}")
PINNED_DATE = "Current date: 2025-08-05"  # gpt-oss release date


def load_prompts(split: str, kind: str = "harmful") -> list[str]:
    """Prompts from the reasoning-manipulation dataset copy, e.g. split='train'."""
    return pd.read_csv(dataset_dir() / f"{split}_{kind}_prompts.csv")["prompt"].tolist()


def format_prompt(tokenizer, prompt: str, **template_kwargs) -> str:
    """User-only chat turn with the generation prompt, as in reasoning-manipulation."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False, **template_kwargs,
    )
    return _HARMONY_DATE.sub(PINNED_DATE, text)


def split_cot(response: str) -> tuple[str, str, bool]:
    """Return (cot including </think> or the harmony final-channel header, answer, has_cot)."""
    match = _THINK_SPLIT.search(response)
    if match is None:
        return response, "", False
    return match.group(1), match.group(2), True


def strip_end(text: str) -> str:
    """Cut a decoded continuation at the first end-of-turn token."""
    return _END_TOKENS.search(text).group(1)
