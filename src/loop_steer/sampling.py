"""Two-stage CoT/answer sampling (reasoning-manipulation protocol), backend-agnostic.

A *generator* is a callable ``generate(texts, prompt_texts=None, answer=False) -> [(text, n_tokens, hit_limit), ...]``
(same order as ``texts``; ``text`` keeps special tokens such as ``</think>``). ``prompt_texts[i]`` is the formatted user
turn that ``texts[i]`` starts with, and ``answer`` says whether the generated tokens are answer tokens (stage 2) or
CoT tokens (stage 1, continuations); a generator uses them only for position-restricted interventions.
``vllm_generator`` and ``hf_generator`` build one; ``two_stage`` runs the protocol.
"""

import pandas as pd

from loop_steer.cot import format_prompt, split_cot, strip_end


def vllm_generator(llm, sampling, tokenizer=None, gated=False):
    """vLLM generator. With ``tokenizer``, prompts go in as token ids (``add_special_tokens=False``).

    ``gated=True`` (needs ``tokenizer``) attaches each request's position-class boundaries
    (``extra_args["ls_bounds"] = (P, C)``, see ``loop_steer.vllm_hooks``): P = tokens of the user turn, C = tokens of
    the whole input when the generated tokens are answer tokens, else -1 (everything after the user turn is CoT).
    """
    if gated and tokenizer is None:
        raise ValueError("gated generation needs the tokenizer (prompts are passed as token ids)")

    def generate(texts, prompt_texts=None, answer=False):
        if tokenizer is not None:
            ids = [tokenizer(t, add_special_tokens=False).input_ids for t in texts]
            prompts = [{"prompt_token_ids": i} for i in ids]
        else:
            prompts = texts
        params = sampling
        if gated:
            if prompt_texts is None:
                raise ValueError("gated generation needs prompt_texts")
            params = []
            for t, p, i in zip(texts, prompt_texts, ids):
                if not t.startswith(p):
                    raise ValueError("prompt_texts[i] must be a prefix of texts[i]")
                sp = sampling.clone()
                n_prompt = len(tokenizer(p, add_special_tokens=False).input_ids)
                sp.extra_args = {**(sp.extra_args or {}), "ls_bounds": (n_prompt, len(i) if answer else -1)}
                params.append(sp)
        outs = llm.generate(prompts, params)
        return [(o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason == "length")
                for o in outs]
    return generate


class _LastPositionHead:
    """Context manager: ``lm_head`` only sees the last position (generation needs nothing else).

    Some remote-code models unembed every position, which for a 166k vocabulary and a long
    prefill needs tens of GB. ``generate`` only reads ``logits[:, -1]``, so slicing is exact.
    """

    def __init__(self, model):
        self.model, self.orig = model, model.lm_head

    def __enter__(self):
        import torch

        orig = self.orig

        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = orig

            def forward(self, x):
                return self.inner(x[:, -1:, :])

        self.model.lm_head = Head()

    def __exit__(self, *exc):
        self.model.lm_head = self.orig


def hf_generator(model, tokenizer, *, max_new_tokens, temperature, top_p=1.0, batch_size=96, seed=0):
    """HF ``generate`` in left-padded batches, longest inputs first (less padding).

    Sampling is plain temperature sampling (top_k disabled, top_p=1 unless given), matching the
    vLLM runs; the model's own generation_config (e.g. top_k=20, top_p=0.95) is overridden.
    Hooks registered on ``model`` stay active for every decode step.
    """
    import torch

    tokenizer.padding_side = "left"
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

    def generate(texts, prompt_texts=None, answer=False):  # position classes: not supported (vLLM only)
        torch.manual_seed(seed)
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        results = [None] * len(texts)
        with _LastPositionHead(model):
            for start in range(0, len(order), batch_size):
                idx = order[start: start + batch_size]
                enc = tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True,
                                add_special_tokens=False).to(model.device)
                with torch.inference_mode():
                    out = model.generate(
                        **enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None, top_p=top_p, top_k=0,
                        pad_token_id=pad, eos_token_id=eos,
                    )
                for row, i in zip(out[:, enc.input_ids.shape[1]:].tolist(), idx):
                    ended = eos in row
                    ids = row[: row.index(eos)] if ended else row
                    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    results[i] = (text, len(ids), not ended)
                print(f"  generated {min(start + batch_size, len(order))}/{len(order)}", flush=True)
        return results

    return generate


_ROW_COLUMNS = ["prompt_idx", "prompt", "cot_rep", "out_rep", "cot", "stage1_len_tokens", "output",
                "output_truncated"]


def _answer_rows(generate_answer, cots, out_reps):
    """Stage 2: ``out_reps`` answers per CoT; ``cots`` = [(prompt_idx, prompt, cot_rep, prefix, cot, n_tokens)]."""
    stage2 = [(c, rep) for c in cots for rep in range(out_reps)]
    outs = generate_answer([c[3] + c[4] for c, _ in stage2], prompt_texts=[c[3] for c, _ in stage2],
                           answer=True) if stage2 else []
    return pd.DataFrame([
        {
            "prompt_idx": idx, "prompt": prompt, "cot_rep": cot_rep, "out_rep": out_rep,
            "cot": cot, "stage1_len_tokens": n_tok, "output": strip_end(text), "output_truncated": hit,
        }
        for ((idx, prompt, cot_rep, _, cot, n_tok), out_rep), (text, _, hit) in zip(stage2, outs)
    ], columns=_ROW_COLUMNS)


def two_stage(generate, tokenizer, prompts, *, cot_reps, out_reps, prompt_offset=0, generate_answer=None):
    """Sample ``cot_reps`` responses per prompt, then ``out_reps`` answers per valid CoT.

    Stage-1 responses without ``</think>`` are dropped from the rows and returned in
    ``df.attrs['invalid']``. They are of two kinds, told apart by ``hit_limit``: runs that hit the
    token cap (CoT possibly still going; see ``extend_capped``) and ones that stopped by themselves
    without thinking (a direct answer, to be judged as is).
    ``generate_answer`` is the stage-2 generator (default ``generate``).
    Returns one row per (prompt, cot, answer).
    """
    generate_answer = generate_answer or generate
    formatted = [format_prompt(tokenizer, p) for p in prompts]
    stage1 = [(i, rep) for i in range(len(prompts)) for rep in range(cot_reps)]
    outs = generate([formatted[i] for i, _ in stage1], prompt_texts=[formatted[i] for i, _ in stage1])
    cots, invalid = [], []
    for (i, rep), (text, n_tokens, hit) in zip(stage1, outs):
        cot, _, ok = split_cot(text)
        if ok:  # n_tokens = stage-1 length incl. answer
            cots.append((prompt_offset + i, prompts[i], rep, formatted[i], cot, n_tokens))
        else:
            invalid.append({"prompt_idx": prompt_offset + i, "prompt": prompts[i], "cot_rep": rep,
                            "response": text, "n_tokens": n_tokens, "hit_limit": bool(hit)})
    n_invalid = len(invalid)
    print(f"Stage 1: {len(cots)} valid CoTs, {n_invalid} missing </think>", flush=True)

    df = _answer_rows(generate_answer, cots, out_reps)
    df.attrs["n_invalid_cot"] = n_invalid
    df.attrs["n_cot_total"] = len(stage1)
    df.attrs["invalid"] = pd.DataFrame(invalid)  # stage-1 responses without </think>
    return df


def extend_capped(generate_more, generate_answer, tokenizer, invalid, *, cap, out_reps):
    """Continue stage-1 responses that hit the token cap, then sample answers for those that now close.

    ``invalid``: stage-1 records of an earlier run (``invalid/<candidate>.parquet``). Rows with
    ``n_tokens >= cap`` are fed back as ``prompt + response`` and continued by ``generate_more``
    (same intervention must be active); the others (direct answers) are left alone. Returns the
    answer rows for CoTs that closed, like ``two_stage``; ``df.attrs['invalid']`` holds the
    continuations that still have no ``</think>`` (``hit_limit`` says whether they hit the new cap).
    """
    capped = invalid[invalid.n_tokens >= cap].reset_index(drop=True)
    formatted = [format_prompt(tokenizer, p) for p in capped.prompt]
    outs = generate_more([f + r for f, r in zip(formatted, capped.response)], prompt_texts=formatted) \
        if len(capped) else []
    cots, still = [], []
    for row, f, (more, n_more, hit) in zip(capped.itertuples(), formatted, outs):
        text = row.response + more
        cot, _, ok = split_cot(text)
        n_tokens = int(row.n_tokens) + n_more
        if ok:
            cots.append((row.prompt_idx, row.prompt, row.cot_rep, f, cot, n_tokens))
        else:
            still.append({"prompt_idx": row.prompt_idx, "prompt": row.prompt, "cot_rep": row.cot_rep,
                          "response": text, "n_tokens": n_tokens, "hit_limit": bool(hit)})
    print(f"Extension: {len(capped)} capped, {len(cots)} now closed, {len(still)} still missing </think>",
          flush=True)
    df = _answer_rows(generate_answer, cots, out_reps)
    df.attrs["n_invalid_cot"] = len(still)
    df.attrs["n_cot_total"] = len(capped)
    df.attrs["invalid"] = pd.DataFrame(still)
    return df
