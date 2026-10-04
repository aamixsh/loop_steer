"""Two-stage CoT/answer sampling (reasoning-manipulation protocol), backend-agnostic.

A *generator* is a callable ``generate(texts) -> [(text, n_tokens, hit_limit), ...]``
(same order as ``texts``; ``text`` keeps special tokens such as ``</think>``).
``vllm_generator`` and ``hf_generator`` build one; ``two_stage`` runs the protocol.
"""

import pandas as pd

from loop_steer.cot import format_prompt, split_cot, strip_end


def vllm_generator(llm, sampling, tokenizer=None):
    """vLLM generator. With ``tokenizer``, prompts go in as token ids (``add_special_tokens=False``)."""

    def generate(texts):
        if tokenizer is not None:
            prompts = [{"prompt_token_ids": tokenizer(t, add_special_tokens=False).input_ids} for t in texts]
        else:
            prompts = texts
        outs = llm.generate(prompts, sampling)
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

    def generate(texts):
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


def two_stage(generate, tokenizer, prompts, *, cot_reps, out_reps, prompt_offset=0):
    """Sample ``cot_reps`` responses per prompt, then ``out_reps`` answers per valid CoT.

    Stage-1 responses without ``</think>`` are dropped from the rows and returned in
    ``df.attrs['invalid']`` for inspection.
    Returns one row per (prompt, cot, answer).
    """
    formatted = [format_prompt(tokenizer, p) for p in prompts]
    stage1 = [(i, rep) for i in range(len(prompts)) for rep in range(cot_reps)]
    outs = generate([formatted[i] for i, _ in stage1])
    cots, invalid = [], []
    for (i, rep), (text, n_tokens, _) in zip(stage1, outs):
        cot, _, ok = split_cot(text)
        if ok:
            cots.append((i, rep, cot, n_tokens))  # stage-1 length incl. answer
        else:
            invalid.append({"prompt_idx": prompt_offset + i, "prompt": prompts[i], "cot_rep": rep,
                            "response": text, "n_tokens": n_tokens})
    n_invalid = len(invalid)
    print(f"Stage 1: {len(cots)} valid CoTs, {n_invalid} missing </think>", flush=True)

    stage2 = [(c, rep) for c in cots for rep in range(out_reps)]
    outs = generate([formatted[c[0]] + c[2] for c, _ in stage2])
    df = pd.DataFrame([
        {
            "prompt_idx": prompt_offset + i, "prompt": prompts[i], "cot_rep": cot_rep, "out_rep": out_rep,
            "cot": cot, "stage1_len_tokens": n_tok, "output": strip_end(text), "output_truncated": hit,
        }
        for ((i, cot_rep, cot, n_tok), out_rep), (text, _, hit) in zip(stage2, outs)
    ])
    df.attrs["n_invalid_cot"] = n_invalid
    df.attrs["n_cot_total"] = len(stage1)
    df.attrs["invalid"] = pd.DataFrame(invalid)  # stage-1 responses without </think>
    return df
