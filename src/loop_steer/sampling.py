"""Two-stage CoT/answer sampling with vLLM (reasoning-manipulation protocol)."""

import pandas as pd

from loop_steer.cot import format_prompt, split_cot, strip_end


def two_stage(llm, tokenizer, prompts, *, cot_reps, out_reps, sampling, prompt_offset=0):
    """Sample ``cot_reps`` responses per prompt, then ``out_reps`` answers per valid CoT.

    Stage-1 responses without ``</think>`` are dropped from the rows and returned in
    ``df.attrs['invalid']`` for inspection.
    Returns one row per (prompt, cot, answer).
    """
    formatted = [format_prompt(tokenizer, p) for p in prompts]
    stage1 = [(i, rep) for i in range(len(prompts)) for rep in range(cot_reps)]
    outs = llm.generate([formatted[i] for i, _ in stage1], sampling)
    cots, invalid = [], []
    for (i, rep), o in zip(stage1, outs):
        cot, _, ok = split_cot(o.outputs[0].text)
        if ok:
            cots.append((i, rep, cot, len(o.outputs[0].token_ids)))  # stage-1 length incl. answer
        else:
            invalid.append({"prompt_idx": prompt_offset + i, "prompt": prompts[i], "cot_rep": rep,
                            "response": o.outputs[0].text, "n_tokens": len(o.outputs[0].token_ids)})
    n_invalid = len(invalid)
    print(f"Stage 1: {len(cots)} valid CoTs, {n_invalid} missing </think>", flush=True)

    stage2 = [(c, rep) for c in cots for rep in range(out_reps)]
    outs = llm.generate([formatted[c[0]] + c[2] for c, _ in stage2], sampling)
    df = pd.DataFrame([
        {
            "prompt_idx": prompt_offset + i, "prompt": prompts[i], "cot_rep": cot_rep, "out_rep": out_rep,
            "cot": cot, "stage1_len_tokens": n_tok, "output": strip_end(o.outputs[0].text),
            "output_truncated": o.outputs[0].finish_reason == "length",
        }
        for ((i, cot_rep, cot, n_tok), out_rep), o in zip(stage2, outs)
    ])
    df.attrs["n_invalid_cot"] = n_invalid
    df.attrs["n_cot_total"] = len(stage1)
    df.attrs["invalid"] = pd.DataFrame(invalid)  # stage-1 responses without </think>
    return df
