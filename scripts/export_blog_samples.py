#!/usr/bin/env python3
"""Replies behind the blog's dots: gzipped JSON, base64-encoded as ``.json.gz.txt`` under docs/blog/samples/, fetched
by the page when a dot is clicked.

    .venv/bin/python scripts/export_blog_samples.py

refusal-<run>.json.gz   every candidate of a refusal run (switchboard, extra loops), for the prompts in
                        export_blog_data.SAFE_PROMPTS only: {"cands": {cand: {prompt_idx: [refused of 9, closed,
                        thinking tokens, thinking, answer]}}}; the sample shown agrees with the prompt's majority verdict
                        (a closed CoT with a non-empty answer when one exists)
kick-<run>/<cand>.json.gz  one kick run's 100 MATH-500 responses in dot order: {"rows": [[outcome, closed, thinking
                        tokens, thinking, answer, predicted]]} (outcome codes as export_blog_data.kick_outcomes)
math500.json.gz         the 100 problems in dot order: {"ids", "problems", "answers"}

Replies to the other harmful prompts are never exported: the page shows their prompt and verdict only.
"""

import base64
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_blog_data  # noqa: E402
from export_blog_data import KICKS, MODELS, NDOTS, RUNS, SAFE_PROMPTS, gen_dirs, kick_outcomes, read_scored  # noqa: E402


OUT = Path(__file__).resolve().parents[1] / "docs" / "blog" / "samples"
REFUSAL_RUNS = {"dory": ("dory", ["grid_test", "grid_test_b"]), "dory4": ("dory", ["grid_test4", "grid_test4_b"]),
                "ouro": ("ouro", ["grid_test", "grid_test_b"]), "ouro6": ("ouro", ["grid_test6"]),
                "nanbeige": ("nanbeige", ["grid_test", "grid_test_b"]), "qwen": ("qwen", ["grid_test", "grid_test_b"])}


def write(path, obj):
    """gzip, then base64 as ``<name>.gz.txt``: artifacts serve text but not .gz, and the raw JSON is ~10x larger"""
    path = path.with_name(path.name + ".txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode()
    path.write_bytes(base64.b64encode(gzip.compress(raw, compresslevel=9)))
    return path.stat().st_size


def strip_think(cot):
    cot = str(cot or "")
    for tag in ("<think>", "</think>"):
        cot = cot.replace(tag, "")
    return cot.strip()


def refusal_run(model, tags):
    dirs = gen_dirs(model, tags)
    cands = sorted({f.name.removesuffix(".scored.parquet") for d in dirs for sub in (".", "invalid")
                    for f in (d / sub).glob("*.scored.parquet")} if dirs else [])
    out = {}
    for cand in cands:
        parts = []
        for d in dirs:
            for sub in (".", "invalid"):
                f = d / sub / f"{cand}.scored.parquet"
                if not f.exists():
                    continue
                x = read_scored(f)  # the judge of record (BLOG_JUDGE / --judge, as export_blog_data.py)
                x = x[x.prompt_idx.isin(SAFE_PROMPTS)].copy()
                if sub == "invalid":  # the CoT never closed: judged whole, no answer
                    x["closed"], x["think"], x["tokens"], x["answer"], x["out_rep"] = 0, x.response, x.n_tokens, "", 0
                else:
                    x["closed"], x["think"], x["tokens"], x["answer"] = 1, x.cot, x.stage1_len_tokens, x.output
                    if "out_rep" not in x:
                        x["out_rep"] = 0
                parts.append(x[["prompt_idx", "cot_rep", "out_rep", "closed", "think", "tokens", "answer", "sr_refusal"]])
        if not parts:
            continue
        df = pd.concat(parts)
        df["sr_refusal"] = df.sr_refusal.fillna(0)
        res = {}
        for p, sub in df.groupby("prompt_idx"):
            share = float(sub.groupby("cot_rep").sr_refusal.mean().mean())  # as export_blog_data.per_prompt
            agree = sub[(sub.sr_refusal >= 0.5) == (share >= 0.5)].copy()
            agree["has"] = agree.answer.fillna("").str.strip().astype(bool)
            agree = agree.sort_values(["closed", "has", "cot_rep", "out_rep"], ascending=[False, False, True, True])
            r = agree.iloc[0] if len(agree) else sub.iloc[0]
            res[str(int(p))] = [round(9 * share), int(r.closed), int(r.tokens), strip_think(r.think), str(r.answer or "").strip()]
        out[cand] = res
    return out


def kick_run(name, tag, tokenizer):
    m = MODELS[name.rstrip("46")]
    capdir = RUNS / m / "capability" / "math500" / tag
    sizes = 0
    for f in sorted(capdir.glob("*.parquet")):
        cand = f.name.removesuffix(".parquet")
        d = pd.read_parquet(f).sort_values(["problem_id", "sample"]).drop_duplicates("problem_id")
        codes = kick_outcomes(capdir, cand)
        rows = []
        for (_, r), code in zip(d.head(NDOTS).iterrows(), codes):
            text = str(r.response or "")
            # split at the end of the reasoning by plain search (split_cot's lazy regex is quadratic on long unclosed text)
            ends = [(text.find(t), t) for t in ("</think>", "<|channel|>final<|message|>") if t in text]
            closed = bool(ends)
            if closed:
                k, t = min(ends)
                think, answer = text[:k], text[k + len(t):]
            else:
                think, answer = text, ""
            think = strip_think(think)
            # thinking tokens: every generated token less the answer's (the answer is short, the thinking up to 8k)
            toks = int(r.n_tokens) - (len(tokenizer(answer, add_special_tokens=False).input_ids) if closed else 0)
            rows.append([code, int(bool(closed)), toks, think, answer.strip(), None if pd.isna(r.pred) else str(r.pred)])
        sizes += write(OUT / f"kick-{name}" / f"{cand}.json.gz", {"cand": cand, "rows": rows})
    return sizes


def main():
    total = 0
    args = sys.argv[1:]
    if "--judge" in args:  # same switch as export_blog_data.py: which judge picks the sample shown for each prompt
        i = args.index("--judge")
        export_blog_data.JUDGE = args[i + 1]
        del args[i:i + 2]
    only = args or ["refusal", "kick"]  # e.g. `export_blog_samples.py kick` redoes the kick files only
    for run, (model, tags) in REFUSAL_RUNS.items() if "refusal" in only else []:
        n = write(OUT / f"refusal-{run}.json.gz", {"run": run, "cands": refusal_run(model, tags)})
        print(f"refusal-{run}: {n / 1e6:.2f} MB", flush=True)
        total += n
    from loop_steer.models import load_tokenizer
    ids = {"ouro": "ByteDance/Ouro-1.4B-Thinking", "nanbeige": "Nanbeige/Nanbeige4.2-3B",
           "dory": "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16", "qwen": "Qwen/Qwen3-8B"}
    toks = {}
    for name, (tag, _) in KICKS.items() if "kick" in only else []:
        base = name.rstrip("46")
        if base not in toks:
            toks[base] = load_tokenizer(ids[base])
        n = kick_run(name, tag, toks[base])
        print(f"kick-{name}: {n / 1e6:.2f} MB", flush=True)
        total += n
    from capability import load_problems
    probs = {p["id"]: p for p in load_problems("math500")}
    f = next((RUNS / MODELS["ouro"] / "capability" / "math500" / "kickp").glob("none.parquet"))
    pids = sorted(pd.read_parquet(f, columns=["problem_id"]).problem_id.unique())[:NDOTS]
    total += write(OUT / "math500.json.gz", {"ids": pids, "problems": [probs[p]["question"] for p in pids],
                                             "answers": [probs[p]["answer"] for p in pids]})
    print(f"total {total / 1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
