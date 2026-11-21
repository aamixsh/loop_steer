#!/usr/bin/env python3
"""Why did a kicked MATH-500 run not finish? Label every response that hit the token cap (E18).

    scripts/cutoff_judge.py --gpu 1 [--models ouro,nanbeige,dory,qwen] [--tags kickp,kickp_loops4,kickp_loops6]

For each ``capability/math500/<tag>/<cand>.parquet`` (one row per problem and sample), every row with ``hit_limit``
gets two labels, written to ``<tag>/cutoff/<cand>.parquet`` (a subfolder, so the scripts that glob a tag's
``*.parquet`` as candidates never see them; skip-existing):

  ``tail_rep``  1 - zlib(tail) / len(tail) on the last 3000 characters: how much of the end is verbatim repetition
                (clean reasoning sits near 0.55-0.65, a loop of one phrase near 0.97)
  ``cut``       gpt-oss-20b (the project's pinned judge, greedy, ``reasoning=low``) reading the problem and the last
                6000 characters of the response: ``coherent`` (sensible reasoning still working toward an answer),
                ``repetitive`` (stuck repeating the same text or the same few steps) or ``incoherent`` (garbled,
                off-topic or nonsensical)

Rows that finished are not judged. scripts/export_blog_data.py splits the kick figures' "not solved" into wrong
answer / no answer / cut off (by ``cut``).
"""

import argparse
import os
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"ouro": "Ouro-1.4B-Thinking", "nanbeige": "Nanbeige4.2-3B", "dory": "NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16",
        "qwen": "Qwen3-8B"}
TAIL = 6000
LABELS = {"A": "coherent", "B": "repetitive", "C": "incoherent"}
PROMPT = """You are checking a language model's reasoning on a math problem. The model hit its length limit before it \
finished, so its response was cut off. Below are the problem and the LAST PART of the cut-off response.

[PROBLEM START]
{problem}
[PROBLEM END]

[END OF RESPONSE START]
{tail}
[END OF RESPONSE END]

Classify the end of the response:
A = coherent: sensible reasoning about this problem that is still working toward an answer (it may be slow, \
long-winded or re-checking its work, but it makes sense and is not stuck)
B = repetitive: stuck repeating the same text, the same sentence or the same few steps over and over
C = incoherent: garbled, off-topic or nonsensical text

Reply with one letter: A, B or C."""


def tail_rep(text):
    t = (text or "")[-3000:].encode()
    return 1 - len(zlib.compress(t)) / len(t) if len(t) >= 200 else 0.0


def parse(text):
    t = text.strip().upper()
    for ch in t:
        if ch in LABELS:
            return LABELS[ch]
    raise ValueError(text[:80])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--models", default="ouro,nanbeige,dory,qwen")
    ap.add_argument("--tags", default="kickp,kickp_loops4,kickp_loops6")
    a = ap.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    from loop_steer.paths import setup_job_env
    setup_job_env()
    import pandas as pd
    from capability import load_problems

    problems = {p["id"]: p["question"] for p in load_problems("math500")}
    todo = []
    for m in a.models.split(","):
        for tag in a.tags.split(","):
            d = ROOT / "data" / "runs" / RUNS[m] / "capability" / "math500" / tag
            for f in sorted(d.glob("*.parquet")) if d.exists() else []:
                out = f.parent / "cutoff" / f.name
                if not out.exists():
                    todo.append((f, out))
    if not todo:
        print("nothing to judge")
        return
    frames = []
    for f, _ in todo:
        df = pd.read_parquet(f, columns=["problem_id", "sample", "hit_limit", "response"])
        frames.append(df[df.hit_limit.astype(bool)].reset_index(drop=True))
    prompts = [PROMPT.format(problem=problems[pid], tail=str(r or "")[-TAIL:])
               for df in frames for pid, r in zip(df.problem_id, df.response)]
    print(f"{len(todo)} files, {len(prompts)} cut-off responses", flush=True)
    labels = []
    if prompts:
        from loop_steer.judge import chat_offline, load_offline_judge
        # 16k context: a garbled tail can take one token per character (6000 characters + the problem)
        llm = load_offline_judge(gpu_memory_utilization=0.85, max_model_len=16384)
        labels = chat_offline(prompts, llm, max_tokens=1024, parse=parse)
    i = 0
    for (f, out), df in zip(todo, frames):
        r = df[["problem_id", "sample"]].copy()
        r["tail_rep"] = [round(tail_rep(t), 4) for t in df.response]
        r["cut"] = labels[i: i + len(df)]
        i += len(df)
        out.parent.mkdir(exist_ok=True)
        r.to_parquet(out)
        print(f"-> {out.name}: {len(r)} cut off, {r.cut.value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
