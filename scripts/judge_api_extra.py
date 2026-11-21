#!/usr/bin/env python3
"""The other gpt-oss-20b judgments behind the blog, re-judged with an API judge (default gpt-6-luna, low effort); CPU.

    OPENAI_API_KEY=... scripts/judge_api_extra.py coherence data/runs/<model>/generations/plus_harmless/*.scored.parquet ...
    OPENAI_API_KEY=... scripts/judge_api_extra.py cutoff [--models ouro,nanbeige,dory,qwen] [--tags kickp,kickp_loops4,kickp_loops6]

coherence  plus_tune.py's coherence step (the persona-vectors coherence prompt on the answer, whole response when the
           CoT never closed) for each ``*.scored.parquet``: ``<dir>/luna/<cand>.coh.parquet`` beside judge_api.py's
           ``--mirror luna`` refusal scores (old: ``<dir>/<cand>.coh.parquet``).
cutoff     cutoff_judge.py's coherent / repetitive / incoherent label for every kick response that hit the token cap:
           ``<tag>/cutoff/luna/<cand>.parquet`` (old: ``<tag>/cutoff/<cand>.parquet``), same columns.

Calls are cached next to each output (``*.cache.jsonl``, as judge_api.py); skip-existing.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]


def run(convs, keys, parse, a, cache):
    from judge_api import chat_all
    vals, recs = asyncio.run(chat_all(convs, keys, parse, a.model, a.effort, a.concurrency, 3, 4096, cache))
    errs = sorted({str(r["error"])[:60] for r in recs if r["error"]})
    return vals, errs


def cmd_coherence(a):
    import pandas as pd
    from judge_api import key
    from persona_vectors import COHERENCE, answer_of, parse_score

    for f in a.files:
        f = Path(f)
        stem = f.name.removesuffix(".scored.parquet")
        out = f.parent / a.tag / f"{stem}.coh.parquet"
        if out.exists():
            continue
        df = pd.read_parquet(f)
        keep = [c for c in ("prompt_idx", "cot_rep", "out_rep") if c in df]
        r = df[keep].copy()
        out.parent.mkdir(exist_ok=True)
        if df.empty:
            r["coherence"] = pd.Series(dtype=float)
            r.to_parquet(out)
            continue
        prompts = [COHERENCE.format(question=q, answer=answer_of(str(o or ""))[:6000]) for q, o in zip(df.prompt, df.output)]
        vals, errs = run([[{"role": "user", "content": p}] for p in prompts],
                         [key(a.model, a.effort, "coherence", p) for p in prompts], parse_score, a,
                         out.with_name(f"{stem}.coh.cache.jsonl"))
        r["coherence"] = [float("nan") if v is None else v for v in vals]
        r.to_parquet(out)
        print(f"-> {out} (mean {r.coherence.mean():.1f}; errors {errs[:2]})", flush=True)


def cmd_cutoff(a):
    import pandas as pd
    from capability import load_problems
    from cutoff_judge import PROMPT, RUNS, TAIL, parse
    from judge_api import key

    problems = {p["id"]: p["question"] for p in load_problems("math500")}
    for m in a.models.split(","):
        for tag in a.tags.split(","):
            d = ROOT / "data" / "runs" / RUNS[m] / "capability" / "math500" / tag
            for old in sorted((d / "cutoff").glob("*.parquet")) if (d / "cutoff").exists() else []:
                out = d / "cutoff" / a.tag / old.name
                if out.exists():
                    continue
                lab = pd.read_parquet(old)
                src = pd.read_parquet(d / old.name, columns=["problem_id", "sample", "response"])
                df = lab[["problem_id", "sample", "tail_rep"]].merge(src, on=["problem_id", "sample"], how="left")
                out.parent.mkdir(exist_ok=True)
                prompts = [PROMPT.format(problem=problems[p], tail=str(r or "")[-TAIL:]) for p, r in zip(df.problem_id, df.response)]
                if prompts:
                    vals, errs = run([[{"role": "user", "content": p}] for p in prompts],
                                     [key(a.model, a.effort, "cutoff", p) for p in prompts], parse, a,
                                     out.with_name(old.stem + ".cache.jsonl"))
                else:
                    vals, errs = [], []
                df["cut"] = vals
                df[["problem_id", "sample", "tail_rep", "cut"]].to_parquet(out)
                print(f"-> {out.relative_to(ROOT)}: {len(df)} cut off, {df.cut.value_counts(dropna=False).to_dict()} "
                      f"(gpt-oss {lab.cut.value_counts().to_dict()}); errors {errs[:2]}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--tag", default="luna", help="output subfolder")
    ap.add_argument("--concurrency", type=int, default=64)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("coherence")
    p.add_argument("files", nargs="+")
    p = sub.add_parser("cutoff")
    p.add_argument("--models", default="ouro,nanbeige,dory,qwen")
    p.add_argument("--tags", default="kickp,kickp_loops4,kickp_loops6")
    a = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    {"coherence": cmd_coherence, "cutoff": cmd_cutoff}[a.cmd](a)


if __name__ == "__main__":
    main()
