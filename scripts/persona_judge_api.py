#!/usr/bin/env python3
"""Re-judge persona-vector answers with an API judge (default gpt-6-luna, low effort); CPU only.

    OPENAI_API_KEY=... scripts/persona_judge_api.py --models Ouro-1.4B-Thinking Qwen3-8B --file extract_gen

Same prompts as ``persona_vectors.py --stage judge`` (the persona_vectors repo's trait prompt, 0-100 or REFUSAL, and
the coherence prompt) and the same parser; the answer is the text after ``</think>``, first 6000 characters. Reads
``persona/<file>.judged.parquet`` (gpt-oss-20b scores) and writes ``persona/<file>.judged_<tag>.parquet`` with the new
``trait_score`` / ``coherence`` and the old ones as ``gptoss_*``. Calls are cached in ``persona/<file>.<tag>.cache.jsonl``.

For ``extract_gen`` it also prints, per model and trait, how the kept set of ``persona_vectors.py --stage extract``
(trait >= 50 under the positive system prompt, < 50 under the negative one) changes: kept answers per side under each
judge and the share of answers whose keep decision differs. Writes ``persona_judge_compare.csv`` under the runs dir.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="run dir names")
    ap.add_argument("--file", default="extract_gen")
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--tag", default="luna")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--min-kept", type=int, default=10)
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    import pandas as pd

    from judge_api import chat_all, key
    from loop_steer.paths import DATA_ROOT, run_dir
    from persona_vectors import COHERENCE, answer_of, parse_score

    src = DATA_ROOT / "external" / "pv_repo" / "data_generation" / "trait_data_extract"
    rows = []
    for name in args.models:
        wd = run_dir(name) / "persona"
        df = pd.read_parquet(wd / f"{args.file}.judged.parquet")
        df = df.rename(columns={"trait_score": "gptoss_trait_score", "coherence": "gptoss_coherence"})
        evalp = {t: json.loads((src / f"{t}.json").read_text())["eval_prompt"] for t in df.trait.unique()}
        ans = [answer_of(r)[:6000] for r in df.response]
        prompts = ([evalp[t].format(question=q, answer=a) for t, q, a in zip(df.trait, df.question, ans)]
                   + [COHERENCE.format(question=q, answer=a) for q, a in zip(df.question, ans)])
        convs = [[{"role": "user", "content": p}] for p in prompts]
        keys = [key(args.model, args.effort, "persona", p) for p in prompts]
        vals, recs = asyncio.run(chat_all(convs, keys, parse_score, args.model, args.effort, args.concurrency, 3, 4096,
                                          wd / f"{args.file}.{args.tag}.cache.jsonl"))
        n = len(df)
        df["trait_score"] = [float("nan") if v is None else v for v in vals[:n]]
        df["coherence"] = [float("nan") if v is None else v for v in vals[n:]]
        df["judge_failed"] = [v is None for v in vals[:n]]
        out = wd / f"{args.file}.judged_{args.tag}.parquet"
        df.to_parquet(out)
        calls = [u for r in {r["key"]: r for r in recs}.values() for u in r["usage"]]
        tok_in, tok_out = sum(u["in"] for u in calls), sum(u["out"] for u in calls)
        errs = sorted({r["error"][:80] for r in recs if r["error"]})
        print(f"{name}: {n} answers -> {out}; {len(calls)} calls, {tok_in / 1e6:.2f}M in / {tok_out / 1e6:.2f}M out "
              f"tokens; never parsed {int(df.judge_failed.sum())}; errors {errs[:3]}", flush=True)

        if args.file != "extract_gen":
            continue
        for t, g in df.groupby("trait"):
            def kept(col):
                return ((g.polarity == "pos") & (g[col] >= 50)) | ((g.polarity == "neg") & (g[col] < 50))
            k0, k1 = kept("gptoss_trait_score"), kept("trait_score")
            both = g.gptoss_trait_score.notna() & g.trait_score.notna()
            rows.append({
                "model": name, "trait": t,
                "pos_kept_gptoss": int((k0 & (g.polarity == "pos")).sum()),
                "pos_kept_new": int((k1 & (g.polarity == "pos")).sum()),
                "neg_kept_gptoss": int((k0 & (g.polarity == "neg")).sum()),
                "neg_kept_new": int((k1 & (g.polarity == "neg")).sum()),
                "keep_differs": float((k0 != k1).mean()),
                "trait_corr": float(g.gptoss_trait_score[both].corr(g.trait_score[both])),
                "coh_corr": float(g.gptoss_coherence.corr(g.coherence)),
                "refusal_gptoss": float(g.gptoss_trait_score.isna().mean()),
                "refusal_new": float(g.trait_score.isna().mean()),
            })
        part = pd.DataFrame([r for r in rows if r["model"] == name])
        part["skipped_gptoss"] = part[["pos_kept_gptoss", "neg_kept_gptoss"]].min(axis=1) < args.min_kept
        part["skipped_new"] = part[["pos_kept_new", "neg_kept_new"]].min(axis=1) < args.min_kept
        print(part.drop(columns="model").round(3).to_string(index=False), flush=True)
    if rows:
        out = run_dir(args.models[0]).parent / f"persona_judge_compare_{args.file}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
