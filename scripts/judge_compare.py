#!/usr/bin/env python3
"""Refusal directions under a second judge: labels, class sizes and direction changes (CPU only).

    scripts/judge_compare.py                       # all models with a *.scored_luna.parquet
    scripts/judge_compare.py --models Ouro-1.4B-Thinking

Inputs per model: the gpt-oss-20b scores (``train_clean.scored.parquet``) and the API judge's
(``train_clean.scored_<tag>.parquet``, from ``judge_api.py``), and the cached train activations. Four labellings,
each with ``build_sets.py``'s rules (a CoT is labelled when all its answers agree, a prompt when >= 75% of them do):

  gptoss_score / <tag>_score      the StrongREJECT score, refusal < 0.2, non-refusal > 0.8 (the current pipeline)
  gptoss_refusal / <tag>_refusal  the rubric's yes/no refusal item alone (refused vs complied, quality ignored)

Per labelling: the directions of ``directions.py`` (v4_cot, v12_cot150, cot_last150, v4_baseline) from the same
activations, saved as ``directions/train_<labelling>.pt`` (gptoss_score reproduces ``train.pt``). CoTs with no cached
activations (extracted with ``--labeled-only``) are left out and counted. Writes ``analysis/judge_compare.csv``: class
sizes, label agreement with the current labels, cosine to the current direction (at the experiments' site and the
median over sites), and a held-out AUC at that site (direction from even prompts, labels of the odd prompts under
``<tag>_refusal``).
"""

import argparse

import numpy as np
import pandas as pd
import torch

from loop_steer.paths import run_dir

SCORED = {"Nanbeige4.2-3B": "clean_train/none"}  # default: train_clean
KEY_SITE = {  # the refusal direction the experiments use (scripts/exp.py, run_*.sh, inkling_refusal.sh)
    "Ouro-1.4B-Thinking": ("v4_baseline", (3, 16)), "Nanbeige4.2-3B": ("v4_baseline", (1, 15)),
    "Qwen3-8B": ("v4_baseline", (0, 21)), "gpt-oss-20b": ("v4_baseline", (0, 15)),
    "NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16": ("v4_baseline", (0, 22)), "Inkling-2-Mini": ("v4_cot", (1, 12)),
}
METHODS = {"v4_cot": ("label_cot", "cot"), "v12_cot150": ("label_cot", "cot_first150"),
           "cot_last150": ("label_cot", "cot_last150"), "v4_baseline": ("label_prompt", "eoi")}


def labels(df, col, lower=0.2, upper=0.8, pct=0.75):
    """build_sets.py on per-answer values ``col`` (low = refusal)."""
    df = df.dropna(subset=[col])

    def lab(v, need):
        if (v < lower).sum() >= need:
            return "refusal"
        if (v > upper).sum() >= need:
            return "non_refusal"
        return None

    cots = df.groupby(["prompt_idx", "cot_rep"])[col].agg(lambda s: lab(s.to_numpy(), len(s))).rename("label_cot")
    prompts = df.groupby("prompt_idx")[col].agg(lambda s: lab(s.to_numpy(), int(len(s) * pct))).rename("label_prompt")
    return cots.reset_index().merge(prompts.reset_index(), on="prompt_idx")


def auc(pos, neg):
    """P(proj(pos) > proj(neg)), ties 1/2 (as analyze_directions.py)."""
    if not (len(pos) and len(neg)):
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=list(KEY_SITE))
    ap.add_argument("--tag", default="luna")
    args = ap.parse_args()
    rows = []
    for name in args.models:
        rd = run_dir(name)
        base = rd / "generations" / SCORED.get(name, "train_clean")
        new_path = base.with_name(base.name + f".scored_{args.tag}.parquet")
        if not new_path.exists():
            print(f"{name}: no {new_path.name}; skipped")
            continue
        old, new = pd.read_parquet(base.with_name(base.name + ".scored.parquet")), pd.read_parquet(new_path)
        for d in (old, new):
            d["sr_complied"] = 1 - d.sr_refusal
        labs = {"gptoss_score": labels(old, "sr_score"), "gptoss_refusal": labels(old, "sr_complied"),
                f"{args.tag}_score": labels(new, "sr_score"), f"{args.tag}_refusal": labels(new, "sr_complied")}

        blob = torch.load(rd / "activations" / "train.pt", mmap=True, weights_only=False)
        meta, sites, windows = blob["meta"][["prompt_idx", "cot_rep"]], blob["sites"], list(blob["windows"])
        X = {w: blob["acts"][:, windows.index(w)].float() for w in {w for _, w in METHODS.values()}}
        method, site = KEY_SITE[name]
        s_idx = sites.index(site) if site in sites else sites.index(tuple(site))
        even = (meta.prompt_idx % 2 == 0).to_numpy()
        ref_lab = meta.merge(labs[f"{args.tag}_refusal"], how="left", on=["prompt_idx", "cot_rep"])
        dirs, cur = {}, None
        for lname, lab in labs.items():
            m = meta.merge(lab, how="left", on=["prompt_idx", "cot_rep"])
            missing = len(lab.dropna(subset=["label_cot"])) - int(m.label_cot.notna().sum())
            result = {"sites": sites, "windows": windows}
            for meth, (col, w) in METHODS.items():
                sub = m.drop_duplicates("prompt_idx") if col == "label_prompt" else m
                ref, non = (sub.index[sub[col] == k].to_numpy() for k in ("refusal", "non_refusal"))
                mr, mn = X[w][ref].mean(0), X[w][non].mean(0)
                result[meth] = {"dirs": mr - mn, "n_ref": len(ref), "n_non": len(non), "window": w,
                                "mean_ref": mr, "mean_non": mn}
            torch.save(result, rd / "directions" / f"train_{lname}.pt")
            dirs[lname] = result
            cur = cur or result
            d, d0 = result[method]["dirs"], cur[method]["dirs"]
            cos = torch.nn.functional.cosine_similarity(d, d0, dim=-1)
            cos_site = float(cos[s_idx])
            cos = cos[(d.norm(dim=-1) > 1e-4) & (d0.norm(dim=-1) > 1e-4)]  # layer-0 EOI: identical template, no direction
            # held-out: direction from even prompts under this labelling, AUC on odd prompts' tag_refusal CoT labels
            col, w = METHODS[method]
            sub = m[even].drop_duplicates("prompt_idx") if col == "label_prompt" else m[even]
            ra, na = (sub.index[sub[col] == k].to_numpy() for k in ("refusal", "non_refusal"))
            dh = X[w][ra, s_idx].mean(0) - X[w][na, s_idx].mean(0)
            proj = (X[w][:, s_idx] @ dh).numpy()
            odd = ~even
            a = auc(proj[odd & (ref_lab.label_cot == "refusal").to_numpy()],
                    proj[odd & (ref_lab.label_cot == "non_refusal").to_numpy()])
            agree = lab.merge(labs["gptoss_score"], on=["prompt_idx", "cot_rep"], suffixes=("", "_cur"))
            both = agree.label_cot.notna() & agree.label_cot_cur.notna()
            rows.append({
                "model": name, "labelling": lname, "site": f"{method}:{site}",
                "cot_ref": int((lab.label_cot == "refusal").sum()), "cot_non": int((lab.label_cot == "non_refusal").sum()),
                "prompt_ref": result["v4_baseline"]["n_ref"], "prompt_non": result["v4_baseline"]["n_non"],
                "no_acts": missing,
                "cot_flips_vs_cur": int((both & (agree.label_cot != agree.label_cot_cur)).sum()),
                "cot_lost_vs_cur": int((agree.label_cot.isna() & agree.label_cot_cur.notna()).sum()),
                "cot_gained_vs_cur": int((agree.label_cot.notna() & agree.label_cot_cur.isna()).sum()),
                "cos_site": cos_site, "cos_median": float(cos.median()), "cos_min": float(cos.min()),
                "cos_cot_site": float(torch.nn.functional.cosine_similarity(
                    result["v4_cot"]["dirs"][s_idx], cur["v4_cot"]["dirs"][s_idx], dim=0)),
                "auc_heldout": a,
            })
        print(pd.DataFrame([r for r in rows if r["model"] == name]).drop(columns="model").round(3).to_string(index=False),
              flush=True)
    out = run_dir("Ouro-1.4B-Thinking").parent / "judge_compare.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
