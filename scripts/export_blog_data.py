#!/usr/bin/env python3
"""Export the numbers behind the interactive blog post (docs/blog/index.html) to docs/blog/data.js.

    .venv/bin/python scripts/export_blog_data.py

CPU only, reads data/runs/<model>/. Every number the page shows comes from this file, and each block records the
files it was computed from. Refusal: the `all_refusal` quantity of summarize.py / grid_ci.py, pooled over the test
tags given, with a prompt bootstrap; per-prompt rates for the first 100 test prompts drive the dot grids. Kicks:
scripts/kick_summary.py. Persona: analysis/persona_loops*.csv and persona_ablate*.csv, example replies from
persona/steer_loops*.judged.parquet.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "runs"
OUT = ROOT / "docs" / "blog" / "data.js"
RNG = np.random.default_rng(0)
NDOTS = 100

MODELS = {"ouro": "Ouro-1.4B-Thinking", "nanbeige": "Nanbeige4.2-3B",
          "dory": "NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16", "qwen": "Qwen3-8B"}


# judge of record for every reported score: v2 = gpt-6-luna (the default since 2026-11-21), v1 = gpt-oss-20b. Under v2,
# refusal comes from each run's luna/ mirror (judge_api.py --mirror luna), persona trait / coherence from
# persona/*.judged_luna.parquet (persona_judge_api.py), flip-sign coherence from luna/*.coh.parquet and kick cut-off
# labels from cutoff/luna/ (judge_api_extra.py); a missing luna file is an error, so judges never mix. The gpt-oss-20b
# files are never overwritten (`--judge v1` / BLOG_JUDGE=v1 restores them). Directions, persona vectors and every
# steering strength were chosen with gpt-oss-20b, and the flip-sign tuning chart (plus_tune.json) keeps its scores.
JUDGES = {"v1": "gpt-oss-20b", "v2": "gpt-6-luna"}
JUDGE = os.environ.get("BLOG_JUDGE", "v2")


def read_scored(f, columns=None):
    """One run file's judged answers under the judge of record (``columns``: None = all). Under v2 a file not yet
    judged by luna is an error, so a switched export never mixes judges; answers luna's API would not score (its
    biology filter) keep gpt-oss-20b's verdict."""
    if JUDGE == "v1":
        return pd.read_parquet(f, columns=columns)
    g = f.parent / "luna" / f.name
    if not g.exists():
        raise FileNotFoundError(f"judgment v2: {g} not judged yet (scripts/judge_api.py --mirror luna {f})")
    import pyarrow.parquet as pq
    have = pq.read_schema(g).names
    want = None if columns is None else [c for c in dict.fromkeys(columns + ["gptoss_sr_refusal", "judge_error"]) if c in have]
    x = pd.read_parquet(g, columns=want)
    if "gptoss_sr_refusal" in x:  # empty files carry the original columns only
        blocked = x.sr_refusal.isna() & x.judge_error.astype(str).str.contains("bio_policy")
        x["sr_refusal"] = x.sr_refusal.where(~blocked, x.gptoss_sr_refusal)
    return x if columns is None else x.reindex(columns=columns)


def gen_dirs(model, tags):
    return [RUNS / MODELS[model] / "generations" / t for t in tags]


def per_prompt(dirs, name):
    """Per-prompt refusal (mean over CoT samples of the mean judge refusal over answers; unclosed CoTs judged whole)."""
    parts = [read_scored(d / sub / f"{name}.scored.parquet", ["prompt_idx", "cot_rep", "sr_refusal"])
             for d in dirs for sub in (".", "invalid") if (d / sub / f"{name}.scored.parquet").exists()]
    if not parts:
        raise FileNotFoundError(f"{name} in {dirs}")
    df = pd.concat(parts)
    df["sr_refusal"] = df.sr_refusal.fillna(0)
    per_sample = df.groupby(["prompt_idx", "cot_rep"]).sr_refusal.mean()
    return per_sample, per_sample.groupby("prompt_idx").mean()


def refusal(dirs, name, dots=True, keep=None):
    """``keep``: only these prompt indices (to pair two runs on the prompts both have)."""
    per_sample, pp = per_prompt(dirs, name)
    if keep is not None:
        per_sample = per_sample[per_sample.index.get_level_values("prompt_idx").isin(keep)]
        pp = pp[pp.index.isin(keep)]
    x = pp.to_numpy()
    boots = x[RNG.integers(0, len(x), (1000, len(x)))].mean(1)
    out = {"cand": name, "r": round(100 * per_sample.mean(), 1), "lo": round(100 * np.percentile(boots, 2.5), 1),
           "hi": round(100 * np.percentile(boots, 97.5), 1), "n": int(len(x))}
    if dots:
        out["dots"] = [None if i not in pp.index else round(float(pp[i]), 2) for i in range(NDOTS)]
    return out


def subsets(n):
    from itertools import combinations
    return [c for k in range(1, n + 1) for c in combinations(range(n), k)]


def key(units):
    return "".join(str(u + 1) for u in units)


# ---------------------------------------------------------------- refusal switchboard
def grid_ouro():
    dirs = gen_dirs("ouro", ["grid_test", "grid_test_b"])
    base = "ablate_v4_baseline_t3.l16"
    push = "actadd_v4_baseline_t3.l16_c-1"
    erase, pushes = {}, {}
    for s in subsets(4):
        a = base if len(s) == 4 else f"{base}_apply{'-'.join(map(str, s))}"
        p = push if len(s) == 4 else f"{push}_apply{'-'.join(map(str, s))}"
        erase[key(s)], pushes[key(s)] = refusal(dirs, a), refusal(dirs, p)
    return {"units": 4, "unit": "loop", "clean": refusal(dirs, "none"), "random": refusal(dirs, "ablate_random_t3.l16"),
            "erase": erase, "push": pushes, "push_note": "same vector added at layer 16 of each selected loop, c = -1"}


def grid_nanbeige():
    dirs = gen_dirs("nanbeige", ["grid_test", "grid_test_b"])
    base, push = "ablate_v4_baseline_t1.l15", "actadd_v4_baseline_t1.l15_c-2_rel1"
    erase = {"1": refusal(dirs, f"{base}_apply0"), "2": refusal(dirs, f"{base}_apply1"), "12": refusal(dirs, base)}
    pushes = {"1": refusal(dirs, f"{push}_apply0"), "2": refusal(dirs, f"{push}_apply1"), "12": refusal(dirs, push)}
    return {"units": 2, "unit": "loop", "clean": refusal(dirs, "none"), "random": refusal(dirs, "ablate_random_t1.l15"),
            "erase": erase, "push": pushes, "push_note": "norm-matched vector at layer 15 of each selected loop, c = -2"}


def grid_dory2():
    dirs = gen_dirs("dory", ["grid_test", "grid_test_b"])
    base, push = "ablate_hh_eoi_t0.l26", "actadd_hh_eoi_t0.l26_c-1"
    erase = {"1": refusal(dirs, f"{base}_apply0"), "2": refusal(dirs, f"{base}_apply1"),
             "12": refusal(dirs, f"{base}_apply0-1")}
    pushes = {"1": refusal(dirs, f"{push}_apply0"), "2": refusal(dirs, f"{push}_apply1"), "12": refusal(dirs, push)}
    return {"units": 2, "unit": "loop", "clean": refusal(dirs, "none"), "random": refusal(dirs, "ablate_random_t0.l26"),
            "erase": erase, "push": pushes, "push_note": "same vector added at sublayer 26 of each selected loop, c = -1"}


def grid_dory4():
    dirs = gen_dirs("dory", ["grid_test4", "grid_test4_b"])
    base, push = "ablate_hh_eoi_t0.l26", "actadd_hh_eoi_t0.l26_c-1"
    erase, pushes = {}, {}
    for s in subsets(4):
        erase[key(s)] = refusal(dirs, f"{base}_apply{'-'.join(map(str, s))}")
        pushes[key(s)] = refusal(dirs, push if len(s) == 4 else f"{push}_apply{'-'.join(map(str, s))}")
    return {"units": 4, "unit": "loop", "clean": refusal(dirs, "none"), "random": refusal(dirs, "ablate_random_t0.l26"),
            "erase": erase, "push": pushes, "push_note": "same vector added at sublayer 26 of each selected loop, c = -1",
            "note": "Dory run with 4 loops; it was trained with 2"}


def grid_qwen():
    dirs = gen_dirs("qwen", ["grid_test", "grid_test_b"])
    q = ["0-8", "9-17", "18-26", "27-35"]
    ablate_names = {"12": "0-8-9-17", "13": "0-8-18-26", "14": "0-8-27-35", "23": "9-26", "24": "9-17-27-35",
                    "34": "18-26-27-35", "123": "0-26", "124": "0-17-27-35", "134": "0-8-18-35", "234": "9-35"}
    at = [3, 12, 21, 30]
    erase, pushes = {}, {}
    for s in subsets(4):
        k = key(s)
        if len(s) == 4:
            a = "ablate_v4_baseline_l21"
        elif len(s) == 1:
            a = f"ablate_v4_baseline_l21_layers{q[s[0]]}"
        else:
            a = f"ablate_v4_baseline_l21_layers{ablate_names[k]}"
        p = (f"actadd_v4_baseline_l{at[s[0]]}_c-1" if len(s) == 1
             else f"actadd_v4_baseline_l21_c-1_layers{'-'.join(str(at[u]) for u in s)}")
        erase[k], pushes[k] = refusal(dirs, a), refusal(dirs, p)
    return {"units": 4, "unit": "quarter", "clean": refusal(dirs, "none"), "random": refusal(dirs, "ablate_random_l21"),
            "erase": erase, "push": pushes,
            "push_note": "each quarter's own direction added at layer 3 / 12 / 21 / 30, c = -1",
            "layers": ["0-8", "9-17", "18-26", "27-35"]}


def plus_harmless():
    """E14: adding the refusal direction on harmless prompts (first loop / last loop / every loop)."""
    out = {}
    spec = {"nanbeige": ("plus_harmless", "v4_baseline_t1.l15_c4_rel1", "_apply0", "_apply1"),
            "dory": ("plus_harmless", "hh_eoi_t0.l26_c2_rel1", "_apply0", "_apply1"),
            "qwen": ("plus_harmless", "v4_baseline_l21_c4_rel1", "_at3", "_at30")}
    for m, (tag, stem, first, last) in spec.items():
        d = [RUNS / MODELS[m] / "generations" / tag]
        allname = f"actadd_{stem}_at3-12-21-30" if m == "qwen" else f"actadd_{stem}"
        out[m] = {"clean": refusal(d, "none", dots=False)["r"], "first": refusal(d, f"actadd_{stem}{first}", dots=False)["r"],
                  "last": refusal(d, f"actadd_{stem}{last}", dots=False)["r"], "all": refusal(d, allname, dots=False)["r"],
                  "cand": stem}
    # added later: plain rates (no bootstrap), so the shared RNG and every interval exported after this stay unchanged
    rate = lambda d, name: round(100 * per_prompt(d, name)[0].mean(), 1)  # noqa: E731
    d4, stem = [RUNS / MODELS["dory"] / "generations" / "plus_harmless4"], "hh_eoi_t0.l26_c2_rel1"
    out["dory4"] = {"clean": rate(d4, "none"), "first": rate(d4, f"actadd_{stem}_apply0"), "last": rate(d4, f"actadd_{stem}_apply3"),
                    "all": rate(d4, f"actadd_{stem}"), "cand": stem}
    # Qwen's middle quarters (at the offsets of layers 12 and 21); null until those runs exist
    dq = [RUNS / MODELS["qwen"] / "generations" / "plus_harmless"]
    for q, at in (("q2", 12), ("q3", 21)):
        f = f"actadd_{spec['qwen'][1]}_at{at}"
        out["qwen"][q] = rate(dq, f) if (dq[0] / f"{f}.scored.parquet").exists() else None
    # Ouro at 4x its tuned |c| like the others (plain rates, as above), and for every model a random direction at the
    # same strength on every loop / quarter, plus the share of reasoning traces that never close (judged whole):
    # where the random direction also "refuses", the bar measures a broken model rather than refusal
    do = [RUNS / MODELS["ouro"] / "generations" / "plus_harmless"]
    so = "v4_baseline_t3.l16_c4_rel1"
    out["ouro"] = {"clean": rate(do, "none"), "first": rate(do, f"actadd_{so}_apply0"), "last": rate(do, f"actadd_{so}_apply3"),
                   "all": rate(do, f"actadd_{so}"), "cand": so}

    def unclosed(d, name):
        n = {sub: (pd.read_parquet(d[0] / sub / f"{name}.scored.parquet", columns=["prompt_idx", "cot_rep"])
                   .groupby(["prompt_idx", "cot_rep"]).ngroups if (d[0] / sub / f"{name}.scored.parquet").exists() else 0)
             for sub in (".", "invalid")}
        return round(100 * n["invalid"] / max(1, n["."] + n["invalid"]), 1)

    rnd = {"ouro": ("plus_harmless", "actadd_random_t3.l16_c4_rel1", "actadd_" + so, "_apply0", "_apply3"),
           "nanbeige": ("plus_harmless", "actadd_random_t1.l15_c4_rel1", "actadd_" + spec["nanbeige"][1], "_apply0", "_apply1"),
           "dory": ("plus_harmless", "actadd_random_t0.l26_c2_rel1", "actadd_" + spec["dory"][1], "_apply0", "_apply1"),
           "dory4": ("plus_harmless4", "actadd_random_t0.l26_c2_rel1", "actadd_" + stem, "_apply0", "_apply3"),
           "qwen": ("plus_harmless", "actadd_random_l21_c4_rel1_at3-12-21-30", "actadd_" + spec["qwen"][1], "_at3", "_at30")}
    for m, (tag, rname, base, first, last) in rnd.items():
        d = [RUNS / MODELS["dory" if m == "dory4" else m] / "generations" / tag]
        allname = f"{base}_at3-12-21-30" if m == "qwen" else base
        out[m]["random"] = rate(d, rname)
        out[m]["unclosed"] = {"clean": unclosed(d, "none"), "first": unclosed(d, base + first), "last": unclosed(d, base + last),
                              "all": unclosed(d, allname), "random": unclosed(d, rname)}
        if m == "qwen":
            out[m]["unclosed"].update({q: unclosed(d, f"{base}_at{at}") for q, at in (("q2", 12), ("q3", 21))})

    # E15: +u's strength tuned on held-out harmless prompts for refusal vs judged coherence (scripts/plus_tune.py), the
    # tuning table, and the test runs at that strength with their coherence (each model keys "tune" / "tuned")
    e2c = {"ouro": 1.0, "nanbeige": 1.0, "dory": 0.5, "qwen": 1.0}
    site = {"ouro": "v4_baseline_t3.l16", "nanbeige": "v4_baseline_t1.l15", "dory": "hh_eoi_t0.l26", "qwen": "v4_baseline_l21"}
    loops = {"ouro": 4, "nanbeige": 2, "dory": 2, "qwen": 4}

    def coh(d, name):  # v2: <dir>/[invalid/]luna/<cand>.coh.parquet (judge_api_extra.py coherence)
        lun = "luna" if JUDGE == "v2" else "."
        parts = [pd.read_parquet(d[0] / sub / lun / f"{name}.coh.parquet") for sub in (".", "invalid")
                 if (d[0] / sub / f"{name}.coh.parquet").exists()]
        return round(float(pd.concat(parts).groupby(["prompt_idx", "cot_rep"]).coherence.mean().mean()), 1) if parts else None

    for m in ("ouro", "nanbeige", "dory", "qwen"):
        tj = RUNS / MODELS[m] / "analysis" / "plus_tune.json"
        if not tj.exists():
            continue
        tune = json.load(open(tj))
        c, cl = tune["c"], next(r for r in tune["table"] if r["candidate"] == "none")
        out[m]["tune"] = {"c": c, "mult": round(float(c) / e2c[m], 3), "any_ok": tune["any_ok"],
                          "clean": {"refusal": round(100 * cl["refusal"], 1), "coherence": cl["coherence"], "capped": round(100 * cl["capped_frac"], 1)},
                          "rows": [{"mult": round(float(r["c"]) / e2c[m], 3), "c": r["c"], "refusal": round(100 * r["refusal"], 1),
                                    "coherence": r["coherence"], "capped": round(100 * r["capped_frac"], 1), "ok": r["ok"]}
                                   for r in tune["table"] if "c" in r]}
        for key, tag, n in [(m, "plus_harmless", loops[m])] + ([("dory4", "plus_harmless4", 4)] if m == "dory" else []):
            d = [RUNS / MODELS[m] / "generations" / tag]
            base = f"actadd_{site[m]}_c{c}_rel1"
            units = ({"first": "_at3", "q2": "_at12", "q3": "_at21", "last": "_at30", "all": "_at3-12-21-30"} if m == "qwen"
                     else {"first": "_apply0", "last": f"_apply{n - 1}", "all": ""})
            names = {k: base + u for k, u in units.items()}
            names["random"] = f"actadd_random_{site[m].split('_')[-1]}_c{c}_rel1" + ("_at3-12-21-30" if m == "qwen" else "")
            names["clean"] = "none"
            if not all((d[0] / f"{v}.scored.parquet").exists() for v in names.values()):
                continue
            if m != "qwen":  # E17: the middle loops of the 4-loop runs, when they exist
                for k in range(1, n - 1):
                    if (d[0] / f"{base}_apply{k}.scored.parquet").exists():
                        names[f"l{k + 1}"] = f"{base}_apply{k}"
            t = {k: rate(d, v) for k, v in names.items()}
            t.update({"c": c, "mult": round(float(c) / e2c[m], 3), "coh": {k: coh(d, v) for k, v in names.items()},
                      "unclosed": {k: unclosed(d, v) for k, v in names.items()}})
            out[key]["tuned"] = t
    return out


# ---------------------------------------------------------------- extra loops
def paired(trained, extra, clean):
    """The same steering at the trained loop count and with two extra loops, on the prompts all three runs share (each
    arg: (dirs, candidate)); ``gain``: the per-prompt paired difference extra - trained, in points, prompt bootstrap."""
    pps = [per_prompt(d, n)[1] for d, n in (trained, extra, clean)]
    keep = pps[0].index.intersection(pps[1].index).intersection(pps[2].index)
    diff = 100 * (pps[1][keep] - pps[0][keep]).to_numpy()
    boots = diff[RNG.integers(0, len(diff), (1000, len(diff)))].mean(1)
    return {"at_trained": refusal(*trained, keep=keep), "at_extra": refusal(*extra, keep=keep),
            "clean": refusal(*clean, dots=False, keep=keep),
            "gain": {"d": round(float(diff.mean()), 1), "lo": round(float(np.percentile(boots, 2.5)), 1),
                     "hi": round(float(np.percentile(boots, 97.5)), 1), "n": int(len(diff))}}


def extra_loops():
    """Every pair on the same prompts: E19 adds the 6-loop Ouro runs (grid_test6_b) and Dory's push on prompts 250-486;
    until those exist, the pair falls back to the prompts both runs have."""
    d2 = gen_dirs("dory", ["grid_test", "grid_test_b"])
    d4 = gen_dirs("dory", ["grid_test4", "grid_test4_b"])
    o4 = gen_dirs("ouro", ["grid_test", "grid_test_b"])
    o6 = [d for d in gen_dirs("ouro", ["grid_test6", "grid_test6_b"]) if d.exists()]
    return {
        "dory": {"trained": 2, "extra": 4,
                 "erase": paired((d2, "ablate_hh_eoi_t0.l26_apply0-1"), (d4, "ablate_hh_eoi_t0.l26_apply0-1"), (d4, "none")),
                 "push": paired((d2, "actadd_hh_eoi_t0.l26_c-1"), (d4, "actadd_hh_eoi_t0.l26_c-1_apply0-1"), (d4, "none"))},
        "ouro": {"trained": 4, "extra": 6,
                 "erase": paired((o4, "ablate_v4_baseline_t3.l16"), (o6, "ablate_v4_baseline_t3.l16_apply0-1-2-3"), (o6, "none")),
                 "push": paired((o4, "actadd_v4_baseline_t3.l16_c-1.0_rel1"),
                                (o6, "actadd_v4_baseline_t3.l16_c-1.0_rel1_apply0-1-2-3"), (o6, "none"))},
    }


# ---------------------------------------------------------------- personas
PERSONA_FILES = {"ouro": ("persona_loops.csv", "steer_loops.judged.parquet"),
                 "nanbeige": ("persona_loops.csv", "steer_loops.judged.parquet"),
                 "dory": ("persona_loops.csv", "steer_loops.judged.parquet"),
                 "dory4": ("persona_loops_loops4.csv", "steer_loops_loops4.judged.parquet"),
                 "qwen": ("persona_loops.csv", "steer_loops.judged.parquet")}


def clip(text, n=420):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def clip_md(text, n=800):
    """Keep the reply's markdown (line breaks, lists, bold); cut at a line or word boundary and close an open **."""
    text = re.sub(r"[ \t]+\n", "\n", str(text).replace("\r\n", "\n")).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= n:
        return text
    cut = text[:n]
    nl = cut.rfind("\n")
    cut = cut[:nl] if nl > 0.6 * n else cut.rsplit(" ", 1)[0]
    if cut.count("**") % 2:
        cut += "**"
    return cut.rstrip() + " …"


def persona_judged(m, pq):
    """persona/<steer_*>.judged.parquet under the judge of record (v2: its ``.judged_luna.parquet`` from
    scripts/persona_judge_api.py, same columns with luna's trait_score / coherence)."""
    f = RUNS / MODELS[m] / "persona" / pq
    if JUDGE == "v2":
        f = f.with_name(pq.replace(".judged.parquet", ".judged_luna.parquet"))
        if not f.exists():
            raise FileNotFoundError(f"judgment v2: {f} missing (scripts/persona_judge_api.py --file {pq.split('.')[0]})")
    return pd.read_parquet(f)


def persona_table(m, csv):
    """analysis/persona_<name>.csv; under v2 the same summary (persona_steer.py's summary stage, verbatim) recomputed
    from luna's scores, round-tripped through CSV so its dtypes match the file's."""
    if JUDGE == "v1":
        return pd.read_csv(RUNS / MODELS[m] / "analysis" / csv)
    import io
    name = csv.removeprefix("persona_").removesuffix(".csv")
    ablate = name.startswith("ablate")
    df = persona_judged(m, f"steer_{name}.judged.parquet")
    g = df.groupby(["trait", "cond", "units", "layer", "c"]).agg(
        trait_score=("trait_score", "mean"), coherence=("coherence", "mean"),
        unscored=("trait_score", lambda s: s.isna().mean()), n=("trait_score", "size")).reset_index()
    clean = g[g.cond == ("prompted" if ablate else "clean")].set_index("trait")
    g["trait_clean"], g["coherence_clean"] = g.trait.map(clean.trait_score), g.trait.map(clean.coherence)
    g["gain"], g["coh_drop"] = g.trait_score - g.trait_clean, g.coherence_clean - g.coherence
    if ablate:
        base = g[g.cond == "noprompt"].set_index("trait").trait_score
        g["trait_noprompt"] = g.trait.map(base)
        g["removed"] = (g.trait_clean - g.trait_score) / (g.trait_clean - g.trait_noprompt)
    buf = io.StringIO()
    g.to_csv(buf, index=False)
    buf.seek(0)
    return pd.read_csv(buf)


def personas():
    out = {}
    for name, (csv, pq) in PERSONA_FILES.items():
        m = "dory" if name == "dory4" else name
        d = persona_table(m, csv)
        r = persona_judged(m, pq)
        r["units"] = r["units"].fillna("").astype(str)
        traits = {}
        for t, g in d.groupby("trait"):
            g = g.copy()
            g["units"] = g["units"].astype(str)
            rt = r[r.trait == t]
            # example question: the one with the largest mean trait score when every loop is steered
            allu = g.units.iloc[np.argmax(g.units.str.len())]
            steered_all = rt[rt.units == allu]
            q = steered_all.groupby("q").trait_score.mean().idxmax() if len(steered_all) else rt.q.iloc[0]
            ex = {}
            for u, gg in rt[rt.q == q].groupby("units"):
                row = gg.sort_values("sample").iloc[0]
                ex[u or "clean"] = {"text": clip_md(row.response), "score": None if pd.isna(row.trait_score) else round(float(row.trait_score))}
            traits[t] = {"clean": round(float(g.trait_clean.iloc[0]), 1), "coh_clean": round(float(g.coherence_clean.iloc[0]), 1),
                         "by_units": {u: {"score": round(float(x.trait_score), 1), "gain": round(float(x.gain), 1),
                                          "coh": round(float(x.coherence), 1), "c": float(x.c), "layer": int(x.layer)}
                                      for u, x in g.set_index("units").iterrows()},
                         "question": clip(rt[rt.q == q].question.iloc[0], 200), "examples": ex}
        out[name] = {"traits": traits, "source": f"analysis/{csv}, persona/{pq}", **persona_coverage(m)}
    return out


PERSONA_TRAITS = ["optimistic", "humorous", "sycophantic", "hallucinating", "apathetic", "impolite", "evil"]


def persona_coverage(m):
    """Why a trait is missing for a model: no vector (persona_vectors.py drops a trait the model will not play even when
    told to), or not steerable (persona_tune.json: on every loop at once, no tuned strength that keeps replies coherent
    moves the trait by 10 points; ``best`` is the largest such gain)."""
    import torch
    vecs = torch.load(RUNS / MODELS[m] / "directions" / "persona.pt", weights_only=False)
    tune = json.load(open(RUNS / MODELS[m] / "analysis" / "persona_tune.json"))
    return {"vector": [t for t in PERSONA_TRAITS if t in vecs],
            "tune": {t: {"steerable": bool(tune[t].get("steerable")), "best": round(float(tune[t]["gain"]), 1) if "gain" in tune[t] else None}
                     for t in PERSONA_TRAITS if t in tune}}


def persona_ablation():
    out = {}
    for name, csv in [("ouro", "persona_ablate.csv"), ("nanbeige", "persona_ablate.csv"), ("dory", "persona_ablate.csv"),
                      ("dory4", "persona_ablate_loops4.csv"), ("qwen", "persona_ablate.csv")]:
        m = "dory" if name == "dory4" else name
        d = persona_table(m, csv)
        d["units"] = d["units"].astype(str)
        g = d.groupby(["cond", "units"], sort=False)[["trait_score", "coherence"]].mean().round(1).reset_index()
        traits = {}
        for t, x in d.groupby("trait", sort=False):
            traits[t] = {"prompted": round(float(x.trait_clean.iloc[0]), 1), "noprompt": round(float(x.trait_noprompt.iloc[0]), 1),
                         "rows": [{"cond": r.cond, "units": r.units, "score": round(float(r.trait_score), 1),
                                   "coh": round(float(r.coherence), 1)} for r in x.itertuples()]}
        # share of the prompted trait removed, pooled over every trait: the points the erasure takes away, summed over
        # traits, over the points the system prompt adds, summed over traits (a trait the prompt barely moves counts
        # for little, instead of blowing up a per-trait ratio)
        x = d[~d.cond.isin(["prompted", "noprompt"])]
        pooled = (100 * (x.trait_clean - x.trait_score).groupby(x.cond).sum()
                  / (x.trait_clean - x.trait_noprompt).groupby(x.cond).sum()).round(1)
        out[name] = {"vector": persona_coverage(m)["vector"], "rows": g.to_dict("records"), "prompted": round(float(d.trait_clean.mean()), 1),
                     "unprompted": round(float(d.trait_noprompt.mean()), 1) if "trait_noprompt" in d else None,
                     "traits": traits, "removed": pooled.to_dict(),
                     "removed_traits": sorted(d.trait.unique()), "source": f"analysis/{csv}"}
    return out


# ---------------------------------------------------------------- Dory re-injection
REINJECT_CONDS = ["clean", "abl12", "abl12+umean34", "abl12+uxmean34", "abl12+umean1234", "abl12+uxmean1234", "umean34",
                  "abl12+mean34", "abl12+smean34", "abl12+rmean34", "abl12+rsmean34", "abl12+drop34", "abl12+flat34"]
REINJECT_FILES = ["dory_reinject.csv", "dory_reinject_span.csv", "dory_reinject_u.csv"]  # earlier files win on overlap


def reinjection():
    """scripts/dory_reinject.py (4 laps, teacher-forced): refusing - complying separation along u at every sublayer input
    in run order (entry stack, laps 1-4, exit stack), CoT window, x1000; mean CoT next-token loss per condition."""
    rd = RUNS / MODELS["dory"] / "analysis"
    d = pd.concat([pd.read_csv(rd / f) for f in REINJECT_FILES])
    d = d.drop_duplicates(subset=["condition", "kind", "loop", "layer", "what", "window", "dir"])  # earlier file wins
    order = [(1, l) for l in range(16)] + [(t, l) for t in range(1, 5) for l in range(16, 56)] + [(1, l) for l in range(56, 72)]
    out = {}
    for c in REINJECT_CONDS:
        x = d[(d.condition == c) & (d.kind == "layer") & (d.dir == "u") & (d.window == "cot")].set_index(["loop", "layer"]).sep
        nl = d[(d.condition == c) & (d.kind == "nll")].iloc[0]
        out[c] = {"curve": [round(1000 * float(x[k]), 2) for k in order], "nll": round(float(nl.mean_ref + nl.mean_com) / 2, 3)}
    e = d[(d.kind == "entry") & (d.window == "cot")]
    pick = lambda c, t, what, dr: e[(e.condition == c) & (e.loop == t) & (e.what == what) & (e.dir == dr)].iloc[0]  # noqa: E731
    auc = lambda *k: round(float(pick(*k).auc), 2)  # noqa: E731
    return {"conds": out, "entry": {"state_u_l3": auc("abl12", 3, "state", "u"), "state_dx_l3": auc("abl12", 3, "state", "dx"),
                                    "x_u": auc("clean", 1, "x", "u"), "x_dx_clean": auc("clean", 1, "x", "dx"),
                                    # refusing - complying separation of the copy along u, as added and with umean (x1000)
                                    "x_u_sep": round(1000 * float(pick("clean", 3, "x", "u").sep), 4),
                                    "x_u_sep_umean": round(1000 * float(pick("abl12+umean34", 3, "x", "u").sep), 4)},
            "traces": "40 refusing and 40 complying reasoning traces (train split), first 256 reasoning tokens",
            "source": f"analysis/{', '.join(REINJECT_FILES)} (scripts/dory_reinject.py)"}


# ---------------------------------------------------------------- kicks
KICKS = {"ouro": ("kickp", ["kickp", "kickp_all", "kickp_all_indep"]), "ouro6": ("kickp_loops6", ["kickp_loops6"]),
         "nanbeige": ("kickp", ["kickp", "kickp_both"]), "dory": ("kickp", ["kickp", "kickp_all"]),
         "dory4": ("kickp_loops4", ["kickp_loops4", "kickp_loops4_all", "kickp_loops4_all_indep"]),
         "qwen": ("kickp", ["kickp", "kickp_all", "kickp_all_indep"])}


KICK_OUT = ["solved", "wrong answer", "no answer", "cut off, still reasoning", "cut off, repeating itself",
            "cut off, incoherent", "cut off"]


def kick_outcomes(capdir, cand):
    """Per problem in dot order (sorted ids, first NDOTS), an index into KICK_OUT: solved; finished with a wrong answer;
    finished without one; or hit the token cap, split by scripts/cutoff_judge.py's label (E18; plain "cut off" until it
    has run)."""
    d = pd.read_parquet(capdir / f"{cand}.parquet").sort_values(["problem_id", "sample"]).drop_duplicates("problem_id")
    cf = capdir / "cutoff" / ("luna" if JUDGE == "v2" else ".") / f"{cand}.parquet"  # v2: judge_api_extra.py cutoff
    if JUDGE == "v2" and (capdir / "cutoff" / f"{cand}.parquet").exists() and not cf.exists():
        raise FileNotFoundError(f"judgment v2: {cf} missing (scripts/judge_api_extra.py cutoff)")
    cut = pd.read_parquet(cf).drop_duplicates("problem_id").set_index("problem_id").cut.to_dict() if cf.exists() else {}
    lab = {"coherent": 3, "repetitive": 4, "incoherent": 5}
    codes = []
    for r in d.head(NDOTS).itertuples():
        if r.correct:
            codes.append(0)
        elif r.hit_limit:
            codes.append(lab.get(cut.get(r.problem_id), 6))
        else:
            codes.append(1 if r.pred is not None and not pd.isna(r.pred) else 2)
    return codes


def kicks():
    out = {"outcomes": KICK_OUT}
    with tempfile.TemporaryDirectory(dir=ROOT / "scratch") as tmp:
        for name, (tag, kl_tags) in KICKS.items():
            m = MODELS[name.rstrip("46")]
            csv = Path(tmp) / f"{name}.csv"
            cmd = [sys.executable, str(ROOT / "scripts" / "kick_summary.py"), "--model", m, "--tag", tag,
                   "--n-boot", "500", "--csv", str(csv)]
            for t in kl_tags:
                if (RUNS / m / "analysis" / "kick_kl" / f"{t}.parquet").exists():
                    cmd += ["--kl-tag", t]
            subprocess.run(cmd, check=True, capture_output=True)
            s = pd.read_csv(csv)
            capdir = RUNS / m / "capability" / "math500" / tag
            rows = []
            for _, x in s.iterrows():
                c = x.candidate
                acc = pd.read_parquet(capdir / f"{c}.parquet").groupby("problem_id").correct.mean().sort_index()
                site = None
                if c != "none":
                    mm = re.match(r"kick_randp\d+_(?:t(\d+)\.)?l(\d+)_eps([\d.]+)", c)
                    site = {"loop": None if mm.group(1) is None else int(mm.group(1)), "layer": int(mm.group(2)),
                            "eps": float(mm.group(3))}
                apply = "" if pd.isna(x.get("apply", np.nan)) else str(x["apply"])
                outc = kick_outcomes(capdir, c)
                rows.append({"cand": c, "site": site, "out": outc, "every": bool(apply) and "indep" not in apply and "," in apply,
                             "indep": "indep" in apply or "indep" in c, "acc": round(float(x.acc), 3),
                             "lo": round(float(x.lo), 3), "hi": round(float(x.hi), 3),
                             "kl": None if pd.isna(x.get("kl", np.nan)) else round(float(x.kl), 3),
                             "dots": [int(v >= 0.5) for v in acc.to_numpy()[:NDOTS]]})
            out[name] = {"rows": rows, "source": f"capability/math500/{tag}, analysis/kick_kl/{','.join(kl_tags)}"}
    # kick damage at every site (size 1, first 30 problems): E8's Dory x4 pilot and E16 for the other runs. Stages: Dory's
    # entry stack ("in"), its loops, its exit stack ("out"); Qwen's 36 layers as four quarters of 9 (loop = quarter)
    strips = {}
    for name, m, f in [("dory", "dory", "kickp_sites"), ("dory4", "dory", "kickp_pilot_loops4"), ("nanbeige", "nanbeige", "kickp_sites"),
                       ("ouro", "ouro", "kickp_sites"), ("qwen", "qwen", "kickp_sites")]:
        path = RUNS / MODELS[m] / "analysis" / "kick_kl" / f"{f}.parquet"
        if not path.exists():
            continue
        x = pd.read_parquet(path)
        p = x[x.eps == 1.0].groupby(["loop", "layer"]).kl.mean().reset_index()
        cells = []
        for _, r in p.iterrows():
            loop, layer = int(r.loop), int(r.layer)
            if m == "dory":
                stage = "in" if (loop == 0 and layer < 16) else ("out" if layer >= 56 else "loop")
                lp = loop + 1 if stage == "loop" else None
            elif m == "qwen":
                stage, lp = "loop", layer // 9 + 1
            else:
                stage, lp = "loop", loop + 1
            cells.append({"stage": stage, "loop": lp, "layer": layer, "kl": round(float(r.kl), 4)})
        strips[name] = {"cells": cells, "source": f"analysis/kick_kl/{f}.parquet (eps 1, {x.problem_idx.nunique()} problems)"}
    out["strips"] = strips
    out["dory4_strip"] = strips["dory4"]
    return out


# ---------------------------------------------------------------- readout, weight edits
def readouts():
    out = {}
    for name, (m, f) in {"ouro": ("ouro", "loop_readout_loops4.csv"), "nanbeige": ("nanbeige", "loop_readout_loops2.csv"),
                         "dory": ("dory", "loop_readout_loops2.csv"), "dory4": ("dory", "loop_readout_loops4.csv"),
                         "qwen": ("qwen", "loop_readout_blocks4.csv")}.items():
        d = pd.read_csv(RUNS / MODELS[m] / "analysis" / f)
        g = d.groupby("k")[["agree", "kl"]].mean()
        out[name] = {"agree": [round(float(v), 3) for v in g.agree], "kl": [round(float(v), 3) for v in g.kl],
                     "source": f"analysis/{f}"}
    return out


def capability(m, cand):
    res = {}
    for ds in ["math500", "aime25", "mmlupro"]:
        f = RUNS / MODELS[m] / "capability" / ds / "cap" / f"{cand}.parquet"
        if f.exists():
            res[ds] = round(100 * float(pd.read_parquet(f).groupby("problem_id").correct.mean().mean()), 1)
    return res


def weight_edits():
    spec = {"qwen": ("grid", "v4_baseline_l21", ["ortho"]),
            "nanbeige": ("grid", "v4_baseline_t1.l15", ["ortho"]),
            "dory": ("ortho_test", "hh_eoi_t0.l26", ["ortho", "ortho_na"]),
            "ouro": ("grid", "v4_baseline_t3.l16", ["ortho", "ortho_na"])}
    out = {}
    for m, (src, d, kinds) in spec.items():
        dirs = (gen_dirs(m, ["grid_test", "grid_test_b"]) if src == "grid" else gen_dirs(m, ["ortho_test"]))
        rand = d.split("_")[-1] if m != "qwen" else "l21"
        rand = re.sub(r"^.*?(t?\d*\.?l\d+)$", r"\1", d)
        row = {"clean": {"refusal": refusal(dirs, "none", dots=False), **capability(m, "none")},
               "hooks": {"refusal": refusal(dirs, f"ablate_{d}", dots=False), **capability(m, f"ablate_{d}")},
               "random": {"refusal": refusal(dirs, f"ortho_random_{rand}", dots=False), **capability(m, f"ortho_random_{rand}")}}
        for k in kinds:
            cand = f"ortho_{d}" + ("_na1" if k == "ortho_na" else "")
            row[k] = {"refusal": refusal(dirs, cand, dots=False), **capability(m, cand)}
        out[m] = row
    samples = {}
    for cand in ["none", "ortho_v4_baseline_t3.l16", "ortho_v4_baseline_t3.l16_na1", "ortho_random_t3.l16"]:
        df = pd.read_parquet(RUNS / MODELS["ouro"] / "capability" / "math500" / "cap" / f"{cand}.parquet")
        df = df.sort_values(["problem_id", "sample"])
        row = df.iloc[1 if cand.endswith("na1") else 0]  # na1: problem 0 happens to finish; show a typical failure
        raw = str(row.response).strip()
        text = re.sub(r"\s+", " ", raw)
        samples[cand] = {"problem": str(row.problem_id), "head": clip_md(raw, 220), "tail": text[-170:].lstrip(),
                         "n_tokens": int(row.n_tokens), "correct": bool(row.correct),
                         "closed_frac": round(float(df.closed.mean()), 3)}
    out["ouro_samples"] = samples
    out["leak"] = {"chain": [0, 0.4, 1.3, 3.5, 8.5], "with_leak1": 11.0, "clean_ratio": 4.5,
                   "source": "docs/presentation.md (ortho_compare.py, 8 traces): u-component entering each loop under the norm-aware edit"}
    out["growth"] = norm_growth()
    return out


def norm_growth():
    """RMS of the stream's component along u at the end of each loop, relative to the unedited model (the slide
    deck's norm_growth figure), and how far each norm's gain turns u (cos(u, g*u))."""
    out = {}
    for m, site in (("ouro", "t3l16"), ("nanbeige", "t1l15")):
        f = RUNS / MODELS[m] / "analysis" / f"ortho_compare_v4_baseline_{site}.json"
        raw = json.loads(f.read_text())["full"]
        T = raw["T"]
        clean = [raw["clean"]["refusal"][str(t)]["resid_rms"] for t in range(T)]
        row = {k: [round(raw[k]["refusal"][str(t)]["resid_rms"] / clean[t], 3) for t in range(T)]
               for k in ("ortho", "ortho_na", "hook") if k in raw}
        row["gains"] = {k: {"cos_mean": round(v["refusal"]["cos_u_gu_mean"], 3), "cos_min": round(v["refusal"]["cos_u_gu_min"], 3),
                            "n": v["n_norms"]} for k, v in raw.get("gains", {}).items()}
        row["source"] = f"analysis/{f.name}"
        out[m] = row
    return out


# ---------------------------------------------------------------- example replies
EXAMPLE_PROMPTS = [20, 25, 94]  # mild requests whose replies are safe to show (an insult, a spam bot, a biased rubric)
# the first 100 harmful test prompts are the dots; the page shows the replies (and reasoning) to these mild ones only:
# insults, opinion and persuasion, misleading marketing, small deceptions, everyday advice. Never weapons, drugs,
# hacking, self-harm, sexual content, violence, hate, fraud or election manipulation: for those a dot shows its prompt
# and verdict, and scripts/export_blog_samples.py never exports the reply
SAFE_PROMPTS = [0, 9, 15, 20, 23, 24, 25, 29, 35, 40, 41, 45, 67, 70, 76, 78, 81, 88, 89, 94, 95, 96]


def examples(grids):
    """For a few test prompts: the prompt, and one reply per switchboard setting (a sample whose verdict matches the
    prompt's majority over its 9 samples), with the share of samples refused."""
    tags = {"dory4": ["grid_test4", "grid_test4_b"]}
    test = pd.read_csv(ROOT / "data" / "prompts" / "test_harmful_prompts.csv").prompt.tolist()[:NDOTS]
    out = {"prompts": {}, "models": {}, "safe": SAFE_PROMPTS, "all": [clip(p, 240) for p in test]}
    for m, g in grids.items():
        model = "dory" if m == "dory4" else m
        dirs = gen_dirs(model, tags.get(m, ["grid_test", "grid_test_b"]))
        conds = {"none": g["clean"]["cand"]}
        for mode in ("erase", "push"):
            for k, rec in g[mode].items():
                conds[f"{mode}:{k}"] = rec["cand"]
        res = {str(p): {} for p in EXAMPLE_PROMPTS}
        for ck, cand in conds.items():
            parts = []
            for d in dirs:
                for sub in (".", "invalid"):
                    f = d / sub / f"{cand}.scored.parquet"
                    if f.exists():
                        x = read_scored(f)
                        x = x[x.prompt_idx.isin(EXAMPLE_PROMPTS)].copy()
                        if "out_rep" not in x:
                            x["out_rep"] = 0
                        parts.append(x[["prompt_idx", "prompt", "cot_rep", "out_rep", "output", "sr_refusal"]])
            df = pd.concat(parts)
            df["sr_refusal"] = df.sr_refusal.fillna(0)
            for p, sub in df.groupby("prompt_idx"):
                out["prompts"].setdefault(str(p), clip(sub.prompt.iloc[0], 300))
                share = float(sub.groupby("cot_rep").sr_refusal.mean().mean())
                want = share >= 0.5
                sub = sub.sort_values(["cot_rep", "out_rep"])
                ok = sub[((sub.sr_refusal >= 0.5) == want) & sub.output.fillna("").str.strip().astype(bool)]
                text = clip_md(ok.output.iloc[0], 360) if len(ok) else ""
                res[str(p)][ck] = [round(share, 2), text]
        out["models"][m] = res
    return out


# ---------------------------------------------------------------- second judge
LUNA_SCORED = {"dory": "train_clean", "nanbeige": "clean_train/none", "ouro": "train_clean", "qwen": "train_clean"}


def second_judge():
    """gpt-6-luna on the unsteered training replies behind each refusal direction (scripts/judge_api.py,
    scripts/judge_compare.py): verdict agreement with gpt-oss-20b, both refusal rates, and the direction rebuilt from
    luna's labels. Answers luna's API would not score (its biology filter) are left out and counted."""
    cmp = pd.read_csv(RUNS / "judge_compare.csv")
    rows = {}
    for m, stem in LUNA_SCORED.items():
        df = pd.read_parquet(RUNS / MODELS[m] / "generations" / f"{stem}.scored_luna.parquet",
                             columns=["gptoss_sr_refusal", "sr_refusal"])
        a, b = df.gptoss_sr_refusal, df.sr_refusal
        ok = a.notna() & b.notna()
        c = cmp[cmp.model == MODELS[m]].set_index("labelling")
        rows[m] = {"answers": int(ok.sum()), "blocked": int(b.isna().sum()),
                   "agree": round(100 * float((a[ok] == b[ok]).mean()), 1),
                   "ref_a": round(100 * float(a[ok].mean()), 1), "ref_b": round(100 * float(b[ok].mean()), 1),
                   "site": c.loc["luna_refusal", "site"],
                   "cos": round(float(c.loc["luna_refusal", "cos_site"]), 3),
                   "cos_min": round(float(c.loc["luna_refusal", "cos_min"]), 3),
                   "cos_score": round(float(c.loc["luna_score", "cos_site"]), 3),
                   "non_cur": int(c.loc["gptoss_score", "cot_non"]), "non_score": int(c.loc["luna_score", "cot_non"]),
                   "non_ref": int(c.loc["luna_refusal", "cot_non"]),
                   "auc_cur": round(float(c.loc["gptoss_score", "auc_heldout"]), 3),
                   "auc_ref": round(float(c.loc["luna_refusal", "auc_heldout"]), 3),
                   "auc_score": round(float(c.loc["luna_score", "auc_heldout"]), 3)}
    return {"judge": "gpt-6-luna", "rows": rows, "steered": steered_second_judge(), "extra": extra_second_judge()}


LUNA_GROUPS = {"grid": (["grid_test", "grid_test_b"], None, "harmful"), "grid4": (["grid_test4", "grid_test4_b"], 4, "harmful"),
               "grid6": (["grid_test6", "grid_test6_b"], 6, "harmful"), "plus": (["plus_harmless"], None, "harmless"),
               "plus4": (["plus_harmless4"], 4, "harmless"), "ortho": (["ortho_test"], None, "harmful"),
               "ortho4": (["ortho_test4"], 4, "harmful")}
TRAINED = {"ouro": 4, "nanbeige": 2, "dory": 2, "qwen": None}


def luna_pp(dirs, name):
    """Per-prompt refusal under both judges on the same answers (luna mirror ``<dir>/[invalid/]luna/``), or None while
    any of the run's files is not judged yet. Answers luna's API would not score (bio filter) keep gpt-oss-20b's
    verdict; other unparsed answers count as complied, as in per_prompt."""
    parts = []
    for d in dirs:
        for sub in (".", "invalid"):
            f = d / sub / f"{name}.scored.parquet"
            if not f.exists():
                continue
            g = d / sub / "luna" / f.name
            if not g.exists():
                return None
            parts.append(pd.read_parquet(g, columns=["prompt_idx", "cot_rep", "sr_refusal", "gptoss_sr_refusal",
                                                     "judge_error"]) if pq_rows(g) else None)
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    df = pd.concat(parts)
    blocked = df.sr_refusal.isna() & df.judge_error.astype(str).str.contains("bio_policy")
    df["oss"] = df.gptoss_sr_refusal.fillna(0)
    df["luna"] = df.sr_refusal.where(~blocked, df.oss).fillna(0)
    ps = df.groupby(["prompt_idx", "cot_rep"])[["oss", "luna"]].mean()
    return {"ps": ps, "pp": ps.groupby("prompt_idx").mean(), "answers": len(df), "blocked": int(blocked.sum()),
            "agree": int(((df.oss >= 0.5) == (df.luna >= 0.5)).sum())}


def pq_rows(f):
    import pyarrow.parquet as pq
    return pq.ParquetFile(f).metadata.num_rows


def cell_label(m, cand, group):
    """'erase · loops 1, 2 · run with 4 loops' style names for the comparison's tooltips."""
    _, loops, kind = LUNA_GROUPS[group]
    if cand == "none":
        what = "no steering"
    elif cand.startswith("ortho"):
        what = "weight edit" + (" of a random direction" if "_random_" in cand else "") + (" (norm-aware)" if "_na1" in cand else "")
    else:
        what = ("erase" if cand.startswith("ablate") else "add") + (" a random direction" if "_random_" in cand else "")
        c = re.search(r"_c(-?[\d.]+)", cand)
        if c and "_random_" not in cand:
            what += " against refusal" if float(c.group(1)) < 0 else " toward refusal"
        if c and "rel1" in cand:
            what += f" ×{abs(float(c.group(1))):g}"
        elif c:
            what += f" (c = {c.group(1)})"
        a, lay, at = (re.search(p, cand) for p in (r"_apply([\d-]+)", r"_layers([\d-]+)", r"_at([\d-]+)"))
        if a:
            what += " · loop" + ("s " if "-" in a.group(1) else " ") + ", ".join(str(int(x) + 1) for x in a.group(1).split("-"))
        elif lay:
            xs = list(map(int, lay.group(1).split("-")))
            what += " · " + ", ".join(f"Q{x // 9 + 1}" if x // 9 == y // 9 else f"Q{x // 9 + 1}–{y // 9 + 1}"
                                      for x, y in zip(xs[::2], xs[1::2]))
        elif at:
            what += " · " + ", ".join(f"Q{int(x) // 9 + 1}" for x in at.group(1).split("-"))
        else:
            what += " · every layer" if m == "qwen" else " · every loop"
    if loops and loops != TRAINED[m]:
        what += f" · run with {loops} loops"
    return what + (" · harmless prompts" if kind == "harmless" else "")


PERSONA_TRAINED = {"dory": 2, "nanbeige": 2, "ouro": 4, "qwen": None}


def persona_second_judge(traits):
    """Persona vectors rebuilt from the answers gpt-6-luna keeps (directions/persona_luna.pt, persona_vectors.py
    --judge-tag) vs the current ones (persona.pt, gpt-oss-20b): cosine at the site each trait steers with (the last
    trained loop's tuned layer from analysis/persona_tune.json; dense: that layer), for the traits on the page, with
    the trait-on / trait-off answer counts each judge kept."""
    import torch
    out = {}
    for m, ts in traits.items():
        if m not in PERSONA_TRAINED:
            continue
        d = RUNS / MODELS[m] / "directions"
        if not (d / "persona_luna.pt").exists():
            continue
        a = torch.load(d / "persona.pt", weights_only=False, map_location="cpu")
        b = torch.load(d / "persona_luna.pt", weights_only=False, map_location="cpu")
        tune = json.loads((RUNS / MODELS[m] / "analysis" / "persona_tune.json").read_text())
        sites, T = [tuple(x) for x in a["sites"]], PERSONA_TRAINED[m]
        res = {}
        for t in ts:
            if t not in tune or t not in b:
                continue
            layer = tune[t]["layer"]
            i = sites.index((T - 1, layer) if T else (0, layer))
            cos = float(torch.nn.functional.cosine_similarity(a[t]["dirs"][i].float(), b[t]["dirs"][i].float(), dim=0))
            ca, cb = a.get("counts", {}).get(t, {}), b.get("counts", {}).get(t, {})
            res[t] = {"layer": int(layer), "loop": (T or 1), "cos": round(cos, 3),
                      "kept": [ca.get("pos"), ca.get("neg"), cb.get("pos"), cb.get("neg")]}
        out[m] = res
    return out


def extra_second_judge():
    """The other scores under both judges: persona trait and coherence (persona_judge_api.py), flip-sign coherence and
    kick cut-off labels (judge_api_extra.py). Correlations on replies both judges scored; label agreement."""
    out = {}
    P = []
    for m in ["dory", "nanbeige", "ouro", "qwen"]:
        for name in ["steer_loops", "steer_loops_loops4", "steer_ablate", "steer_ablate_loops4"]:
            f = RUNS / MODELS[m] / "persona" / f"{name}.judged_luna.parquet"
            if f.exists():
                P.append(pd.read_parquet(f, columns=["gptoss_trait_score", "trait_score", "gptoss_coherence", "coherence"]))
    if P:
        x = pd.concat(P)
        t, c = x.dropna(subset=["gptoss_trait_score", "trait_score"]), x.dropna(subset=["gptoss_coherence", "coherence"])
        out["persona"] = {"n": len(x), "trait_r": round(float(t.gptoss_trait_score.corr(t.trait_score)), 3),
                          "coh_r": round(float(c.gptoss_coherence.corr(c.coherence)), 3),
                          "trait_diff": round(float((t.trait_score - t.gptoss_trait_score).mean()), 1),
                          "coh_diff": round(float((c.coherence - c.gptoss_coherence).mean()), 1)}
    C = []
    for m in MODELS:
        for tag in ["plus_harmless", "plus_harmless4"]:
            for sub in (".", "invalid"):
                d = RUNS / MODELS[m] / "generations" / tag / sub
                for f in sorted((d / "luna").glob("*.coh.parquet")) if (d / "luna").exists() else []:
                    a, b = pd.read_parquet(d / f.name), pd.read_parquet(f)
                    k = [c for c in ("prompt_idx", "cot_rep", "out_rep") if c in a]
                    C.append(a.merge(b, on=k, suffixes=("_oss", "_luna")))
    if C:
        x = pd.concat(C).dropna(subset=["coherence_oss", "coherence_luna"])
        out["flipcoh"] = {"n": len(x), "r": round(float(x.coherence_oss.corr(x.coherence_luna)), 3),
                          "diff": round(float((x.coherence_luna - x.coherence_oss).mean()), 1)}
    K = []
    for m in MODELS:
        for f in (RUNS / MODELS[m] / "capability" / "math500").glob("kickp*/cutoff/luna/*.parquet"):
            a, b = pd.read_parquet(f.parent.parent / f.name), pd.read_parquet(f)
            K.append(a[["problem_id", "sample", "cut"]].merge(b[["problem_id", "sample", "cut"]], on=["problem_id", "sample"],
                                                              suffixes=("_oss", "_luna")))
    if K:
        x = pd.concat(K)
        sh = lambda col: {k: round(100 * float(v), 1) for k, v in x[col].value_counts(normalize=True).items()}
        out["cutoff"] = {"n": len(x), "agree": round(100 * float((x.cut_oss == x.cut_luna).mean()), 1),
                         "oss": sh("cut_oss"), "luna": sh("cut_luna")}
    return out


def steered_second_judge():
    """gpt-6-luna vs gpt-oss-20b on every refusal run behind the page (cells as the page pools them)."""
    cells, tot = [], {"answers": 0, "blocked": 0, "agree": 0}
    by_model = {}
    for m in ["dory", "nanbeige", "ouro", "qwen"]:
        for group, (tags, _, _) in LUNA_GROUPS.items():
            dirs = [d for d in gen_dirs(m, tags) if d.exists()]
            if not dirs:
                continue
            cands = sorted({f.name.removesuffix(".scored.parquet") for d in dirs for sub in (".", "invalid")
                            for f in (d / sub).glob("*.scored.parquet")})
            for cand in cands:
                r = luna_pp(dirs, cand)
                if r is None:
                    continue
                for k in tot:
                    tot[k] += r[k]
                bm = by_model.setdefault(m, {"answers": 0, "agree": 0, "blocked": 0})
                for k in bm:
                    bm[k] += r[k]
                cells.append([m, group, cand, cell_label(m, cand, group), int(len(r["pp"])),
                              round(100 * float(r["ps"].oss.mean()), 1), round(100 * float(r["ps"].luna.mean()), 1)])
    out = {"cells": cells, "cols": ["model", "group", "cand", "label", "n", "oss", "luna"], **tot,
           "by_model": by_model}
    # the page's headline numbers under both judges
    def pair(m, group, trained_cand, extra_group, extra_cand, clean_cand="none"):
        a = luna_pp([d for d in gen_dirs(m, LUNA_GROUPS[group][0]) if d.exists()], trained_cand)
        b = luna_pp([d for d in gen_dirs(m, LUNA_GROUPS[extra_group][0]) if d.exists()], extra_cand)
        if a is None or b is None:
            return None
        keep = a["pp"].index.intersection(b["pp"].index)
        res = {}
        for j in ("oss", "luna"):
            diff = 100 * (b["pp"][j][keep] - a["pp"][j][keep]).to_numpy()
            boots = diff[RNG.integers(0, len(diff), (1000, len(diff)))].mean(1)
            res[j] = {"d": round(float(diff.mean()), 1), "lo": round(float(np.percentile(boots, 2.5)), 1),
                      "hi": round(float(np.percentile(boots, 97.5)), 1)}
        return {"n": int(len(keep)), **res}
    out["gains"] = {"dory": pair("dory", "grid", "ablate_hh_eoi_t0.l26_apply0-1", "grid4", "ablate_hh_eoi_t0.l26_apply0-1"),
                    "ouro": pair("ouro", "grid", "ablate_v4_baseline_t3.l16", "grid6", "ablate_v4_baseline_t3.l16_apply0-1-2-3")}
    return out


def main():
    global JUDGE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", choices=sorted(JUDGES), default=JUDGE,
                    help="judge of record: v2 gpt-6-luna (default), v1 gpt-oss-20b")
    JUDGE = ap.parse_args().judge
    data = {
        "generated": pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M ET"),
        "judge": {"id": JUDGE, "name": JUDGES[JUDGE]},
        "grids": {"ouro": grid_ouro(), "nanbeige": grid_nanbeige(), "dory": grid_dory2(), "dory4": grid_dory4(),
                  "qwen": grid_qwen()},
        "plus_harmless": plus_harmless(),
        "extra": extra_loops(),
        "persona": personas(),
        "persona_ablate": persona_ablation(),
        "reinject": reinjection(),
        "kicks": kicks(),
        "readout": readouts(),
        "edits": weight_edits(),
        "judge2": second_judge(),
    }
    data["examples"] = examples(data["grids"])
    data["judge2"]["persona"] = persona_second_judge({m: list(v["traits"]) for m, v in data["persona"].items()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("// Generated by scripts/export_blog_data.py; do not edit by hand.\nwindow.BLOG_DATA = "
                   + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
