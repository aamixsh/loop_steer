#!/usr/bin/env python3
"""Export the numbers behind the interactive blog post (docs/blog/index.html) to docs/blog/data.js.

    .venv/bin/python scripts/export_blog_data.py

CPU only, reads data/runs/<model>/. Every number the page shows comes from this file, and each block records the
files it was computed from. Refusal: the `all_refusal` quantity of summarize.py / grid_ci.py, pooled over the test
tags given, with a prompt bootstrap; per-prompt rates for the first 100 test prompts drive the dot grids. Kicks:
scripts/kick_summary.py. Persona: analysis/persona_loops*.csv and persona_ablate*.csv, example replies from
persona/steer_loops*.judged.parquet.
"""

import json
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


def gen_dirs(model, tags):
    return [RUNS / MODELS[model] / "generations" / t for t in tags]


def per_prompt(dirs, name):
    """Per-prompt refusal (mean over CoT samples of the mean judge refusal over answers; unclosed CoTs judged whole)."""
    parts = [pd.read_parquet(d / sub / f"{name}.scored.parquet", columns=["prompt_idx", "cot_rep", "sr_refusal"])
             for d in dirs for sub in (".", "invalid") if (d / sub / f"{name}.scored.parquet").exists()]
    if not parts:
        raise FileNotFoundError(f"{name} in {dirs}")
    df = pd.concat(parts)
    df["sr_refusal"] = df.sr_refusal.fillna(0)
    per_sample = df.groupby(["prompt_idx", "cot_rep"]).sr_refusal.mean()
    return per_sample, per_sample.groupby("prompt_idx").mean()


def refusal(dirs, name, dots=True):
    per_sample, pp = per_prompt(dirs, name)
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
    return out


# ---------------------------------------------------------------- extra loops
def extra_loops():
    d2 = gen_dirs("dory", ["grid_test", "grid_test_b"])
    d4 = gen_dirs("dory", ["grid_test4", "grid_test4_b"])
    o4 = gen_dirs("ouro", ["grid_test", "grid_test_b"])
    o6 = gen_dirs("ouro", ["grid_test6"])
    return {
        "dory": {"trained": 2, "extra": 4,
                 "erase": {"at_trained": refusal(d2, "ablate_hh_eoi_t0.l26_apply0-1"),
                           "at_extra": refusal(d4, "ablate_hh_eoi_t0.l26_apply0-1"),
                           "clean": refusal(d4, "none", dots=False)},
                 "push": {"at_trained": refusal(d2, "actadd_hh_eoi_t0.l26_c-1"),
                          "at_extra": refusal(d4, "actadd_hh_eoi_t0.l26_c-1_apply0-1"),
                          "clean": refusal(d4, "none", dots=False)}},
        "ouro": {"trained": 4, "extra": 6,
                 "erase": {"at_trained": refusal(o4, "ablate_v4_baseline_t3.l16"),
                           "at_extra": refusal(o6, "ablate_v4_baseline_t3.l16_apply0-1-2-3"),
                           "clean": refusal(o6, "none", dots=False)},
                 "push": {"at_trained": refusal(o4, "actadd_v4_baseline_t3.l16_c-1.0_rel1"),
                          "at_extra": refusal(o6, "actadd_v4_baseline_t3.l16_c-1.0_rel1_apply0-1-2-3"),
                          "clean": refusal(o6, "none", dots=False)}},
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


def personas():
    out = {}
    for name, (csv, pq) in PERSONA_FILES.items():
        m = "dory" if name == "dory4" else name
        d = pd.read_csv(RUNS / MODELS[m] / "analysis" / csv)
        r = pd.read_parquet(RUNS / MODELS[m] / "persona" / pq)
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
        d = pd.read_csv(RUNS / MODELS[m] / "analysis" / csv)
        d["units"] = d["units"].astype(str)
        g = d.groupby(["cond", "units"], sort=False)[["trait_score", "coherence"]].mean().round(1).reset_index()
        traits = {}
        for t, x in d.groupby("trait", sort=False):
            traits[t] = {"prompted": round(float(x.trait_clean.iloc[0]), 1), "noprompt": round(float(x.trait_noprompt.iloc[0]), 1),
                         "rows": [{"cond": r.cond, "units": r.units, "score": round(float(r.trait_score), 1),
                                   "coh": round(float(r.coherence), 1)} for r in x.itertuples()]}
        # share of the prompted trait removed, (prompted - erased) / (prompted - no prompt), over traits the prompt
        # raises by at least 15 points (smaller gaps make the ratio noise)
        keep = d[d.trait_clean - d.trait_noprompt >= 15]
        out[name] = {"vector": persona_coverage(m)["vector"], "rows": g.to_dict("records"), "prompted": round(float(d.trait_clean.mean()), 1),
                     "unprompted": round(float(d.trait_noprompt.mean()), 1) if "trait_noprompt" in d else None,
                     "traits": traits, "removed": (100 * keep.groupby("cond").removed.mean()).round(1).to_dict(),
                     "removed_traits": sorted(keep.trait.unique()), "source": f"analysis/{csv}"}
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


def kicks():
    out = {}
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
                rows.append({"cand": c, "site": site, "every": bool(apply) and "indep" not in apply and "," in apply,
                             "indep": "indep" in apply or "indep" in c, "acc": round(float(x.acc), 3),
                             "lo": round(float(x.lo), 3), "hi": round(float(x.hi), 3),
                             "kl": None if pd.isna(x.get("kl", np.nan)) else round(float(x.kl), 3),
                             "dots": [int(v >= 0.5) for v in acc.to_numpy()[:NDOTS]]})
            out[name] = {"rows": rows, "source": f"capability/math500/{tag}, analysis/kick_kl/{','.join(kl_tags)}"}
    pilot = pd.read_parquet(RUNS / MODELS["dory"] / "analysis" / "kick_kl" / "kickp_pilot_loops4.parquet")
    p = pilot[pilot.eps == 1.0].groupby(["loop", "layer"]).kl.mean().reset_index()
    strip = []
    for _, x in p.iterrows():
        loop, layer = int(x.loop), int(x.layer)
        stage = "in" if (loop == 0 and layer < 16) else ("out" if layer >= 56 else "loop")
        strip.append({"stage": stage, "loop": loop + 1 if stage == "loop" else None, "layer": layer,
                      "kl": round(float(x.kl), 4)})
    out["dory4_strip"] = {"cells": strip, "source": "analysis/kick_kl/kickp_pilot_loops4.parquet (eps 1, 30 problems)"}
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


def examples(grids):
    """For a few test prompts: the prompt, and one reply per switchboard setting (a sample whose verdict matches the
    prompt's majority over its 9 samples), with the share of samples refused."""
    tags = {"dory4": ["grid_test4", "grid_test4_b"]}
    out = {"prompts": {}, "models": {}}
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
                        x = pd.read_parquet(f)
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


def main():
    data = {
        "generated": pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M ET"),
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
    }
    data["examples"] = examples(data["grids"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("// Generated by scripts/export_blog_data.py; do not edit by hand.\nwindow.BLOG_DATA = "
                   + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
