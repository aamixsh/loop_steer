#!/usr/bin/env python3
"""Prompt-only contrast set for a harmful - harmless direction (Arditi et al.): train harmful prompts (minus the
selection prompts, offset 500-549) vs train harmless prompts, no CoTs. Writes a labels-style parquet for
``extract_activations.py`` (column ``label_hh`` = harmful | harmless; only the prompt windows are meaningful).
"""

import argparse

import pandas as pd

from loop_steer.cot import load_prompts
from loop_steer.paths import run_dir

SELECTION = range(500, 550)  # intervene.py --offset 500 --n-prompts 50 (run_select.sh)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    harmful = [(i, p) for i, p in enumerate(load_prompts("train", "harmful")) if i not in SELECTION]
    harmless = list(enumerate(load_prompts("train", "harmless")))
    df = pd.DataFrame(
        [{"prompt_idx": i, "prompt": p, "label_hh": "harmful"} for i, p in harmful]
        + [{"prompt_idx": 100_000 + i, "prompt": p, "label_hh": "harmless"} for i, p in harmless])
    df["cot"] = ""
    out = run_dir(args.model) / "generations" / "contrast_labels.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"{len(harmful)} harmful, {len(harmless)} harmless -> {out}")


if __name__ == "__main__":
    main()
