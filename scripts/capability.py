#!/usr/bin/env python3
"""Capability check under interventions: accuracy on MATH-500 or AIME 2025 for the same candidates as intervene.py.

Candidate syntax is that of ``scripts/intervene.py`` (``none``, ``ablate:v4_baseline:t3.l16:apply=2``, ...). Each
problem is asked once per sample with a single-turn prompt that requests a final answer in ``\\boxed{}``; the model
thinks and answers in one generation (CoT cap ``--max-tokens``). A sample is *closed* if its CoT ends with
``</think>``; the answer is the last ``\\boxed{}`` after it (an unclosed sample has no answer and counts as wrong).
Grading is string match after light LaTeX normalisation, or numeric equality (integer match for AIME), so treat it as a
slightly pessimistic estimate and compare candidates with each other, not with published numbers.

    scripts/capability.py --model ByteDance/Ouro-1.4B-Thinking --dataset math500 --tag cap \\
        --candidates none ablate:v4_baseline:t3.l16 ablate:v4_baseline:t3.l16:apply=2

Writes data/runs/<model>/capability/<dataset>/<tag>/<candidate>.parquet (skip-existing with --skip-existing) and
prints accuracy and the share of closed samples. Kick candidates with method ``randp<seed>`` give problem i (its index
among the first ``--limit``) its own random direction (``loop_steer.kicks.problem_directions``; prefix caching off). Datasets come from the Hugging Face Hub (public):
``HuggingFaceH4/MATH-500`` (500 problems), ``opencompass/AIME2025`` (30 problems) and ``TIGER-Lab/MMLU-Pro``
(``mmlupro``: a fixed subset of 20 test questions per category, 280 in all, drawn with seed 0; options A-J listed in the
prompt, graded on the option letter in ``\\boxed{}``, or failing that the last "answer is (X)"; files written before
this fallback are regraded with ``scripts/regrade_capability.py``).
"""

import argparse
import functools
import json
import os
import re

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
MMLUPRO_PER_CATEGORY = 20


def load_problems(dataset: str) -> list[dict]:
    """[{'id', 'question', 'answer'}] from the Hub (cached copy is used when offline)."""
    from huggingface_hub import hf_hub_download

    if dataset == "math500":
        path = hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset")
        rows = [json.loads(line) for line in open(path)]
        return [{"id": r["unique_id"], "question": r["problem"], "answer": str(r["answer"])} for r in rows]
    if dataset == "aime25":
        rows = []
        for name in ("aime2025-I.jsonl", "aime2025-II.jsonl"):
            rows += [json.loads(line) for line in open(hf_hub_download("opencompass/AIME2025", name, repo_type="dataset"))]
        return [{"id": f"aime25-{i}", "question": r["question"], "answer": str(r["answer"])} for i, r in enumerate(rows)]
    if dataset == "mmlupro":  # fixed subset: 20 questions per category (14) drawn with seed 0, multiple choice A-J
        import pandas as pd

        d = pd.read_parquet(hf_hub_download("TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet",
                                            repo_type="dataset"))
        d = (d.sort_values("question_id").groupby("category").sample(n=MMLUPRO_PER_CATEGORY, random_state=0)
             .sort_values(["category", "question_id"]))
        out = []
        for r in d.itertuples():
            opts = "\n".join(f"{chr(65 + k)}. {o}" for k, o in enumerate(r.options))
            out.append({"id": f"mmlupro-{r.question_id}", "category": r.category, "answer": r.answer,
                        "question": f"{r.question}\n\nOptions:\n{opts}\n\nAnswer with the letter of the correct option."})
        return out
    raise ValueError(f"unknown dataset {dataset!r}")


def last_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` (brace-matched), or None."""
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        depth += (text[j] == "{") - (text[j] == "}")
        if depth == 0:
            return text[i + 1:j]
    return None


ANSWER_IS = re.compile(r"answer is:?\s*\**\s*\(?([A-J])\)?(?![A-Za-z])")


def extract_answer(answer: str, truth: str) -> str | None:
    """The last ``\\boxed{}`` of the answer; for multiple choice (MMLU-Pro), else the last "answer is (X)" letter."""
    pred = last_boxed(answer)
    if pred is None and re.fullmatch(r"[A-J]", truth):  # Ouro answers in MMLU-Pro's own format, not \boxed{}
        found = ANSWER_IS.findall(answer)
        pred = found[-1] if found else None
    return pred


# a unit after the answer: "5\text{ cm}", "12\pi \text{ inches per second}", "65 \mbox{ cubic units}^2"
UNIT_TAIL = re.compile(r"^(.*\S)\s*(?:\\(?:text|textnormal|mbox|mathrm)\{\s*[^{}]*\}\s*(?:\^\{?\d\}?)?)+$")


def normalize(ans: str) -> str:
    """Light LaTeX normalisation (grader v2, 2026-11-21). v1 kept unit text ("5\\text{ cm}"), stripped "$" before "\\$",
    turned a matrix row break "\\\\ " into "\\" when removing "\\ ", and compared \\sqrt2 / \\sqrt{2}, x^5 / x^{5},
    \\frac13 / 1/3 inside larger answers, "(C)" / "C" and "12^{th}" / "12" as different."""
    s = ans.strip().replace("\\\\", "\x00")  # matrix row breaks survive the spacing commands below
    m = UNIT_TAIL.match(s)
    if m and re.search(r"[\d}]", m.group(1)):  # a trailing unit after a number or expression, never a text-only answer
        s = m.group(1)
    s = re.sub(r"\\(?:text|textnormal|textbf|mbox|mathrm)\{([^}]*)\}", r"\1", s)
    for a, b in (("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\displaystyle", ""), ("\\left", ""), ("\\right", ""),
                 ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""),
                 ("%", ""), ("\\$", ""), ("$", "")):
        s = s.replace(a, b)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\x00\[[^\]]*\]", "\x00", s)  # matrix row spacing: \\[2pt] -> \\
    s = s.rstrip(".")
    s = re.sub(r"^(?:[a-zA-Z]\w*|\\mathbf\{?\w+\}?)=", "", s)  # "x=3" -> "3", "\mathbf{v}=..." -> "..."
    s = re.sub(r"^[a-zA-Z]\\in", "", s)  # "x \in [-2,7]" -> "[-2,7]"
    s = re.sub(r"\^\{?(?:st|nd|rd|th)\}?$", "", s)  # "12^{th}" -> "12"
    s = re.sub(r"^\(([A-Z])\)$", r"\1", s)  # "(C)" -> "C"
    s = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)  # \sqrt2 -> \sqrt{2}
    s = re.sub(r"([\^_])\{(\w)\}", r"\1\2", s)  # x^{5} -> x^5
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)  # \frac13 -> \frac{1}{3}
    s = re.sub(r"\\frac(\d)\{", r"\\frac{\1}{", s)  # \frac9{19} -> \frac{9}{19}
    s = re.sub(r"\\frac\{(\w+)\}(\d)", r"\\frac{\1}{\2}", s)  # \frac{3}5 -> \frac{3}{5}
    s = re.sub(r"\\frac\{(-?\w+)\}\{(\w+)\}", r"\1/\2", s)  # \frac{1}{3} -> 1/3 (simple parts, anywhere)
    return s.replace("\x00", "\\\\")


def _top_split(s: str) -> list[str]:
    """Split at commas outside brackets."""
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += (ch in "([{") - (ch in ")]}")
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + [cur]


def _same(a: str, b: str) -> bool:
    if a == b:
        return True
    x, y = _number(a), _number(b)
    return x is not None and y is not None and abs(x - y) < 1e-9


def _number(s: str) -> float | None:
    t = s.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s) else s
    try:
        if "/" in t:
            a, b = t.split("/")
            return float(a) / float(b)
        return float(t)
    except (ValueError, ZeroDivisionError):
        return None


def is_correct(pred: str | None, truth: str) -> bool:
    if pred is None:
        return False
    if re.fullmatch(r"[A-J]", truth):  # multiple choice (MMLU-Pro): the option letter, alone or leading the answer
        m = re.fullmatch(r"\(?([A-J])\)?(?:[.:)\s].*)?", re.sub(r"\\text(?:bf)?\{([^}]*)\}", r"\1", pred).strip(), re.S)
        return m is not None and m.group(1) == truth
    a, b = normalize(pred), normalize(truth)
    if _same(a, b):
        return True
    if re.fullmatch(r"[A-Za-z]+", a) and a.lower() == b.lower():  # a word answer: "East" for \text{east}
        return True
    base = re.fullmatch(r"([0-9A-Z]+)_\{?\d+\}?", b)  # "2516" for 2516_8: the digits, without the base subscript
    if base and "_" not in a and a == base.group(1):
        return True
    # "1 \pm \sqrt{19}" stands for both roots
    tb = [b.replace("\\pm", s, 1) for s in "+-"] if b.count("\\pm") == 1 and "," not in b else _top_split(b)
    ta = [re.sub(r"^[a-zA-Z]\w*=", "", x) for x in _top_split(a)]  # "x=3,x=5,x=7"
    if len(tb) > 1 and len(ta) == len(tb):  # a list of solutions, in any order
        rest = list(tb)
        for x in ta:
            hit = next((y for y in rest if _same(x, y)), None)
            if hit is None:
                return False
            rest.remove(hit)
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", choices=["math500", "aime25", "mmlupro"], required=True)
    ap.add_argument("--directions", default="train")
    ap.add_argument("--candidates", required=True, nargs="+")
    ap.add_argument("--tag", default="cap")
    ap.add_argument("--samples", type=int, default=1, help="Samples per problem")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N problems")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--n-loops", type=int, default=None,
                    help="Run a looped model with this many loops instead of its trained number; use a separate --tag")
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from loop_steer.paths import run_dir, setup_job_env
    setup_job_env()
    import pandas as pd
    import torch
    from intervene import intervention_specs, load_norms, parse_candidate
    from vllm import LLM, SamplingParams

    from loop_steer import vllm_hooks
    from loop_steer.cot import format_prompt, split_cot, strip_end
    from loop_steer.models import is_inkling, load_tokenizer, loop_overrides, prepare_vllm, vllm_kwargs, vllm_model
    from loop_steer.ortho import orthogonalize_

    cands = [parse_candidate(c) for c in args.candidates]
    if any(c["positions"] or c["decay"] for c in cands):
        ap.error("pos= and decay= are not supported here (capability prompts have no CoT/answer split)")
    uses_hooks = any(c["kind"] in ("ablate", "actadd", "kick") for c in cands)
    rd = run_dir(args.model)
    if any(c["kind"] != "none" for c in cands):
        dirs = torch.load(rd / "directions" / f"{args.directions}.pt", weights_only=False)
        site_index = {site: i for i, site in enumerate(dirs["sites"])}
    norms = load_norms(rd, args.n_loops) if any(c.get("rel") for c in cands) else None
    per_request = any(c["kind"] == "kick" and c["method"].startswith("randp") for c in cands)
    out_dir = rd / "capability" / args.dataset / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = load_problems(args.dataset)[: args.limit]
    tok = load_tokenizer(args.model)
    prompts = [format_prompt(tok, p["question"] + SUFFIX) for p in problems]

    as_ids = prepare_vllm(args.model)
    llm = LLM(model=vllm_model(args.model), seed=args.seed, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, enforce_eager=uses_hooks and not is_inkling(args.model),
              # per-problem kick directions: a cached prefix computed under another problem's kick must not be reused
              enable_prefix_caching=not per_request,
              hf_overrides=loop_overrides(args.model, args.n_loops), **vllm_kwargs(args.model))
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, n=args.samples,
                              skip_special_tokens=False)
    if per_request:  # request i carries the index of its problem's kick direction (vllm_hooks: ls_vec)
        sampling = [sampling.clone() for _ in problems]
        for i, sp in enumerate(sampling):
            sp.extra_args = {"ls_vec": i}
    inputs = [{"prompt_token_ids": tok(t, add_special_tokens=False).input_ids} for t in prompts] if as_ids else prompts
    weights_dirty = False
    for c in cands:
        out = out_dir / f"{c['name']}.parquet"
        if args.skip_existing and out.exists():
            continue
        vec = None
        if c["kind"] == "ortho":
            vec = dirs[c["method"]]["dirs"][site_index[c["site"]]]
        if weights_dirty or c["kind"] == "ortho":
            llm.apply_model(functools.partial(orthogonalize_, direction=vec, norm_aware=c["norm_aware"],
                                              biases=c["biases"]))
            weights_dirty = c["kind"] == "ortho"
        if uses_hooks:
            specs = (intervention_specs(c, dirs, site_index, norms=norms, n_loops=args.n_loops, n_requests=len(problems))
                     if c["kind"] in ("ablate", "actadd", "kick") else [])
            llm.apply_model(functools.partial(vllm_hooks.install, specs=specs))
        if not per_request:
            llm.reset_prefix_cache()
        rows = []
        for prob, req in zip(problems, llm.generate(inputs, sampling)):
            for k, o in enumerate(req.outputs):
                # split before cutting at the end token: gpt-oss closes its analysis channel with <|end|> too
                cot, answer, closed = split_cot(o.text)
                answer = strip_end(answer)
                response = cot + answer if closed else strip_end(o.text)
                pred = extract_answer(answer, prob["answer"]) if closed else None
                rows.append({"problem_id": prob["id"], "sample": k, "truth": prob["answer"], "pred": pred,
                             "correct": is_correct(pred, prob["answer"]), "closed": closed,
                             "n_tokens": len(o.token_ids), "hit_limit": o.finish_reason == "length",
                             "response": response, "candidate": c["name"],
                             **({"category": prob["category"]} if "category" in prob else {})})
        df = pd.DataFrame(rows)
        df.to_parquet(out)
        print(f"[{c['name']}] {args.dataset}: accuracy {df.correct.mean():.3f}, closed {df.closed.mean():.3f}, "
              f"mean tokens {df.n_tokens.mean():.0f}, n={len(df)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
