#!/usr/bin/env python3
"""How far does a random kick move the model's next-token distribution? Teacher-forced KL from the unkicked model.

    scripts/kick_kl.py --gpu 0 --model nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 --n-loops 4 --tag kickp_loops4 \\
        --sites t0.l36,t1.l36,t2.l36,t3.l36 --eps 0.5,0.7,1,1.5
    scripts/kick_kl.py --gpu 0 --model Qwen/Qwen3-8B --tag kickp --sites l4,l13,l22,l31
    scripts/kick_kl.py ... --sites all --limit 30        # every site (pilot)

The teacher-forced counterpart of ``capability.py`` runs with ``kick:randp<seed>:<site>:eps=<e>``: the unkicked
model's own MATH-500 responses (``capability/<dataset>/<resp-tag>/none.parquet``; prompt + the first
``--max-resp-tokens`` response tokens) are run through the model once clean and once per (site, eps) with problem i's
random direction (``loop_steer.kicks.problem_directions(seed, --limit, D)[i]``, the same one the generation run gives
problem i) added as ``eps * |h| * v`` to the stream entering the site's layer, on every token except position 0 and
attention sinks (``|h| > --sink-ratio`` x the sequence median), as the vLLM kick does.

Per problem, site and eps: ``kl`` = mean over response tokens of KL(clean || kicked) next-token distributions,
``agree`` = share of response tokens whose top-1 token is unchanged, ``kl_last`` = KL at the last response token.
Writes ``analysis/kick_kl/<tag>.parquet``. Sites: ``t<loop>.l<layer>`` / ``l<layer>`` (comma-separated), ``all``
(every site of the run) or ``mid`` (each loop's, or each of 4 dense blocks', middle layer). HF backend, one GPU.
"""

import argparse
import os
import re


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-loops", type=int, default=None)
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--tag", required=True, help="output name; also the capability tag of the responses by default")
    ap.add_argument("--resp-tag", default=None, help="capability tag whose none.parquet holds the clean responses")
    ap.add_argument("--sites", default="mid")
    ap.add_argument("--eps", default="0.5,0.7,1,1.5")
    ap.add_argument("--seed", type=int, default=0, help="randp seed (as kick:randp<seed>)")
    ap.add_argument("--limit", type=int, default=100, help="the --limit of the generation run (fixes the directions)")
    ap.add_argument("--max-resp-tokens", type=int, default=512)
    ap.add_argument("--sink-ratio", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from loop_steer.paths import run_dir, setup_job_env
    setup_job_env()
    import pandas as pd
    import torch

    from capability import SUFFIX, load_problems
    from loop_steer.cot import format_prompt
    from loop_steer.kicks import problem_directions
    from loop_steer.models import load_model, load_tokenizer
    from loop_steer.selfcorr import pad_batch, stream_hooks, structure

    rd = run_dir(args.model)
    out = rd / "analysis" / "kick_kl" / f"{args.tag}.parquet"
    if args.skip_existing and out.exists():
        print(f"{out} exists; skipping")
        return
    resp = pd.read_parquet(rd / "capability" / args.dataset / (args.resp_tag or args.tag) / "none.parquet")
    resp = resp[resp["sample"] == 0].set_index("problem_id")
    problems = load_problems(args.dataset)[: args.limit]
    tok = load_tokenizer(args.model)
    items = []  # (problem index, ids, first response position, n response tokens)
    for i, p in enumerate(problems):
        if p["id"] not in resp.index:
            continue
        prompt = tok(format_prompt(tok, p["question"] + SUFFIX), add_special_tokens=False).input_ids
        r = tok.encode(resp.loc[p["id"], "response"], add_special_tokens=False)[: args.max_resp_tokens]
        if len(r) >= 2:
            items.append((i, prompt + r, len(prompt), len(r)))

    model = load_model(args.model, n_loops_override=args.n_loops)
    st = structure(model)
    dev = model.device
    D = model.config.hidden_size
    V = problem_directions(args.seed, args.limit, D).to(dev)
    if args.sites == "all":
        sites = list(st.sites)
    elif args.sites == "mid":
        sites = [st.site_at(k, 0.5) for k in range(st.T)]
    else:
        sites = []
        for s in args.sites.split(","):
            m = re.fullmatch(r"(?:t(\d+)\.)?l(\d+)", s.strip())
            sites.append((int(m.group(1) or 0), int(m.group(2))))
        bad = [s for s in sites if s not in st.order]
        if bad:
            raise ValueError(f"sites {bad} are not sites of this run (Dory input/output sublayers are loop 0)")
    eps_list = [float(e) for e in args.eps.split(",")]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    print(f"{len(items)} responses, {len(sites)} sites x {len(eps_list)} eps", flush=True)

    rows = []
    for b0 in range(0, len(items), args.batch):
        chunk = items[b0: b0 + args.batch]
        ids, valid = pad_batch([c[1] for c in chunk], pad)
        ids, valid = ids.to(dev), valid.to(dev)
        v = V[[c[0] for c in chunk]][:, None, :]  # [B, 1, D]
        sel = torch.zeros_like(valid)  # positions whose next-token prediction is a response token
        for j, (_, _, a0, n) in enumerate(chunk):
            sel[j, a0 - 1: a0 - 1 + n] = True

        def logprobs(hook=None):
            with torch.inference_mode(), stream_hooks(model, hook):
                lg = model(input_ids=ids, use_cache=False).logits  # right padding: valid positions are exact
            return torch.log_softmax(lg.float(), -1)

        lp0 = logprobs()
        p0 = lp0.exp()
        for site in sites:
            for eps in eps_list:
                def inject(s, h, site=site, eps=eps):
                    if s != site:
                        return None
                    n = h.float().norm(dim=-1)  # [B, S]
                    med = torch.stack([n[k][valid[k]].median() for k in range(n.shape[0])])[:, None]
                    keep = (n <= args.sink_ratio * med) & valid
                    keep[:, 0] = False
                    return h + (eps * n.unsqueeze(-1) * v * keep.unsqueeze(-1)).to(h.dtype)

                lp = logprobs(inject)
                kl = (p0 * (lp0 - lp)).sum(-1)  # [B, S]
                agree = lp.argmax(-1) == lp0.argmax(-1)
                for j, (i, _, a0, n) in enumerate(chunk):
                    m = sel[j]
                    rows.append({"problem_idx": i, "loop": site[0], "layer": site[1], "unit": st.unit[site],
                                 "order": st.order[site], "eps": eps, "kl": float(kl[j][m].mean()),
                                 "agree": float(agree[j][m].float().mean()), "kl_last": float(kl[j, a0 - 2 + n]),
                                 "n_tokens": n})
                del lp
        print(f"  {min(b0 + args.batch, len(items))}/{len(items)} responses", flush=True)
    df = pd.DataFrame(rows)
    df["model"] = rd.name
    df["n_loops"] = st.T if st.looped else 1
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    summ = df.groupby(["unit", "loop", "layer", "eps"])[["kl", "agree"]].mean().unstack("eps").round(3)
    with pd.option_context("display.width", 200, "display.max_rows", 400):
        print(summ.to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
