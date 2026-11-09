#!/usr/bin/env python3
"""Task performance under random kicks: multiple-choice accuracy (ARC-Challenge), no thinking, teacher-forced.

    scripts/kick_eval.py --gpu 0 --model ByteDance/Ouro-1.4B-Thinking
    scripts/kick_eval.py --gpu 0 --model ByteDance/Ouro-1.4B-Thinking --n-loops 8 --tag loops8

The performance counterpart of ``perturb_response.py --kick-mask nosink``: the same kick (``eps * |h| * v`` per token
at one site, ``v`` = the k-th random direction of that script, attention sinks untouched: the first token and tokens
with |h| > ``--sink-ratio`` x the sequence median) at the same sites (every unit at fractions ``--fracs`` of its
layers; dense models: ``--blocks`` virtual units). The prompt asks for the letter of the right option and ends with the
model's no-thinking assistant prefix plus ``The answer is (``; the next token is read off over the four letters.

Per (site, direction, eps): accuracy (argmax over the letters), mean probability of the right letter (renormalised
over the letters), the probability mass on the four letters at all (format kept), and KL / flip rate of the letter
distribution against the clean model. Writes ``analysis/kick_eval/<tag>/mc.parquet``. HF backend, one GPU.
"""

import argparse
import os

LETTERS = "ABCD"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300, help="ARC-Challenge test questions with four options")
    ap.add_argument("--eps", default="0.1,0.2,0.3,0.5,0.7,1.0,1.5,2.0")
    ap.add_argument("--fracs", default="0,0.25,0.5,0.75")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--sink-ratio", type=float, default=10.0)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--n-loops", type=int, default=None)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from loop_steer.paths import run_dir, setup_job_env
    setup_job_env()
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download

    from loop_steer.models import load_model, load_tokenizer
    from loop_steer.selfcorr import no_think_prompt, pad_batch, stream_hooks, structure

    rd = run_dir(args.model)
    out = rd / "analysis" / "kick_eval" / args.tag
    if args.skip_existing and (out / "mc.parquet").exists():
        print(f"{out}/mc.parquet exists; skipping")
        return
    out.mkdir(parents=True, exist_ok=True)

    # ---- questions
    path = hf_hub_download("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet", repo_type="dataset")
    qs = []
    for r in pd.read_parquet(path).itertuples():
        if len(r.choices["text"]) != 4 or r.answerKey not in list(r.choices["label"]):
            continue
        qs.append({"q": r.question, "opts": list(r.choices["text"]), "ans": list(r.choices["label"]).index(r.answerKey)})
        if len(qs) == args.n:
            break
    tok = load_tokenizer(args.model)
    gpt_oss = "gpt-oss" in args.model
    seqs = []
    for q in qs:
        body = q["q"] + "\n\n" + "\n".join(f"{L}. {o}" for L, o in zip(LETTERS, q["opts"]))
        msg = [{"role": "user", "content": body + "\n\nAnswer with the letter of the correct option only."}]
        text = no_think_prompt(tok, msg)
        if gpt_oss:  # Harmony: empty analysis channel, then the final channel
            text += "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
        seqs.append(tok(text + "The answer is (", add_special_tokens=False).input_ids)
    letter_ids = []
    for L in LETTERS:
        ids = tok(L, add_special_tokens=False).input_ids
        assert len(ids) == 1, (L, ids)
        letter_ids.append(ids[0])
    answers = torch.tensor([q["ans"] for q in qs])
    print(f"{len(qs)} questions; prompt tail: {tok.decode(seqs[0][-24:])!r}", flush=True)

    model = load_model(args.model, n_loops_override=args.n_loops)
    st = structure(model, blocks=args.blocks)
    dev = model.device
    D = model.config.hidden_size
    g = torch.Generator().manual_seed(0)  # perturb_response.py's random0, random1, ...
    dirs = {f"random{k}": torch.randn(D, generator=g) for k in range(args.n_random)}
    dirs = {k: (v / v.norm()).to(dev, torch.float32) for k, v in dirs.items()}
    batches = [seqs[i:i + args.batch] for i in range(0, len(seqs), args.batch)]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    lid = torch.tensor(letter_ids, device=dev)

    def run(inject=None):
        """[N, 4] letter log-probs (over the letters) and [N] probability mass on the letters."""
        lps, mass = [], []
        for b in batches:
            ids, valid = pad_batch(b, pad)
            ids, valid = ids.to(dev), valid.to(dev)
            last = valid.sum(1) - 1
            hook = None if inject is None else (lambda s, h, valid=valid: inject(s, h, valid))
            with torch.inference_mode(), stream_hooks(model, hook):
                logits = model(input_ids=ids, use_cache=False).logits
            lp = torch.log_softmax(logits[torch.arange(len(b), device=dev), last].float(), -1)
            del logits
            sel = lp[:, lid]
            mass.append(sel.exp().sum(-1).cpu())
            lps.append(torch.log_softmax(sel, -1).cpu())
        return torch.cat(lps), torch.cat(mass)

    def row(lp, mass, lp0, **kw):
        p = lp.exp()
        return {**kw, "acc": float((lp.argmax(-1) == answers).float().mean()),
                "p_correct": float(p[torch.arange(len(qs)), answers].mean()), "letter_mass": float(mass.mean()),
                "kl": float((lp0.exp() * (lp0 - lp)).sum(-1).mean()),
                "flip": float((lp.argmax(-1) != lp0.argmax(-1)).float().mean())}

    lp0, m0 = run()
    rows = [row(lp0, m0, lp0, inj_unit=-1, inj_layer=-1, dir="clean", eps=0.0)]
    print(f"clean: acc {rows[0]['acc']:.3f}, p_correct {rows[0]['p_correct']:.3f}, letter mass {rows[0]['letter_mass']:.3f}",
          flush=True)

    eps_list = [float(e) for e in args.eps.split(",")]
    fracs = [float(f) for f in args.fracs.split(",")]
    inj_sites = sorted({st.site_at(k, f) for k in range(st.T) for f in fracs}, key=lambda s: st.order[s])
    for site in inj_sites:
        for dname, v in dirs.items():
            for eps in eps_list:
                def inject(s, h, valid, site=site, v=v, eps=eps):
                    if s != site:
                        return None
                    n = h.float().norm(dim=-1)  # [B, S]
                    med = torch.stack([n[i][valid[i]].median() for i in range(n.shape[0])])[:, None]
                    keep = (n <= args.sink_ratio * med) & valid
                    keep[:, 0] = False
                    return h + (eps * n.unsqueeze(-1) * v * keep.unsqueeze(-1)).to(h.dtype)

                lp, m = run(inject)
                rows.append(row(lp, m, lp0, inj_unit=st.unit[site], inj_layer=site[1], dir=dname, eps=eps))
        r = pd.DataFrame([x for x in rows if x["inj_layer"] == site[1] and x["inj_unit"] == st.unit[site]])
        print(f"  {site} (unit {st.unit[site]}): acc by eps " +
              " ".join(f"{e:g}:{a:.2f}" for e, a in r.groupby("eps").acc.mean().items()), flush=True)
    df = pd.DataFrame(rows)
    df["model"] = rd.name
    df.to_parquet(out / "mc.parquet")
    print(f"-> {out}/mc.parquet")


if __name__ == "__main__":
    main()
