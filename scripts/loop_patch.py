#!/usr/bin/env python3
"""Swap the stream at a loop boundary with another prompt's: do the later loops restore the original?

    scripts/loop_patch.py --gpu 0 --model ByteDance/Ouro-1.4B-Thinking
    scripts/loop_patch.py --gpu 0 --model Qwen/Qwen3-8B        # dense control: 4 blocks of 9 layers
    scripts/loop_patch.py --gpu 0 --model Qwen/Qwen3-8B --every 2   # patch every 2nd site: recovery vs layers left

Pairs a harmful and a harmless train prompt with the same formatted length. In the *target* run, the stream entering
unit k (a loop; for a dense model one of ``--blocks`` blocks of layers) is replaced by the *source* prompt's stream
at the same boundary, on a set of positions (unit 1 is skipped where its input is the token embeddings; Dory's unit 1
input is the input stack's output):
  last  the final prompt token only           (later loops can recover it from the other positions)
  eoi   the template tokens after the user turn (where the refusal direction is defined)
  all   every position                        (only what the model re-injects can bring the target back; Dory adds
                                               its input stack's output before every loop)
Both ways: harmful target / harmless source and the reverse. At every later site and at the final stream it records,
on the prompt-end tokens,
  u_rec     (u.h_patched - u.h_source) / (u.h_target - u.h_source), from pair means: 1 = the target's refusal signal is
            fully restored, 0 = it stays the source's
  dist_rec  1 - |h_patched - h_target| / |h_source - h_target|: the same for the whole state
and at the final stream the next-token KL at the last prompt token to the target's and to the source's distribution
(about 0 for models whose next token after the template is fixed). ``remaining`` = decoder-layer applications after the
patch point, to compare models at equal compute left. ``--every N`` patches at every N-th site in execution order
instead of the unit boundaries (tag ``every<N>``).
Writes ``analysis/loop_patch/<tag>.csv``. HF backend, one GPU.
"""

import argparse
import os
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-pairs", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--u", default=None)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--n-loops", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--every", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from loop_steer.paths import run_dir, setup_job_env
    setup_job_env()
    import pandas as pd
    import torch

    from loop_steer.cot import format_prompt, load_prompts
    from loop_steer.models import eoi_length, load_model, load_tokenizer
    from loop_steer.selfcorr import DEFAULT_U, FINAL, load_dirfile, pad_batch, stream_hooks, structure

    rd = run_dir(args.model)
    args.tag = args.tag or (f"every{args.every}" if args.every else "main")
    out = rd / "analysis" / "loop_patch" / f"{args.tag}.csv"
    if args.skip_existing and out.exists():
        print(f"{out} exists; skipping")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(args.model)
    n_eoi = eoi_length(tok)

    by_len = {}
    for kind in ("harmful", "harmless"):
        for p in load_prompts("train", kind):
            ids = tok(format_prompt(tok, p), add_special_tokens=False).input_ids
            if len(ids) <= args.max_len:
                by_len.setdefault(len(ids), {"harmful": [], "harmless": []})[kind].append(ids)
    rng = random.Random(0)
    pairs = []
    for n, d in sorted(by_len.items()):
        rng.shuffle(d["harmful"]); rng.shuffle(d["harmless"])
        pairs += list(zip(d["harmful"], d["harmless"]))
    rng.shuffle(pairs)
    pairs = pairs[: args.n_pairs]
    print(f"{args.model}: {len(pairs)} length-matched harmful/harmless pairs", flush=True)

    model = load_model(args.model, n_loops_override=args.n_loops)
    st = structure(model, blocks=args.blocks)
    dev = model.device
    u = load_dirfile(rd, args.u or DEFAULT_U[rd.name]).to(dev)
    u = u / u.norm()
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    if args.every:
        points = [s for i, s in enumerate(st.sites) if i % args.every == 0 and s != (0, 0)]
    else:  # unit boundaries, except the token embeddings (same template tokens in both prompts)
        points = [b for b in st.boundaries[: st.T] if b != (0, 0)]

    def run(ids, patch=None):
        """Forward pass -> (boundary states, per-site eoi-mean u-projection [B], per-site eoi states, last logp)."""
        bstates, proj, eoi_h, fin = {}, {}, {}, {}

        def on_site(s, h):
            if patch is not None and s == patch[0]:
                h = torch.where(patch[1][..., None], patch[2].to(h.dtype), h)
                new = h
            else:
                new = None
            x = h if new is None else new
            if s in points:
                bstates[s] = x.detach().clone()
            eoi_h[s] = x[eoi].float().view(x.shape[0], n_eoi, -1)
            proj[s] = (eoi_h[s] @ u).mean(1)
            return new

        def on_final(h):
            fin["h"] = h[eoi].float().view(h.shape[0], n_eoi, -1)

        with torch.inference_mode(), stream_hooks(model, on_site, on_final):
            logits = model(input_ids=ids, use_cache=False).logits
        eoi_h[FINAL] = fin["h"]
        proj[FINAL] = (fin["h"] @ u).mean(1)
        lp = torch.log_softmax(logits[torch.arange(len(ids)), lens - 1].float(), -1)
        return bstates, proj, eoi_h, lp

    rows = []
    for b0 in range(0, len(pairs), args.batch):
        batch = pairs[b0: b0 + args.batch]
        for way, (ti, si) in {"harmful<-harmless": (0, 1), "harmless<-harmful": (1, 0)}.items():
            t_ids, valid = pad_batch([p[ti] for p in batch], pad)
            s_ids, _ = pad_batch([p[si] for p in batch], pad)
            t_ids, s_ids, valid = t_ids.to(dev), s_ids.to(dev), valid.to(dev)
            lens = valid.sum(1)
            eoi = torch.zeros_like(valid)
            for i, n in enumerate(lens.tolist()):
                eoi[i, n - n_eoi: n] = True
            masks = {"last": torch.zeros_like(valid), "eoi": eoi, "all": valid}
            masks["last"][torch.arange(len(lens)), lens - 1] = True
            tb, tproj, th, tlp = run(t_ids)
            sb, sproj, sh, slp = run(s_ids)
            for bnd in points:
                for pname, m in masks.items():
                    _, pproj, ph, plp = run(t_ids, patch=(bnd, m, sb[bnd]))
                    for s in list(st.sites) + [FINAL]:
                        if s != FINAL and st.order[s] < st.order[bnd]:
                            continue
                        dts = (sh[s] - th[s]).norm(dim=-1).mean(1)  # [B]
                        dtp = (ph[s] - th[s]).norm(dim=-1).mean(1)
                        rows.append({"way": way, "patch_unit": st.unit[bnd] + 1, "patch_site": str(bnd),
                                     "remaining": len(st.sites) - st.order[bnd], "positions": pname, "site": str(s),
                                     "unit": st.T + 1 if s == FINAL else st.unit[s] + 1,
                                     "layer": -1 if s == FINAL else s[1], "depth": st.depth(s), "batch": b0,
                                     "n": len(batch), "u_target": float(tproj[s].mean()),
                                     "u_source": float(sproj[s].mean()), "u_patched": float(pproj[s].mean()),
                                     "dist_ts": float(dts.mean()), "dist_tp": float(dtp.mean()),
                                     "kl_target": float((tlp.exp() * (tlp - plp)).sum(-1).mean()) if s == FINAL else None,
                                     "kl_source": float((slp.exp() * (slp - plp)).sum(-1).mean()) if s == FINAL else None})
        print(f"  pairs {b0}-{b0 + len(batch)} done", flush=True)

    df = pd.DataFrame(rows)
    keys = ["way", "patch_unit", "patch_site", "remaining", "positions", "site", "unit", "layer", "depth"]
    agg = df.groupby(keys).apply(lambda g: pd.Series({
        **{c: (g[c] * g.n).sum() / g.n.sum() for c in ("u_target", "u_source", "u_patched", "dist_ts", "dist_tp")},
        "kl_target": (g.kl_target.astype(float) * g.n).sum() / g.n.sum(),
        "kl_source": (g.kl_source.astype(float) * g.n).sum() / g.n.sum()}), include_groups=False).reset_index()
    agg["u_rec"] = (agg.u_patched - agg.u_source) / (agg.u_target - agg.u_source)
    agg["dist_rec"] = 1 - agg.dist_tp / agg.dist_ts
    agg.to_csv(out, index=False)
    fin = agg[agg.site == str(FINAL)]
    with pd.option_context("display.width", 200):
        print(fin.pivot_table(index=["way", "positions"], columns="remaining", values="u_rec").round(2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
