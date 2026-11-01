# Paper plan: experiments to establish the looped-model claims

What has to be run before the looped-model results can go into a blog post or paper. Edit in place. Results go to
`results/`, the deck and the README as usual; `PLAN.md` keeps the day-to-day queue and points here.

## Claims and where they stand

| Claim | Current evidence | Main gap |
|---|---|---|
| **C1. Looped models self-correct a removed direction.** Ablating u in an early loop is undone by the next loop; the last loops decide. | Loop-subset ablation on the test set (Ouro, Nanbeige), selection set (Dory). Rebuild on teacher-forced traces: Ouro loop 2 recomputes 83% of u's separation, Nanbeige 67%. Clean restoring-force slopes in Ouro loops 2–4. | Not shown to be specific to loops (no dense control). Different setups per model. Rebuild only on clean teacher-forced traces, not Dory. Pushes (actadd) are only partly corrected. |
| **C2. Learned norm gains reintroduce a removed direction.** | Leak 1 and leak 2 measured in Ouro (8 traces); norm-aware edit and 2-loop runs behave as predicted; Dory weight edit gets half the hooks' effect. | Correlational: no intervention on the gain itself. One direction and site per model, 8 traces. Says nothing about C1 (the inter-loop norm restores only 4% / 0% of the rebuild). |
| **C3. Looped models resist weight-only abliteration.** | Write-side weight edit breaks Ouro, half-works on Dory, works on Nanbeige. Exact rank-T edits destroy the model. | Only one attack family tried. Fewer loops (ours, arXiv 2610.15218) already bypass it. No reader-side edit, no fine-tuning attack, no non-looped sandwich-norm control. |

Related work (deck slide "Concurrent and prior work"): the LessWrong post (24 Sep 2026), arXiv 2610.15218 and
2610.10625 (both Oct 2026) are concurrent work and are cited as such: this project started in late September
2026, and each appeared within a week of the start or during it. They overlap with us: the LessWrong post finds
that steering works best in the last loop, and 2610.15218 finds that fewer loops remove refusal and that the direction
settles late. Prior work: dense-model self-repair (McGrath et al. 2023, Rushing & Nanda 2024) and the Gemma 2
norm-after-write note. Our contribution: the causal loop map with a judge across three architectures, the rebuild and
its cause, and the weight-edit mechanism.

## A. One setup for all models

Today the three looped models differ in direction recipe, source loop and evaluation set. Fix one protocol and rerun.

1. **Direction recipe.** Ouro and Nanbeige use refusing − complying at the prompt end (`v4_baseline`); Dory uses
   harmful − harmless (`hh_eoi`). Extract `hh_eoi` for Ouro and Nanbeige (`scripts/contrast_prompts.py`, as in
   `run_dory.sh contrast`) and report both recipes on all three. Headline = `hh_eoi` (Arditi recipe, no dependence on
   how many complying samples a model gives).
2. **Source site.** u comes from loop 4 L16 (Ouro), loop 2 L15 (Nanbeige), loop 1 L26 (Dory). Pick one rule (same
   relative depth in the last loop, chosen on the selection set by the same metric) and report a source-loop sweep (u
   from each loop, applied everywhere) for every model.
3. **Loop counts.** Each model at its trained count and one other: Ouro 4 and 2, Nanbeige 2 and 3, Dory 2 and 4. Needs
   `--n-loops` in `extract_activations.py`, `capability.py` and the rebuild script where missing.
4. **Evaluation.** All 487 test prompts, StrongREJECT judge, capped CoTs continued to 8192, invalid CoTs counted, same
   sampling (temperature 0.6; repeat the headline with the models' recommended top_p 0.95 / top_k 20).
5. **Config grid per model** (test set): clean, random u, hooks in all loops, each loop alone, each leave-one-out set,
   last loop alone, input/output stacks where they exist, actadd all loops and per loop, weight edit (plain,
   norm-aware), position-restricted hooks (prompt, CoT, answer, prompt + CoT).
6. **Dense references** under the same protocol: Qwen3-8B and gpt-oss-20b (hooks, weight edit, actadd).

## B. Statistical validity

1. **Bootstrap intervals over prompts** for every number (`summarize.py`, `loop_sweep_summary.py`), and paired
   bootstrap or McNemar tests for the comparisons we state (loop 1 vs loop 4, hooks vs weight edit).
2. **Seeds:** 3 sampling seeds (`--seed`) for the headline and loop-subset configs; report seed spread next to the
   prompt bootstrap.
3. **Move selection-only claims to the test set:** the all-subsets sweep, the loop × layer map (a coarser grid), the
   position-restricted runs, Dory's loop subsets, Ouro's weight edit and the 2-loop runs.
4. **Second judge.** A different judge model on all headline runs plus a hand-labelled sample of 100 answers per model
   (gpt-oss currently judges itself). Report agreement.
5. **Second harmful set** (e.g. HarmBench or JailbreakBench) for the headline and loop-1 / last-loop configs.
6. **Random-direction controls** with matched norm for every intervention type, including actadd per loop.
7. **Mechanism sample size:** rebuild and leak measurements on ≥ 100 traces per class, 3 directions (two recipes plus
   random) and 2 sites per model, with intervals.

## C. Self-correction (C1)

What we already have on actadd across loops: Ouro all 15 loop subsets at L16, c = −1 (selection): one loop 38–51%,
two 23–31%, three 18–25%, all 16%, clean 57%; Ouro loop × layer map, c = −2 (loop 1 weak at every layer, loop 4
works at L4–12); Nanbeige c = −2 per loop 59 / 44 / 24%; Dory c = −1 in loop 1 only 84–92% (no effect); test set:
actadd in all loops only. Missing: per-loop actadd on the test set, dose curves, perturbation tracking, the +u
direction, and any dense comparison.

1. **Dense control (decides whether loops matter).** Qwen3-8B and gpt-oss: ablate u only in the first quarter, half or
   three quarters of the layers; measure refusal and the rebuild along depth. Needs a `layers=` option for hooks
   (also wanted for Dory site-only / final-stream-only). If dense models rebuild as much over the same number of
   layers, C1 becomes "ordinary self-repair, repeated per loop".
2. **Rebuild on every model with one script.** Dory branch for `loop_rebuild.py` (input stack re-added before each
   loop; separate re-injection from recomputation with `apply=-1,0`). Same windows, same directions as section A.
3. **Rebuild in generation, not just teacher forcing.** Measure u's separation on the model's own CoT under
   loop-restricted ablation, per CoT position; and the logit-lens refusal probability after each loop.
4. **Actadd dose by loop.** Coefficient sweep (c = −0.5 to −4) for a push in loop 1 only vs the last loop only, at an
   early-middle layer (Ouro L8, Nanbeige and Dory equivalents), with matched random pushes. A fading push needs a
   larger c in loop 1 for the same effect; if the curves have the same shape and only shift, the early loop is just
   weaker, not corrected.
5. **Perturbation tracking.** Inject c·u (and a random vector of the same norm) at loop 1; at every later site record
   the norm of steered − clean, its component along u and its cosine with u. Tests the "rotates away" account of the
   LessWrong post with a control, and compares with the same injection in a dense model.
6. **The other sign.** Add +u on harmless prompts in loop 1 only vs the last loop only: does induced refusal also get
   corrected? A set point should pull both ways.
7. **Response function of the restoring force.** At the start of loop k, set u's coordinate to the clean mean plus an
   offset (−3σ to +3σ) and measure where loop k leaves it. A mean-reverting slope < 1 per loop, steeper in later
   loops, would make the set-point claim causal (today it is a correlation).
8. **What recomputes u.** Attribute loop 2's write along u to attention vs MLP and to heads (direct attribution);
   ablate the top contributors together with loop-1 ablation and check whether the effect then persists.
9. **What it is recomputed from.** Probe harmfulness (harmful vs harmless) after a loop-1 ablation; ablate u and the
   harmfulness direction together in loop 1 and test whether the rebuild stops.
10. **Beyond the trained loop count.** Ouro and Dory with extra loops: does the set point hold, and does ablating only
    the newest loop still work?

## D. Normalization mechanism (C2)

1. **Gain flattening (causal).** In weight-edited Ouro, replace the inter-loop norm's gain by its mean (and, separately,
   the sandwich norms' gains): leak 2 (leak 1) should vanish. Report the damage the flattening does to the clean model.
2. **Non-looped sandwich-norm control:** Gemma 2 (2B or 9B). Prediction: leak 1 without compounding; the weight edit
   works worse than hooks by a fixed amount.
3. **Dory leak measurement:** `ortho_compare`-style branch (per-sublayer lr, no final norm); then a blend-aware weight
   edit, the analogue of norm-aware ortho.
4. **Leak vs loop count with continued CoTs:** Ouro at 2, 3 and 4 loops, plain / norm-aware / hooks, capped CoTs
   continued, so unclosed CoTs are not confounded with longer thinking.
5. **State the mechanism precisely:** dividing by rms(x) keeps x ⊥ u; the learned per-dimension gain (RMSNorm weight,
   nGPT eigen learning rate) rotates it. Report cos(u, g ⊙ u) and the leak for each norm in each model.

## E. Tamper resistance (C3)

Define the threat model first: an attacker with the weights who wants a refusal-free model that still reasons.

1. **Weight-only edits:** plain, norm-aware, rank-T, reader-side (project g ⊙ u out of every matrix that reads the
   stream and the unembedding), blend-aware for Dory. Refusal, unclosed CoTs, MATH-500.
2. **Fewer loops** alone and with each edit (Ouro at 2 loops, Dory at 1), with capability at that loop count.
3. **Fine-tuning attacks:** LoRA on a few hundred harmful prompts with compliant answers, and distillation of the
   hooked model into the weights; Ouro, Nanbeige, Dory vs Qwen3-8B at matched budgets. Report the cost (steps, samples)
   to reach 10% refusal at < 5 points MATH-500 loss.
4. **Wording rule:** unless items 1–3 show a real gap with the dense models, the claim is "the standard abliteration
   recipe does not transfer", not "tamper resistant".

## Order and rough cost

Rough GPU time on one RTX PRO 6000: a test-set config is about 0.5–1 h (generation, judge, continuations), a rebuild
or tracking run under 1 h, a capability run 1–2 h.

1. Code: `layers=` hook option, `--n-loops` everywhere, Dory rebuild branch, bootstrap in the summaries (CPU, 1 day).
2. C1 dense control and perturbation tracking (C.1, C.5): these can overturn the main claim. ~1 GPU-day.
3. Section A grid on the test set for the three looped models, 3 seeds on the key configs. ~6–8 GPU-days.
4. C.4, C.6, C.7 (dose, sign, response function). ~2 GPU-days.
5. D.1–D.3 (gain flattening, Gemma 2, Dory leak). ~1–2 GPU-days.
6. E (attacks), fine-tuning last. ~3–4 GPU-days.
7. B.4–B.5 (second judge, second harmful set) on the final configs. ~2 GPU-days.

## What would change the story

- Dense models rebuild as much after early-layer ablation → C1 is not about loops.
- An early push survives at a larger dose with the same curve shape → "weaker early loops", not correction.
- Gain flattening does not stop the leak → C2 is wrong about the cause.
- A reader-side edit or a cheap LoRA removes refusal as easily as in Qwen → drop "tamper resistant".
