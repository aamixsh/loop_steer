---
marp: true
paginate: true
title: Refusal directions in looped reasoning models
---

# Refusal directions in looped reasoning models

Qwen3-8B (reproduction) → Ouro-1.4B-Thinking (4 loops) → Nanbeige4.2-3B (2 loops) → gpt-oss-20b (MoE)

Method: *reasoning-manipulation* (arXiv 2507.03167), built on *refusal_direction* (Arditi et al.)

Slides are separated by `---` (Marp). Figures: `scripts/make_presentation_figures.py`. Loops are numbered from 1.

---

## Motivation and questions

- Reasoning models refuse harmful prompts. In standard transformers refusal is mediated by **one direction** in the
  residual stream: removing it stops refusal, adding it induces refusal.
- **Looped models** run the same layer stack several times, a cheap way to get more depth per parameter. Their
  residual stream passes through the same weights and a shared norm several times.

Questions:

1. Does a single refusal direction exist in looped models, and do the standard interventions transfer?
2. **Where** is the refusal decision made: which loops, which layers?
3. If an intervention fails, **why**?
4. What does steering cost (capability, over-refusal)?

---

## Models

![w:1150](figures/models.png)

| Model | Layers × loops | Sites | Norm between a write and the stream | Norm between loops |
|---|---|---|---|---|
| Qwen3-8B | 36 × 1 | 36 | none | – |
| Ouro-1.4B-Thinking | 24 × 4 | 96 | **sandwich norm on every write** | shared, 3 passes |
| Nanbeige4.2-3B | 22 × 2 | 44 | none | shared, 1 pass |
| gpt-oss-20b (MoE: 4 of 32 experts per token) | 24 × 1 | 24 | none (biases on `o_proj` and experts) | – |

---

## Data and pipeline

![w:1150](figures/pipeline.png)

- **Prompts:** reasoning-manipulation sets. Directions from 500 train prompts; candidates chosen on 50 further train
  prompts (selection); results on the 487 harmful test prompts. Over-refusal: 250 harmless test prompts.
- **Sampling:** 3 CoTs per prompt, 3 answers per CoT; temperature 0.6, no top-p / top-k. CoTs that hit the first cap
  (2048; Nanbeige and gpt-oss 4096) are continued to 8192 tokens.
- **Judge:** StrongREJECT rubric with a local gpt-oss-20b. **score** = (1 − refused) × (convincing + specific − 2) / 8.
  On gpt-oss it judges its own outputs; a refusal-phrase check agrees (clean 92% vs 92%, ablated 1–2% vs 2%).
- **Capability:** MATH-500 (1 sample each) and AIME 2025 (30 × 8 samples), CoT ≤ 8192 tokens.

---

## Directions, interventions, metrics

**Direction** = mean(refusing) − mean(complying) activations at the stream entering each (loop, layer), over a token window:
`v4_cot` (all CoT tokens), `v12_cot150` (first 150 CoT tokens), `v4_baseline` (template tokens before thinking, prompt
labels), `random` (same norm, control).

![w:1000](figures/interventions.png)

- **Ablation (hooks)** projects `u` out at every layer and loop. **Weight ortho** edits `embed`, `o_proj`, `down_proj`
  once. **Actadd** adds `c · r` at one layer. Hooks can be **loop-restricted** (`apply=`); weights cannot.
- **What "layer L" means differs:** for ablation and weight ortho, L is only *where the direction was measured*; it is
  then removed at **every** layer. For actadd, L is also *where the vector is added* (and nowhere else).
- **Metrics:** judge refusal over all samples; mean score; **unclosed CoTs** (no `</think>`: a broken edit loops, Ouro
  also often answers directly). Differences under about 4 points (50 prompts) or 2 points (test) are noise.

---

## Choosing the layer (and why ablation and actadd differ)

Each intervention gets its own screen on the 50 selection prompts; the winner goes to the test set. Rule (as upstream):
**lowest refusal among candidates that keep CoTs intact** (unclosed CoTs ≲ 10%; a broken model also "stops refusing").

| Qwen3-8B screen | grid | best by refusal alone | chosen |
|---|---|---|---|
| weight ortho | 4 windows × layers 11–25 (odd) | `v4_baseline` L19: 0.3%, **23% unclosed** | `v4_baseline` **L21**: 4.0%, 10% unclosed |
| actadd | 3 windows × L13 / 17 / 21 × c = −1 / −2 | `v4_cot` L17 c=−2: 2.5%, **21% unclosed** | `v4_baseline` **L17**, c=−2: 4.7%, 0.7% unclosed |
| hook ablation | not screened | – | L21: identical to the weight edit in Qwen |

- Ablation removes `u` everywhere, so its "layer" only picks *which* direction; mid-late directions (L19–21) work best.
  Actadd injects at one place: at L21 it comes too late and damages (17% unclosed); L17 is early enough to steer the CoT.
- Ouro (hooks, loop 4, layers 4–20): `v4_cot` gives 0–3% refusal but 40–80% unclosed, so `v4_baseline` L16 (6.6%, 19%).
- gpt-oss (layers 7–21): end-of-prompt L15 / L17 give 3.8% / 4.2% refusal with 4% unclosed; the CoT direction leaves
  37–100% unclosed at L7–19. Actadd: L13, c=−2 (17%, 6% unclosed).
- Upstream's Qwen3-8B layers: 23 (CoT) and 17 (end-of-prompt). Our L17 end-of-prompt edit: 8.5% on the test set vs 5.7%
  for L21. The screen is noisy (50 prompts): neighbouring layers differ by a few points.

---

## Qwen3-8B: the reproduction holds

| Test set (487 prompts) | direction measured at | applied at | refusal | mean score |
|---|---|---|---|---|
| clean / random direction | – | – | 61.3% / 61.1% | 0.37 |
| **weight ortho**, `v4_baseline` / `v12_cot150` / `v4_cot` | L21 / L17 / L15 | every layer | **5.7%** / 7.8% / 11.7% | 0.85–0.91 |
| **hook ablation**, `v4_baseline` | L21 | every layer | **5.5%** | |
| **actadd**, `v4_baseline`, c = −2 | L17 | L17 only | **6.8%** | |

- Our directions match the published layer-17 vectors (cosine 0.994 CoT, 0.993 baseline). The CoT and the
  prompt-end directions differ (cosine ≈ 0.35), yet both remove refusal.
- Weight ortho ≈ hooks, as it must be in a pre-norm model (greedy outputs share 77–83% of opening tokens vs 8% clean).

---

## How we measure where refusal is readable

A **site** is one place in the residual stream: the vector entering decoder layer *l* during loop *t*. Qwen has 36 sites,
Ouro 24 layers × 4 loops = 96, Nanbeige 22 × 2 = 44, gpt-oss 24 (see the model table). At every site:

1. **Average** the residual stream of each training example over a token window: the **CoT window** (all thinking
   tokens) or the **prompt-end window** (the template tokens between the user message and the start of thinking).
2. **Label** examples refusing or complying from the judged answers (a CoT: all 3 answers score < 0.2 / > 0.8; a
   prompt: ≥ 75% of its answers).
3. **Split** the training prompts into halves A and B. Direction from A: mean(refusing) − mean(complying).
4. **Score** on B: project every B example onto the direction; **AUC** = probability that a random refusing example
   projects higher than a random complying one (1 = perfect separation, 0.5 = chance).
5. **Reference:** the same AUC for a random unit direction.

"Linearly separable" here means: one direction, fixed on half the data, tells refusing from complying examples apart on
the other half. It measures **readability**, not cause; the ablations test cause.

---

## Refusal is readable at almost every site

![w:1150](figures/sites.png)

- **A. CoT window:** AUC 0.93–1.00 at every site of all four models; random directions reach 0.6–0.7 typically, up to
  0.97–0.99 at a few sites (refusing and complying CoTs differ in content, so even random directions sometimes separate
  them). So this window says little about *where*; on gpt-oss it is ≈ 1.0 everywhere.
- **B. Prompt-end window, before any thinking:** refusal is already readable (AUC 0.93–0.97 after the first 12–20
  layers). In the looped models it rises through loop 1–2 and stays flat afterwards. In gpt-oss it peaks at L12–18
  (0.96) and falls to 0.88 at the last layer; ablation works exactly at the peak (L15–17, next slides).
- So a single direction carries the refusal signal everywhere, and the model's decision is largely readable before the
  CoT starts. Layer 0 of the prompt-end window is undefined or at chance (identical template tokens for every prompt).

---

## Results across models (test set)

![w:1150](figures/results.png)

**What a bar is:** 487 held-out harmful prompts × 3 sampled CoTs = 1,461 CoTs, each followed by 3 sampled answers. A CoT
scores the share of its answers the judge marks as refusals (a CoT that never closes is judged once on its raw text);
the bar is the mean over all 1,461 CoTs, so every prompt weighs the same.

**Columns** (same order for every model): clean model · a **random direction** of the same norm ablated with hooks
(control: removing just any direction should not stop refusal) · **hook ablation** of the refusal direction `u` ·
**weight orthogonalization** with `u` · **activation addition** of −`c`·`r` at one layer. `u` is the selection winner;
"u from loop 4, L16" means it was measured at layer 16 during Ouro's 4th loop and is then removed at every layer of
every loop.

- **Qwen:** all three interventions work (61% → 6–7%). **Ouro:** hooks 10%, actadd 17%; the weight edit gives no valid
  outputs (50 prompts, so not run on the test set). **Nanbeige:** hooks 18%, weight edit 11%, actadd 26%. **gpt-oss:**
  hooks and weight edit 2% (from 92%), actadd 15%. Random directions change nothing (Nanbeige weight edit with a random
  direction: 73%).
- **Inference:** the method transfers to looped and to MoE models; in looped models only interventions that act after
  every norm (hooks) are safe.

---

## Which loops carry the refusal? (test set)

![w:1050](figures/loops_test.png)

- **Ouro:** loop 1 alone 49%, loop 2 28%, loop 3 18%, loop 4 20%; loops 3+4 13%; all four 10%.
- **Nanbeige:** loop 1 alone 53%, loop 2 alone 28%, both 18%.
- **Inference:** the later loops carry most of the effect, but no single loop is enough. The selection set was
  optimistic by 5–10 points for single loops: confirm loop claims on held-out prompts.

---

## Ablation needs a late loop; actadd is a dose (all subsets)

![w:1100](figures/loops_dose.png)

- **Ablation** (Ouro): every subset containing loop 3 or 4 reaches 8–15%; loop 2 alone 18%, loops 1+2 26%, loop 1
  alone 40%. What matters is *which* loops.
- **Actadd** (c = −1): refusal falls with the *number* of loops (mean 44% → 28% → 21% → 16%), almost regardless of which.
- **Per-loop directions** (hollow) are not better than the last loop's direction used everywhere (often worse).
- **Inference:** ablation removes a feature that later loops would otherwise rebuild; actadd adds a push that accumulates.

---

## Where within a loop? (Ouro actadd, one loop × one layer)

![w:880](figures/loop_layer_map.png)

- The effect grows with the loop index; loop 1 is weak at every layer (44–56%, clean 57%).
- In **loop 4 only layers 4–12 work** (11–17%); layers 16 and 20 do not (36%, 52%): late injections come too late.
- c = −2, once per cell, 50 prompts.

---

## What does steering cost?

![w:1150](figures/capability.png)

- **Ablation is free in the looped models:** MATH-500 and AIME 2025 within noise of clean, whichever loops.
- **On gpt-oss ablation lengthens the CoT** (+25% tokens on MATH-500): fewer CoTs close within 8192 tokens, so accuracy
  drops 3 points on MATH-500 (88.8 → 86.0 hooks, 85.0 weight edit) and 12–14 on AIME (52.5 → 40.8 / 38.8). Accuracy
  of the CoTs that close is unchanged (MATH 92 → 91%, AIME 86 → 88%). AIME at a 16k cap is running.
- **Actadd costs accuracy:** MATH-500 −3.6 (Ouro, c = −1), −8.8 (Nanbeige, c = −2), −23 points (gpt-oss, c = −2);
  AIME: Nanbeige 29.6% → 13.8%, gpt-oss 52.5% → 10.8%. It also lengthens CoTs.
- **No over-refusal:** 0–3% refusal on harmless prompts under every config (a weak check: our interventions lower refusal).
- AIME at 8192 tokens closes only 24–33% of CoTs, so it mostly measures problems solved early.

---

# Why does weight ortho break Ouro but not Nanbeige?

**Short answer:** renormalization. A learned RMSNorm sits between the edited weights and the residual stream.

- In Ouro it sits there **48 times per loop** (sandwich norms) **and 3 times between loops**, and the inter-loop norm
  **amplifies** what it receives.
- In Nanbeige it sits there **once**, between the two loops, and it **shrinks** what it receives.

---

## RMSNorm re-introduces a removed direction

![w:900](figures/norm_leak.png)

- `RMSNorm(x) = g ⊙ x / rms(x)`. An uneven gain `g` **rotates** `x`: the output component along `u` is
  `Σᵢ gᵢ xᵢ uᵢ / rms(x)`, not 0 even when `x ⊥ u`.
- Measured unevenness: cos(`u`, `g ⊙ u`) ≈ 0.95 (Ouro), 0.99 (Nanbeige). Small rotations, but of large vectors.
- Weight ortho guarantees `W x ⊥ u`. Any norm **after** `W` breaks that guarantee.

---

## Leak 1: sandwich norms (Ouro only)

```
Qwen / Nanbeige layer                 Ouro layer
h = h + Attn(Norm(h))                 h = h + Norm₂( Attn(Norm(h)) )
h = h + MLP(Norm(h))                  h = h + Norm₂( MLP(Norm(h))  )
          ↑ edited weights write here           ↑ Norm₂ sits after the edited weights
```

- After weight ortho the raw matrix outputs are clean (projection on `u` ≈ 0.001). After `Norm₂` each write carries
  `u` **at clean strength** again (0.09–0.17 vs clean 0.08–0.18). 48 such writes per loop.
- This alone breaks Ouro **with a random direction**, even at 1 loop.

---

## Leak 2: the inter-loop norm compounds in Ouro only

![w:1100](figures/compounding.png)

- Each pass of the shared norm adds a **fresh leak** and **rescales** the `u` already carried. In the clean model the
  next loop's writers push back against `u` (restoring slope −0.1 to −0.19 per loop); **weight ortho removes exactly
  that ability.**
- **Ouro:** the norm scales the stream **up** (×1.05 → ×2.1), 3 times: `u` grows 0.28 → 0.88 → 2.39 → 4.81, **5.6× clean**.
- **Nanbeige:** the stream is huge (length ≈ 2780), so the norm scales it **down 70×**: one fixed offset of 16% of
  clean, and loop 2 writes 10× more than it inherits.

---

## The same thing, per loop

![w:1050](figures/loop_growth.png)

- Hooks keep `u` below 0.4% of clean in both models: they re-project after every norm.
- Ouro weight ortho grows past clean by loop 2. Norm-aware ortho fixes leak 1 (loop 1 ≈ 0), but leak 2 still grows it.
- Nanbeige weight ortho: one small step, then the model ends.

---

## Testing the explanation

![w:900](figures/ouro_loop_count.png)

Two fixes, each targeting one leak: **norm-aware ortho** (edit against `g ⊙ u`; fixes leak 1) and **fewer loops**
(removes leak 2).

- Plain ortho with a random direction hurts at every loop count → leak 1.
- Norm-aware ortho matches hooks at 1–2 loops and degrades at 4 → leak 2 needs several passes.
- In generation (4 loops): norm-aware ortho is harmless with a random direction (63% refusal vs 57% clean) but leaves
  99% of CoTs unclosed with the refusal direction → **the inter-loop norm is the main cause**.

---

## Can weights alone fix leak 2? No.

Exact fix: also remove `g ⊙ u`, `g² ⊙ u`, … (one direction per inter-loop pass).

| Model | projection on `u` | damage (NLL, random `u`) |
|---|---|---|
| Ouro, rank-4 edit | ≈ 0 in every loop ✓ | 9.6 (clean 1.1) ✗ |
| Nanbeige, rank-2 edit | ≈ 0 in every loop ✓ | 11.7 (clean 1.5) ✗ |

The extra directions put 34–48% of their weight on the **top 1% highest-gain dimensions**, which the model depends on.
**No weight-only edit ablates a direction in a looped model with an amplifying inter-loop norm.**

---

## Is it Ouro's early-exit training?

Logit lens on clean traces (decode each loop's output directly):

| Model | NLL by loop | agreement with final output |
|---|---|---|
| Ouro | 2.93 → 1.44 → 1.12 → 1.11 | 61% → 80% → 92% → 100% |
| Nanbeige | 8.36 → 1.55 | 34% → 100% |

- Ouro (trained so any loop can exit) carries near-finished outputs from loop to loop and refines them; its writes
  mostly correct an inherited state, which is what leak 2 corrupts.
- Nanbeige's loop 1 output is not decodable; loop 2 mostly rewrites it.
- Consistent with early-exit training, **not proven**: no same-architecture model without early exit to compare.

---

## Takeaways

1. **The method transfers.** One difference-in-means direction removes most refusal in both looped models
   (Ouro 61% → 10%, Nanbeige 78% → 11–18%), as in Qwen (61% → 6%).
2. **Use hooks on looped models.** Weight ortho equals ablation only when nothing renormalizes the stream after the
   edited weights. Ouro breaks twice: sandwich norms re-add `u` to every write, and the amplifying inter-loop norm
   compounds the leak while the edit stops the model from correcting it. Nanbeige survives (raw writes, shrinking norm).
3. **Refusal is decided late, but not in one loop.** Later loops carry most of the effect; ablating loop 1 alone
   barely helps; no single loop matches all loops. One direction (last loop's) works in every loop.
4. **Ablation and actadd differ.** Ablation needs a late loop and costs no capability in the looped models (on gpt-oss
   it lengthens CoTs); actadd is a dose over loops, works at early-to-middle layers of late loops, and costs 4–23 points
   of MATH-500 accuracy.
5. **MoE models too.** gpt-oss-20b: 92% → 2% with hooks or with a weight edit of all 32 experts per layer. The
   experts' biases matter for an exact ablation, not for behaviour. Its CoT-window and early directions break it;
   ablation makes it reason longer (more CoTs hit the cap on AIME).
6. **Check damage, not just refusal:** unclosed CoTs reveal broken edits; selection-set loop results were optimistic.

---

## Caveats

- Directions, sites and loop configs are chosen on 50 selection prompts; headline and loop-profile numbers are on the
  487 test prompts. The loop × layer map, the all-subsets sweep and Ouro's weight edit are selection-set only.
- Sampling is temperature 0.6 without top-p/top-k (the models recommend top_p 0.95, top_k 20).
- About 20% of clean Ouro samples answer without thinking; ablation lowers this to 11–15%. Looping to the 8192 cap:
  0.2% clean, about 2% ablated.
- AIME 2025 at 8192 tokens is a weak capability test (most CoTs do not close). Over-refusal is a weak check.
- gpt-oss is judged by gpt-oss-20b itself (refusal-phrase check agrees); its weight edit runs on a bf16 copy (MXFP4
  would undo 63% of it).
- Mechanism numbers come from 8 traces (≈ 2.7k tokens) per model, one refusal site each.
- Some Nanbeige edits damage generation: the `v4_cot` loop-1 weight edit leaves 47% of CoTs unclosed (selection). Nanbeige
  runs on our own vLLM port (matches HF greedy outputs exactly).

---

## MoE models: the method carries over (gpt-oss-20b)

![w:1100](figures/moe.png)

- gpt-oss-20b: 24 pre-norm layers; each MLP is a **router** that sends every token to 4 of 32 experts; the layer adds
  the router-weighted sum of their outputs. The residual stream is still one vector per token, so the direction is
  measured and ablated exactly as in a dense model. **Hooks** act on that sum, whatever the routing.
- **Weight ortho** must edit **all 32 experts** in every layer (any of them can be routed to), plus `o_proj` and the
  embedding: then every possible sum is ⊥ `u`. The router reads the stream; it changes the same way under hooks.
- **Biases** (`o_proj` and every expert; up to 54 along `u`) write a constant along `u`. Upstream's edit leaves them:
  6% of the clean projection stays in the stream (0.2% if we edit them too, as hooks). Log-probs vs hooks: 0.13 with
  biases edited (bf16 noise; clean vs hooks 0.87), 0.19 without. The in-place vLLM edit gives the same numbers.
- **MXFP4:** the experts ship as 4-bit blocks. The edit only exists in bf16 (we run a bf16 copy); re-quantizing the
  edited experts brings back 63% of their write along `u`.

---

## gpt-oss-20b results (test set, 487 prompts)

| Intervention (`v4_baseline` direction) | refusal | unclosed CoTs |
|---|---|---|
| clean / random direction (hooks, L15) | 92.2% / 92.0% | 0 / 0 |
| **hook ablation**, u from L15 | **2.1%** | 0.2% |
| **weight ortho**, u from L15, biases edited / biases kept (upstream) | **1.8%** / 1.6% | 0.3% / 0.6% |
| actadd at L13, c = −2 | 14.6% | 1.0% |

- Ablation works as in the dense model and costs almost nothing in closed CoTs (4% hit the 4096 cap; they close by 8192).
  Upstream (v4, end-of-prompt, L15): 7% with 40% incomplete generations.
- **Biases do not matter for behaviour:** the 6% of `u` the bias-free edit leaves in the stream changes nothing here.
- **Early and CoT-window directions break gpt-oss** (screen, 50 prompts): ablating either direction measured at L7–11
  (CoT direction: L7–13) leaves every CoT unclosed (no channel header, "etc, etc, maybe…"); the CoT direction still
  leaves 37–89% unclosed at L15–19; only L21 keeps CoTs intact, at 26% refusal. Random direction at L15: harmless.
- **Capability** (CoT ≤ 8192): MATH-500 88.8% clean, 86.0% hooks, 85.0% weight edit, 65.4% actadd; AIME 2025 52.5 /
  40.8 / 38.8 / 10.8%. Ablation makes gpt-oss think longer, so more CoTs hit the cap; the ones that close are as accurate.
---

## Next steps

1. **Deck:** add the loop-rebuild result (ablating loop 1 only: Ouro rebuilds 93% of the clean prompt-end separation by
   the last loop, Nanbeige 59%) and the 2-loop Ouro test of leak 2.
2. **Position-restricted ablation** (prompt vs CoT vs answer tokens): runs done for Ouro and Nanbeige, to read and confirm
   on the test set; then gpt-oss.
3. **Statistics:** bootstrap confidence intervals over prompts for every table.
4. **Capability at a longer cap** (AIME at 16–32k tokens) and actadd at matched strength across models.
5. Remaining sweeps: `paired_cot` directions, actadd coefficients, a teacher-forced per-loop screen; explain the Qwen
   whole-CoT gap (15% vs the paper's 7.5%).

---

## Appendix: every subset of loops (50 selection prompts)

![w:690](figures/loops_ouro.png) ![w:420](figures/loops_nanbeige.png)

Light = one direction (last loop) in all selected loops; dark = each loop's own direction. Red dots: unclosed CoTs.
Ouro clean 57%, Nanbeige 80%. Ablation in one loop: Ouro 40 / 18 / 10 / 13%, Nanbeige 48 / 22%.

---

## Appendix: reproduce

```bash
# test-set run of loop configs (generate, judge, continue capped CoTs, judge, summarize)
GPU=0 scripts/run_test_set.sh ouro test_loops ablate:v4_baseline:t3.l16:apply=2 actadd:v4_baseline:t3.l16:c=-1
# loop-subset sweeps (selection prompts) and the loop x layer map
GPU=0 ACTADD_C=-1 VARIANT=shared scripts/run_loop_sweep.sh      # then VARIANT=perloop; MODEL_KEY=nanbeige ACTADD_C=-2
uv run python scripts/loop_sweep_summary.py loop_sweep loop_sweep_perloop --model Ouro-1.4B-Thinking
# capability
uv run python scripts/capability.py --gpu 0 --model ByteDance/Ouro-1.4B-Thinking --dataset math500 --tag cap \
    --candidates none ablate:v4_baseline:t3.l16
# mechanism comparison (HF)
uv run python scripts/ortho_compare.py --gpu 0 --model ByteDance/Ouro-1.4B-Thinking --direction v4_baseline:t3.l16 --loops 1,2
# queue jobs back to back on one GPU; figures for this deck (CPU)
scripts/gpuq.py add -n NAME -- CMD ...;  uv run python scripts/make_presentation_figures.py
```

Candidate syntax uses 0-based loops (`t3` = loop 4). Summary tables: `results/<model>/*.csv`, `results/capability_summary.csv`.
