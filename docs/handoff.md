# Handoff notes

State of the project and what to do next. For setup on a new machine see "Moving to another machine" in the
README. Results and the explanation of the weight-orthogonalization difference are in the README and
[presentation.md](presentation.md).

## What is done

- Pipeline (sampling, judging, labels, directions, interventions) for Qwen3-8B, Ouro-1.4B-Thinking and Nanbeige4.2-3B.
- Test-set results (487 prompts × 3 CoTs × 3 answers) for the best candidates per model, with capped CoTs continued
  to 8192 tokens (`intervene.py --extend-from`, merged by `summarize.py --cont`). Clean Qwen and Ouro were
  regenerated at an 8192 cap (`clean_test_8k`).
- Mechanism for why weight orthogonalization breaks Ouro (two norm leaks) and not Nanbeige.
- Comparison with the source paper (arXiv 2507.03167): Qwen3-8B clean 61% vs 60%, prompt-end direction 8.5% vs 8.5%,
  whole-CoT direction 15.0% vs 7.5%. The CoT-direction gap is not explained yet.

## Open items, roughly by value

1. **Capability and over-refusal checks** on the best candidates. The paper reports 5–30% relative loss on AIME 2025,
   GPQA Diamond and BBEH Mini for Qwen3-8B. We have none.
2. **Where the decision is made.** Ablation acts on every position, so it cannot say. Add a position restriction
   (prompt tokens / CoT tokens / answer tokens) to the hooks in `src/loop_steer/vllm_hooks.py` and `hooks.py`, then run
   `v4_baseline` and `v4_cot` with each. `apply=` already restricts by loop.
3. **Qwen hook ablation on the test set** (only weight edits were run) and the `v4_cot` direction on Ouro with the
   8192 cap (selection-set runs had many unclosed CoTs at the 2048 cap).
4. **Explain the whole-CoT gap on Qwen.** Candidates: our 500 train prompts × 3 CoTs vs the paper's ~1460 × 5,
   labelling by 3 answers vs 5, top_p 1.0 vs 0.95.
5. **Norm-aware ortho end to end on Ouro.** `orthogonalize_(..., norm_aware=True)` exists but is not wired into
   `intervene.py`. Analysis predicts it still leaks through the inter-loop norm; test it at 2 and 4 loops.
6. `paired_cot` and activation-addition sweeps on the looped models; a teacher-forced per-loop screen for Ouro.
7. About 20% of clean Ouro samples answer without thinking; look at which prompts.
8. Repo housekeeping before making it public: choose a license; decide whether to rewrite git history.

## Conventions and pitfalls

- Run directories: `data/runs/<model>/{activations,directions,generations,analysis,figures}`; logs in `data/runs/logs`.
  Generation tags: `test_*` (test set), `*_ext` (continuations of capped CoTs), `clean_test_8k` (clean at 8192).
- Stage-1 responses without `</think>` are saved in `<tag>/invalid/<candidate>.parquet`. `hit_limit` separates capped
  ones from direct answers (older files lack it; `summarize.py` falls back to `n_tokens >= cap`). Judge them with
  `judge.py <tag> <tag>/invalid`; skip-existing makes every stage resumable.
- `summarize.py --max-tokens` must be the stage-1 cap of the main run (2048 for Qwen and Ouro, 4096 for Nanbeige).
- `ortho.orthogonalize_` must keep the keyword `direction`: `intervene.py` calls it by keyword. After renaming a
  parameter, grep every caller.
- Hooks inside vLLM need `VLLM_ALLOW_INSECURE_SERIALIZATION=1` and eager mode; Nanbeige needs its in-process port
  (`vllm_nanbeige.py`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`). The scripts set both.
- Sampling is temperature 0.6, top_p 1.0, no top_k, for every model.
- The judge (gpt-oss-20b) and Qwen3-8B are pinned in code to the snapshots used for the reported numbers. Changing
  them changes the numbers.
- `scripts/run_*.sh` are the batch chains used on the original machine (GPU index via `GPU=`); the `run_test_*` ones
  are historical. They are all skip-existing.
