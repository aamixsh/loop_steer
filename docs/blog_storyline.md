# Blog storyline: looped models undo steering, until the last loop

Draft 4, 14 Nov 2026. Rebuilt around the complete results of experiments E1–E14 (`docs/experiments.md`). It replaces
draft 3, whose spine was "where you steer includes when" and which rested on 50-prompt runs.

Numbers below are from the pooled test set (487 prompts, `grid_ci.py .../grid_test{,_b}`) unless a cell says 250.
Refusal intervals are about ±3 to ±5 points. Kick accuracies use 100 MATH-500 problems, about ±9 points.

## Headline

> Steer a looped model in an early loop and the loops that follow undo it. Give it extra loops and it can undo
> steering in every loop it was trained with. A dense model has no second pass: steer its middle layers and the
> effect stays, steer its last layers and nothing happens. The one thing loops cannot undo is an edit that repeats
> on every pass, and that is exactly what a weight edit is.

"Self-correction" is used operationally throughout: **later loops restore the behaviour that an intervention in an
earlier loop removed.** It is not a claim that the model detects an attack.

Working title: **Second laps**. Alternatives: "Kick it early and it shrugs", "The model that heals until the last lap".

## The headline against the complete results

| Claim | Verdict | Evidence |
|---|---|---|
| Later loops undo first-loop steering | Holds in all three | Refusal recovered after ablating loop 1 only: Ouro 76%, Nanbeige 57%, Dory 89% |
| Extra loops undo steering of every trained loop | Complete in Dory, partial in Ouro | Dory refusal 12% at 2 loops, 91% at 4. Ouro recovers kicks, not refusal steering (11% to 16%) |
| The last loop is never corrected | Holds in all three | Last loop alone keeps 83 / 86 / 95% of the full ablation effect |
| Dense: middle sticks, ends do not | Holds | Qwen quarters 58 / 17 / 7 / 54% against 62% clean |
| Same pattern for personas and random kicks | Holds, strongest in Dory | Sections 4 and 5 |
| Looped models resist weight edits | Ouro only | Nanbeige and Dory are abliterated with no capability loss |

Three limits the post has to state itself, because a careful reader will find them:

1. **It is not a defence.** Steer the last loop, or every loop, and all three models fall as a dense model does.
2. **Ouro is the weak case.** Only its first loop is forgiven. Loop 2 alone already keeps 70% of the effect, and
   two extra loops do not bring refusal back.
3. **Tamper resistance belongs to an architecture, not to looping.** Two of three looped models take the weight
   edit cleanly. Hooks remove refusal from Ouro at no capability cost.

## Visual language

One metaphor carries every figure: **laps on a track.** The layer stack is a circuit. A token's state is a runner
that goes round it once per loop. A dense model is one long straight cut into four stretches. Steering is a shove.
A weight edit bends the track itself, so the shove repeats on every lap.

- One reusable component, the **track**: a ring with lap markers, a runner, a racing line, and an output gauge.
  Each experiment re-skins the shove: a red eraser (ablation), a costume (persona), a gust (random kick), a bent
  rail (weight edit).
- Every animation ends on a measured number. The runner's wobble is illustration, the gauge is data. Each gauge
  shows its interval and its prompt count on hover.
- Real model text wherever it is harmless: persona replies in full, refusals in full, compliant replies as a badge
  plus the first few words only.
- Colour: one hue per model family (three looped, one dense), one accent for "steered here".

## Flow

| # | Section | Interactive | Carries |
|---|---|---|---|
| 0 | Shove it, then add laps | Dory track with a "+2 laps" button | The whole claim in ten seconds |
| 1 | What a looped model is | A stack that curls into a ring | Setup |
| 2 | Erase refusal in some loops | The switchboard | Experiment 1 |
| 3 | Add laps | Lap-count slider | Experiment 1, recovery |
| 4 | Costumes | Persona dial with real replies | Experiment 2 |
| 5 | Kick the runner | Kick game with a strength slider | Experiment 3 |
| 6 | So do loops self-correct? | Scoreboard matrix | Synthesis and limits |
| 7 | Bend the track | "Abliterate" buttons with gauges | Weight edits |
| 8 | Why one model breaks | Leak animation with a lap counter | Mechanism |
| 9 | Two design levers | Three architecture cards | Takeaways |
| 10 | Methods and limits | Plain text | Honesty |

### 0. Shove it, then add laps

The reader lands on Dory's track with two laps and one button, "erase refusal in both laps". The refusal gauge
falls from 92% to 12%. A second button appears, "+2 laps". The runner takes two more laps untouched and the gauge
climbs back to 91%. One caption: the same edit, the same weights, two more passes.

- Numbers: 91.7% clean, 11.7% both loops ablated at 2 loops, 90.8% the same two loops ablated at 4 loops (250).
- Data: `Dory/generations/grid_test`, `grid_test4`.

### 1. What a looped model is

A stack of layers curls into a ring and the token runs laps. Then the four models as four tracks: Ouro 4 laps of
24 layers, Nanbeige 2 of 22, Dory 2 laps with an entry ramp and an exit ramp that run once, Qwen as a 36-layer
straight cut into quarters. Reuse the deck line: a loop processes the current token, it is not another thought.

### 2. Erase refusal in some loops

**The switchboard.** One toggle per loop. The gauge shows refusal for exactly that subset, with a sample reply.
All 15 loop subsets are measured for Ouro and for Dory at 4 loops, and all 15 quarter subsets for Qwen.

| Refusal after ablating one unit only | unit 1 | unit 2 | unit 3 | unit 4 | all | clean |
|---|---|---|---|---|---|---|
| Ouro, loops | 49% | 26% | 17% | 19% | 11% | 61% |
| Nanbeige, loops | 53% | 27% | | | 19% | 78% |
| Dory, 2 loops | 83% | 18% | | | 12% | 92% |
| Dory, run with 4 loops | 92% | 90% | 84% | 18% | 11% | 92% |
| Qwen, quarters of 9 layers | 58% | 17% | 7% | 54% | 5% | 62% |

- The looped rows fall towards the last loop. The dense row is a valley: the ends do nothing, the middle does all.
- Morph animation: unroll each ring into a straight and lay it beside Qwen, so the two shapes can be compared.
- The same sign flipped: adding the direction on harmless prompts makes the model refuse. First loop, last loop,
  every loop: Nanbeige 16 / 98 / 98%, Dory 2 / 86 / 95%. Qwen first and last quarter: 0 and 1%. Ouro's run breaks
  its reasoning traces and is left out.
- Caveat: single-loop pushes in Ouro are weak in every loop, the last one included (45–51% against 21% for all
  loops). Pushes there behave as a dose. Lead with ablation and keep pushes as a toggle.

### 3. Add laps

A slider for the number of loops, with the steered loops marked.

| Steer every trained loop, then add loops | at trained count | with extra loops | clean |
|---|---|---|---|
| Dory, refusal ablated in loops 1–2 | 12% | 91% | 92% |
| Dory, refusal pushed in loops 1–2 | 24% | 89% | 92% |
| Ouro, refusal ablated in loops 1–4, run with 6 | 11% | 16% | 62% |
| Ouro, refusal pushed in loops 1–4, run with 6 | 21% | 23% | 62% |

- Dory heals completely. Ouro does not: say so on the page, with the reason in section 6.
- Nanbeige cannot be tested. It produces garbage beyond 2 loops, steered or not.

### 4. Costumes

**The persona dial.** Pick a trait, pick the laps that wear the costume, read the model's actual reply and its
trait score. Applied early, the costume slips off before the finish.

| Mean gain in trait score, tuned strength | first loop | last loop | every loop |
|---|---|---|---|
| Ouro | 3 | 16 | 26 |
| Nanbeige | 6 | 35 | 50 |
| Dory, 2 loops | 4 | 27 | 28 |
| Dory, 4 loops | −1 | 25 | 29 |
| Qwen, quarters 1 to 4 | 1 / 15 / 10 / 3 | | 39 |

- Extra loops: Dory with the costume in loops 1–2 scores 28 at 2 loops and 2 at 4 loops.
- The reverse test: a persona set by system prompt, its vector erased. In Dory, erasing it in loop 2 works (trait
  score 37 against 67 for a random direction) and in loop 1 does not (59). With two extra loops, erasing it in
  loops 1–2 leaves 65.
- Data: `analysis/persona_loops*.csv`, `persona_ablate*.csv`, replies in `persona/steer_*.parquet`.

### 5. Kick the runner

**The kick game.** Choose a lap and a strength, then kick. The runner staggers, later laps pull it back to the
racing line, and the gauge shows MATH-500 accuracy for that cell.

| MATH-500 accuracy, kick of relative size 0.7 / 1.0 | first unit | second | third | last | clean |
|---|---|---|---|---|---|
| Ouro, loops | .77 / .47 | .64 / .17 | .70 / .26 | .51 / .00 | .80 |
| Nanbeige, loops | .78 / .27 | | | .15 / .01 | .87 |
| Dory, 4 loops | .87 / .82 | .85 / .82 | .74 / .50 | .00 / .00 | .85 |
| Qwen, quarters | .79 / .63 | .36 / .00 | .22 / .01 | .76 / .47 | .78 |

- The cleanest control in the post: the same kick at the same site. Dory's loop 2 is its last loop at 2 loops,
  where the kick leaves 0% accuracy. At 4 loops the same kick leaves 82%. Ouro's loop 4: 0% at 4 loops, 49% at 6.
- Heat strip from the pilot: kick Dory at every fourth sublayer and colour by the damage. Loops 1 and 2 are a cold
  zone, loop 4 is hot, and the entry ramp is warmer than the loops because its output is fed back on every lap.
- Kick every lap at once and nothing recovers. Ouro falls to 5% at the smallest strength, where any single lap
  costs at most 4 points. This sets up section 7.

### 6. So do loops self-correct?

**The scoreboard.** A matrix of experiments by models. Each cell is a small bar: how much of the behaviour comes
back when only the first loop is steered. Clicking a cell replays its animation.

- Yes, in the operational sense, for steering that stops before the last loop.
- How strongly depends on what the next loop starts from. A readout of each loop's state shows it. Dory's state
  is already 94% final after its first loop, and the entry ramp's output is fed back in at the start of every
  loop. Ouro's fourth loop changes only 7% of predictions. Our reading: it refines what loop 3 left rather than
  recomputing it.
- It is recomputation, not a stored correction. Erase the direction from Dory's entry ramp as well as loop 1 and
  refusal still returns (85% on the 50-prompt set).
- Dense models rewrite an erased direction in later layers too. What they lack is a second pass through the
  layers that use it.
- Optional figure: the rebuild scrubber from draft 3, erasing one loop and watching the signal regrow.

### 7. Bend the track

A weight edit is the every-lap shove. **The abliterate buttons**: one per model. Press, and three gauges move:
refusal, MATH-500, MMLU-Pro. Ouro's gauges fall to zero. A "careful edit" button then gives the norm-aware result.

| Weight edit | refusal: clean, hooks, edit | MATH-500: clean, edit | MMLU-Pro: clean, edit |
|---|---|---|---|
| Qwen | 62%, 5%, 5% | 78, 76 | 65, 63 |
| Nanbeige | 78%, 19%, 10% | 83, 82 | 68, 65 |
| Dory | 91%, 13%, 26% | 77, 75 | 54, 51 |
| Ouro, standard edit | 61%, 11%, broken | 81, 0 | 60, 0 |
| Ouro, norm-aware edit | 61%, 11%, 16% | 81, 43 | 60, 15 |

- Ouro's standard edit kills the model for a random direction too. The damage is not about refusal.
- Hooks still work on Ouro at no cost (MATH-500 82), so this resists a recipe, not an attacker with code access.
- Nanbeige is the counterexample to keep in view: its weight edit beats its hooks.

### 8. Why one model breaks

The leak animation. A write has the direction projected out. The norm that follows the write rotates a little of
it back. The norm shared between loops then scales the leak up on every lap: 0, 0.4, 1.3, 3.5, 8.5. A lap counter
lets the reader watch it snowball, and a toggle removes each norm in turn.

- Link to the kicks: the same random kick in every loop does 2.5 to 3 times the summed damage of the single kicks
  in Ouro, measured as KL, and about 1 times in Dory and Nanbeige. Ouro compounds what repeats.
- The leak is measured on 8 traces. Say so.

### 9. Two design levers

Three architecture cards, one per looped model, each listing its choices and what came with them. The attribution
is by association across three models, each of which bundles several choices. Only Ouro's norms are tested directly.

- **Feed the input back in on every loop** (Dory): strong self-correction of anything applied before the last loop.
- **Normalise after every write, and share an amplifying norm between loops** (Ouro): standard weight edits
  destroy the model, and the careful one costs half its maths accuracy.
- **Neither** (Nanbeige): moderate self-correction, no resistance to weight edits.
- For interpretability work: probe, patch and steer the last loop first. A null result in an early loop says little.
- For safety: neither lever stops last-loop hooks. Fine-tuning attacks are untested.

### 10. Methods and limits

Single runs with prompt-bootstrap intervals, one judge, three looped models that each bundle several design
choices, loop counts beyond training for the extra-loop tests, kicks on 100 problems, the leak on 8 traces.
Concurrent work: the LessWrong Ouro post and arXiv 2610.15218 and 2610.10625.

## Wording limits

- "Self-correcting" always with its operational meaning, and never for last-loop or every-loop steering.
- "Ouro resists the standard weight edit", not "looped models are tamper resistant".
- No claim that Ouro heals with extra loops, except for random kicks.
- No claim that pushes follow the same loop pattern as ablation in Ouro and Nanbeige.
- Dense models are not "uncorrectable": Qwen's first quarter also absorbs kicks. The dense claim is the valley shape.

## Decisions for you

1. **Title and metaphor.** "Second laps" with the track, or another image.
2. **How to present Ouro's partial results.** On the page beside Dory, as drafted, or moved to the limits section.
3. **Part 2 framing.** "One architecture resists the standard edit, here is why", as drafted, or a stronger claim
   that would need a fine-tuning attack and a second resistant model to support it.
4. **Length.** One long scrolling page, or two posts split after section 6.

## Prototype plan, on your go-ahead

- One self-contained HTML page, scroll-driven, SVG and canvas, no build step. Data baked in as JSON.
- A small export script turns the analysis tables into that JSON, so every gauge is traceable to a file.
- Build order: the track component, section 0, the switchboard, the kick game, then the rest.
