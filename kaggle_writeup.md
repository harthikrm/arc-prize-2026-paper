# Let the Solver Choose Its Data

**Subtitle:** Likelihood-band selection of synthetic ARC-AGI-2 tasks from ARC-Mega: +2.7 on the hidden test from 6k tasks and 2 GPU-hours, and why held-out loss is the wrong signal

**Track:** ARC-AGI-2 · **Linked submission:** `[SUBMISSION ID]` (notebook `[URL]`, model `harthikrm/nvarc-mega-r1`, public LB 32.22 / 30.14 on two draws)

## Summary

Large synthetic task pools now exist for ARC, but most of their contents are either already solved by a strong solver or unlearnable by it. I propose **solver-in-the-loop curation**: score every candidate task with the target solver's own zero-shot likelihood of the correct answer (one teacher-forced forward pass, no decoding, no labels beyond the task), and keep only tasks whose per-token NLL falls inside the band spanned by the target benchmark's 25th–95th percentiles and that the solver does not already solve greedily. Applied to ARC-Mega (MindsAI/Tufa Labs' 97M-row pool, ~74M ARC tasks), this scores 114,067 tasks in about 8.4 hours on Kaggle's 4×L4 and keeps 42%.

Continue-training the 2025 winner, NVARC, on only **6,000** selected tasks (2.05 h on one H200) raised its hidden-test score from a mean of **28.5** (three draws of the stock checkpoint) to **31.2** (two draws: 32.22, 30.14), with the competition-time pipeline unchanged. Scaling the same recipe to **47,435** tasks lowered the held-out loss on selected tasks by 39% but **did not** improve the hidden score (29.58), while a forgetting monitor on the public evaluation set turned upward in the second half of training. Everything is released.

## Method (algorithm level)

1. **Probe the pool.** Sample 39 of 190 ARC-Mega training shards; deduplicate under the eight dihedral symmetries plus colour relabelling; reject any task sharing a grid with the 120 public evaluation tasks; keep tasks matching NVARC's miss profile (≥5 colours, ≥300 cells, ≥3 objects in the first demonstration input).
2. **Score zero-shot.** Serialise demonstrations + test input + test output in NVARC's 16-token dialogue format (dropping leading demos down to one if over 8,192 tokens; skip otherwise); one forward pass with the frozen checkpoint gives the answer's per-token NLL and a `greedy_exact` flag (every answer token is the argmax). Throughput: 13.6k tasks/hour on 4×L4.
3. **Band rule.** Score the public evaluation set the same way; take its 25th and 95th NLL percentiles (τ25 = 0.0324, τ95 = 0.3265 nats/token). Select a task if it is not greedy-solved and τ25 ≤ NLL ≤ τ95. Greedy-solved tasks plus the 1,000 public training tasks become **replay** (30% of training samples). Hold out a fixed 300 selected tasks as in-distribution validation.
4. **Short LoRA continuation.** rsLoRA r=128, α=32 on all attention/MLP projections; lr 5e-5 cosine, 3% warm-up; global batch 32; bf16; max length 8,192; loss on output grids only; NVARC-style augmentation (D8 + colour permutation + demo shuffle). Log two losses every 200 steps: held-out loss on the 300 selected tasks, and the **forgetting monitor** — loss on the public evaluation set (the base checkpoint was trained on it, so only a *rise* is informative).
5. **Merge and submit.** Merge the adapter into the base weights; run the public NVARC notebook unchanged (per-task LoRA TTT → batched DFS decoding → augmentation re-scoring → pass@2) with a deadline-safe scaffold (placeholder submission in the first minute; TTT halts at 11.75 h).

## What the solver sees in ARC-Mega

- The pool's 166-task "test" split **is** the public ARC-AGI-2 evaluation set (100% grid overlap); it is not a clean held-out set. Training rows overlap the evaluation set at 0.2%; only 77% of training rows are distinct under augmentation.
- NVARC solves 14.5% of profile-matched Mega tasks greedily (vs. 18% of public training tasks and 1.2% of evaluation outputs). The Mega median NLL (0.036) is half the evaluation median (0.067) and the distribution is much wider.
- Partition of the 114,067 scored tasks: 14% replay (greedy-solved), 33% below the band, **42% selected**, 10% above the band. Within the band the greedy-exact rate is 1–2% while token accuracy is 92–99%: tasks the solver nearly handles but does not.
- The tails are populated differently: below-band tasks are long, copy-heavy outputs (83% same-shape, median 504 answer tokens); above-band tasks are very short outputs (median 30 tokens, 10% same-shape), so the upper cut is partly a length filter. The band matches the evaluation set on token accuracy (97% vs. 97%) and answer length better than either tail.

## Results on the hidden test set (public leaderboard, ≈120 tasks)

| Run | Selected tasks | Steps | GPU-h | Held-out Mega loss | Forgetting monitor | Hidden draws (mean) |
|---|---|---|---|---|---|---|
| baseline (stock NVARC) | — | — | — | 0.0734 | 0.0724 | 28.89, 28.47, 28.19 (28.5) |
| **r1** | 6,000 | ≈270 | 2.05 | 0.0734 → 0.0603 (−18%) | 0.0724 → 0.0692 | **32.22, 30.14 (31.2)** |
| r2 | 47,435 | 2,118 | 11.7 | 0.0734 → 0.0450 (−39%) | 0.0724 → 0.0707 → 0.0716 | 29.58 |

The three stock-checkpoint draws (successive versions of the same deadline scaffold) spread by 0.7 points; I treat ±1.5 as the noise level. The lowest r1 draw beats the highest baseline draw by 1.25 points; with 2 vs. 3 draws the +2.7 effect is suggestive rather than settled, and I report it as such. Local evaluation accuracy is not reported because the base checkpoint was trained on the public evaluation puzzles.

## Why it works, and why the obvious proxy fails

**Why a small selected set helps.** NVARC adapts to every test task with LoRA test-time training, so the base checkpoint's job is to supply a good *prior* over rule families from which a few hundred TTT steps can recover the specific rule. The band holds tasks that are learnable and not yet learnt (in the sense of RHO-LOSS); a short continuation sharpens the prior where it is weakest, and the forgetting monitor confirms the base knowledge is undisturbed (0.0724 → 0.0692).

**Why in-distribution held-out loss misleads.** Held-out loss on selected tasks measures fit to the *generator distribution* of the pool, which the band cannot fully remove. With 8× more steps at the same learning rate, r2 learns Mega-specific regularities that lower Mega loss without improving the prior TTT needs; the hidden score is flat and the forgetting monitor reverses direction. This is a Goodhart effect specific to solvers with test-time adaptation: task-specific likelihood is exactly what TTT will fix anyway, and exactly what the validation loss rewards. Across the three configurations the hidden-score ordering (r1 > r2 ≈ baseline) matches the forgetting-monitor ordering and is the *reverse* of the held-out-loss ordering.

**Recommendation.** Score the pool with the solver before training; train briefly on the band and replay what is solved; select checkpoints by a rise in the target benchmark's loss, not a fall in the pool's loss; expect single-digit-thousand tasks to be enough.

## Levers measured and rejected (same budget)

- **Answer re-ranking:** NVARC's candidates contain a correct grid for 54/172 evaluation outputs (oracle 38.5 task-points vs. 33.3 achieved), but the best of 14 alternative rules recovered only +0.5: in 8 of 9 recoverable misses the wrong grid wins on every signal.
- **CPU DSL search** (2020 winner, depth 3): +0.5 beyond NVARC in 55 CPU-minutes; fills empty slots only.
- **Test-time program induction** with open 30B-class models in NVIDIA's agentic tool format, with a verifier (demo pass, leave-one-out, literal-grid hygiene, consensus): best zero-shot model (Qwen3.6-35B-A3B, 57 dev tasks) passed all demos with 2/122 final programs and solved nothing NVARC missed; in a 5-task smoke test two other models truncated 98–100% of reasoning turns. Side finding: verifiers have false negatives — correct programs were rejected because the model's own test code embedded demo grids; verify only code reachable from the solution function.

## Limitations

Six hidden draws across three configurations; one base model, one seed, one learning rate (so "too much data" and "too many steps" are confounded); no random-6k control; the upper cut is partly a length filter; the target profile is a confound; public LB uses about half the hidden tasks.

## Universality

The rule needs only a solver that assigns likelihoods and a target benchmark to anchor the band; it is agnostic to how the pool was generated and transfers to any few-shot induction setting with abundant synthetic data (program synthesis, grid or sequence puzzles, tool-use traces). The lesson about validation signals applies to any model that adapts at test time.

## Resources

- Paper (PDF, full details and figures): https://github.com/harthikrm/arc-prize-2026-paper/blob/main/paper/arc2026_let_the_solver_choose_its_data.pdf
- Code: `mega_probe.py`, scoring notebook `nvarc-mega-score`, `build_train_data.py`, `train_lora.py` + `arc_format.py`, `selection_lab.py` — https://github.com/harthikrm/arc-prize-2026-paper
- Model: `harthikrm/nvarc-mega-r1` (Kaggle Models) · Base: `sorokin/qwen3_4b_grids15_sft139` · Pool: `mindware/arc-agi-mega`
