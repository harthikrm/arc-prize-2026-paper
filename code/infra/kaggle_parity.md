# Kaggle parity checklist

Every number measured on a rented GPU is re-checked on the real target before we trust it.
Tonight's lesson (2026-10-01): DeepGEMM exists on H100 and not on L4; a kernel library crashed
Qwen at start-up. Fixed with `VLLM_USE_DEEP_GEMM=0`, which also matches what L4 can do.

| Area | Rented pod (RunPod H100) | Kaggle (4x L4, 96 GB) | How we cover the gap |
|---|---|---|---|
| GPU kernels | H100-only libs (DeepGEMM, FA3) available | L4 = Ada (sm_89): FP8 ok, Hopper libs absent | Run with `VLLM_USE_DEEP_GEMM=0`; never read speed off the H100 |
| Speed | 5-8x faster than 4xL4 | 12 h hard cap, timeout = score 0. **On the hidden 240 tasks NVARC alone uses the full 11.5 h; its 10 h vs 12 h runs scored 27.22 vs 29.72, ~1.25 pts/hour.** | Sweep measures tokens/task; `bench_speed.py` on 4xL4 converts to hours; solver phase runs first and is killed at a hard wall-clock; scaffold ends at 11h40m with incremental writes. Break-even: 1 NVARC-missed dev task per solver hour; demand 2 |
| Internet | Downloads work | None. Weights, vLLM wheel, all packages attached as Kaggle datasets | 4xL4 rehearsal runs with `HF_HUB_OFFLINE=1` and network disabled |
| Versions | vllm 0.30.0, torch 2.13+cu130 | Kaggle image v171 (1 Oct 2026): torch 2.11.0+cu128, Python 3.13, transformers 5.16.1. Pre-v171 (v170): torch 2.10.0+cu128, Python 3.12. **No participant has reported offline vLLM working on Kaggle.** | Pin the notebook environment to one image version; build/attach a vLLM wheel for exactly that torch; time a transformers-generate fallback on the 4xL4 pod |
| CPU | 16 vCPU | Few cores (check at rehearsal) | Sandbox/verifier parallelism is tuned on 4xL4, not here (`--parallel-tasks`, `--parallel-per-task`) |
| Start-up | Qwen3.6 FP8: 325 s. gpt-oss-20b: 240 s incl. download. Nemotron cold: 20+ min of Triton/NVCC compiles for the Mamba kernels on 16 cores (2026-10-01) | Kernel compile + CUDA graphs on L4, fewer cores; counts against 12 h | `summary.md` records *server start* per model; ship the compile cache (`~/.cache/vllm`, Triton cache) inside the model dataset or weigh the cold start in the model choice |
| Nemotron | `--trust-remote-code` + parser plugin download | Same files, must be local | Ship the repo snapshot and `nano_v3_reasoning_parser.py` inside the model dataset |
| Submission format | `submission.json` written by solver | `submission.json`, pass@2, exact match, must exist even on crash | Scaffold writes NVARC fallback first, overwrites with verified programs |

## Baseline on the real leaderboard

| Date | Submission | Public score | Notes |
|---|---|---|---|
| 2026-10-01 | NVARC 2025, unchanged (`v1-baseline-nvarc2025-unchanged`) | **31.53** | Succeeded within the 12 h limit on hidden tasks. Notebook run on public eval: 5h24m on 4xL4. This is the floor every later submission must beat |

## Architecture check (cold-start risk)

Hybrid models compile Triton kernels for their non-attention layers on first start; plain transformers do not.

| Model | Architecture (from config.json) | Cold-start risk |
|---|---|---|
| Nemotron-3-Nano-30B-A3B | Hybrid Mamba-2 + attention | Measured: 2,235 s cold on H100 NVL, 16 cores. **Benched 2026-10-01**: 0/8 on the smoke test, 98% truncated turns |
| Qwen3.6-35B-A3B | Hybrid: 30 of 40 layers linear attention (Gated DeltaNet), 1 in 4 full attention | Measured: 325 s on H100 NVL (weights already cached), DeepGEMM off |
| Qwen3-Coder-30B-A3B | Plain transformer MoE (qwen3_moe) | None expected |
| Qwen3-30B-A3B-Thinking-2507 | Plain transformer MoE (qwen3_moe) | Measured: 200 s. **Benched 2026-10-01**: 0/8, 100% truncated turns |
| gpt-oss-20b | Plain transformer MoE, sliding + full attention, MXFP4 | None expected |

## Order of operations (do the rehearsal early)

1. Sweep on H100 picks the model and thinking budget. (now)
2. 4xL4 RunPod pod, Kaggle-pinned vLLM, offline mode: `bench_speed.py`, then a full 80-task run
   with `04_rehearse.sh`. Fix anything that breaks. (week 1)
3. Submit the **zero-shot** pipeline to Kaggle before any fine-tuning. A modest score is fine;
   the point is that every integration bug surfaces with four weeks left, not one. (week 1)
4. Fine-tune. Training only swaps the weights inside a pipeline already proven on Kaggle. (weeks 2-4)
5. Every later submission: same scaffold, new weights dataset. Rehearse on 4xL4 before each.
