# Synthetic-data pilot kit (ARC Prize 2026)

Goal of the pilot: measure, for about $10–15, whether LLM-written puzzle generators give us verified,
non-duplicate ARC puzzles aimed at the rule families NVARC misses — and at what cost per puzzle, per
provider. The numbers decide the training go/no-go (~6 Oct).

## What's in here

| File | Role |
|---|---|
| `seeds.json` | 40 seed concepts: 25 **target** families (what NVARC misses on eval: object relations, multi-step rules, 6+ colours, layering/counting/containment) + 15 **broad** families for coverage. Family-level text, no copied tasks. |
| `prompts.py` | AUTHOR prompt (rule + `generate_input()` + `transform()`) and REIMPLEMENT prompt (transform again from the rule text only). Sectioned output format, parser. |
| `providers.json` | OpenAI-compatible endpoints: `openai`, `deepseek`, `openrouter_gptoss` (gpt-oss-120b, NVARC's model), `gemini`, `mock`. Edit `model` names; keys come from env vars only. |
| `generate.py` | Runs author + 2 re-implementation calls per candidate, smoke-tests the generator before spending more, logs tokens and $ per call. Resumable; `--budget-usd` stop. |
| `sandbox.py` | Runs model-written code in a separate process with CPU/memory limits and a static ban on os/sys/subprocess/network/open. |
| `verify.py` | Validity, determinism, **self-consistency across 3 implementations**, non-triviality, D8 overlap check vs the 120 public-eval tasks (hard reject) and 1000 training tasks (flag), rule-text + example-set dedup across providers. |
| `convert.py` | Verified rules → Kaggle-format `challenges.json` / `solutions.json` + `meta.json` (provider, model, seed, rule) and per-provider subsets. |
| `report.py` | Per-provider table: pass rate, $ per verified rule, $ per unique rule, marginal uniqueness, category coverage. |
| `nemotron_overlap.py` | How much of the free Nemotron-RL-ARC-AGI-v1 (10k puzzles) is new vs public ARC and vs ours. Needs HF access. |

Tested end-to-end with the `mock` provider (`runs/mocktest/`).

## Why this design

* **Programs, not grids.** The LLM writes a generator per rule; we run it. Grids written by hand have
  cell errors; programs give unlimited exact examples. (Same as NVARC and Re-ARC.)
* **Self-consistency.** NVARC sampled 20 output programs per puzzle and kept the consistent ones. We do
  1 author + 2 independent re-implementations from the rule text alone; a rule that two blind
  implementers reproduce identically is unambiguous enough for a solver to infer. Cost ≈ 3 calls/rule.
* **Provenance tags.** Every puzzle carries its provider so any source can be dropped later (provider
  terms, licence for the open-source release, or because it adds nothing).
* **Eval firewall.** Any grid matching a public-eval grid under rotation/flip rejects the whole rule.
  Our local eval is already contaminated for NVARC; we must not add to it.

## Running the pilot

```bash
# on the RunPod pod or a laptop (no GPU needed for data generation)
cd arc2026/sdg
pip install requests numpy

# keys: environment only, never in chat or files
export OPENAI_API_KEY=...        # $10 credits
export GEMINI_API_KEY=...        # AI Studio key (free tier ok for pilot)
export DEEPSEEK_API_KEY=...      # ~$5 top-up
export OPENROUTER_API_KEY=...    # ~$5 top-up, serves gpt-oss-120b

# 1. check model names your keys can use, edit providers.json if needed
python generate.py --list-models openai
python generate.py --list-models gemini

# 2. same 20 target seeds x 4 providers x 1 rep = 80 candidates, ~240 calls, est. $8-12
python generate.py --providers deepseek openrouter_gptoss openai gemini \
                   --seeds target --reps 1 --out runs/pilot1 --budget-usd 15 --workers 4

# 3. verify, report, convert
python verify.py  --run runs/pilot1
python report.py  --run runs/pilot1
python convert.py --run runs/pilot1
```

Gemini free tier is rate-limited (the client backs off on 429, but a 4-provider run may stall on it);
if so run Gemini separately with `--workers 1`.

## Go / no-go thresholds (decide ~6 Oct)

| Measure | Go if | Why |
|---|---|---|
| Pass rate (verified ÷ candidates) | ≥ 30 % for at least one cheap provider | Below that, cost per puzzle and time blow up |
| $ per unique verified rule | ≤ $0.10 (DeepSeek / gpt-oss) | 20k rules for the full run ≈ ≤ $2k ceiling; realistically $300–1000 |
| Target-family coverage | ≥ 15 of 25 target categories produce verified rules | Otherwise the data doesn't hit NVARC's gaps |
| Puzzle statistics | mean input colours ≥ 5, mean input cells ≥ 300, size-varies ≥ 25 % | NVARC misses: 6.3 colours mean, 482 cells median |
| Marginal uniqueness | each kept provider ≥ 40 % rules with no near-neighbour elsewhere | A source that only repeats others isn't worth its terms risk |

Then the **small training test** (~2k rules → ~30–60k samples, short LoRA on NVARC's checkpoint, ~$30):
loss on a held-out slice of target-family puzzles must fall without eval regression. That is the real
go/no-go for the full run (start by ~13 Oct, finish by ~20 Oct).

## Scale-up notes

* Add `--reps 3-5` and `--difficulty mixed`; then write **second-generation seeds** from the verified rules
  (NVARC's "mix" step: combine two rule descriptions into one).
* Convert output is standard ARC JSON; NVARC's dataset builder (`SDG/scripts/build_datasets.py`) and
  augmentation (D8 + colour permutation + example reorder) take it from there. Mix with replay data
  (NVARC augmented puzzles / Re-ARC) so the checkpoint does not forget.
* Keep `meta.json` with the release: it is the provenance record the open-source requirement needs.

## Known limits

* `sandbox.py` is a crash/timeout guard, not a security boundary.
* Rule-text dedup is lexical (3-gram Jaccard ≥ 0.5). Two differently worded identical rules slip through;
  the example-set hash catches only exact example matches. Good enough for the pilot.
* Provider model names in `providers.json` are best guesses as of 3 Oct 2026 — check with `--list-models`.
* Prices in `providers.json` are estimates for spend tracking, not billing truth.
