# Let the Solver Choose Its Data

**Likelihood-band selection of synthetic ARC-AGI-2 tasks, and why held-out loss is the wrong signal**
Harthik Mallichetty — ARC Prize 2026 Paper Track (ARC-AGI-2)

- **Paper (PDF):** [`paper/arc2026_let_the_solver_choose_its_data.pdf`](paper/arc2026_let_the_solver_choose_its_data.pdf)
- **Kaggle writeup (≤1,500 words):** [`kaggle_writeup.md`](kaggle_writeup.md)
- **Kaggle model:** `harthikrm/nvarc-mega-r1` · **Base model:** `sorokin/qwen3_4b_grids15_sft139` (NVARC, ARC Prize 2025 winner) · **Pool:** `mindware/arc-agi-mega`

## In one paragraph

Score every candidate synthetic task with the target solver's own zero-shot teacher-forced NLL of the correct answer (one forward pass, no decoding). Keep the tasks whose per-token NLL lies inside the public evaluation set's 25th–95th percentile band (0.0324–0.3265 nats/token) and that the solver does not already solve greedily; replay the ones it does. Continue-training NVARC's checkpoint on 6,000 such tasks (2 GPU-hours) moved its ARC Prize 2026 hidden-test score from 28.5 (3 draws) to 31.2 (2 draws: 32.22, 30.14) with the competition pipeline unchanged. Scaling to 47,435 tasks cut the held-out loss on selected tasks by 39% but did not improve the hidden score (29.58); a forgetting monitor on the public evaluation set rose instead and is the better model-selection signal.

## Repository layout

| Path | What it is |
|---|---|
| `paper/` | LaTeX source (`main.tex`, `refs.bib`), figures (PDF), 640×360 gallery PNGs, cover image, compiled PDF |
| `code/sdg/mega_probe.py` | Samples ARC-Mega shards from Hugging Face; D8+colour dedup; evaluation firewall; target-profile filter → `challenges.json`/`solutions.json` |
| `code/kaggle/mega_score_worker.py`, `build_score_nb.py`, `nvarc-mega-score.ipynb` | Zero-shot scoring on Kaggle 4×L4: teacher-forced NLL + greedy-exact flag → `mega_scores.csv` |
| `code/sdg/build_train_data.py` | Band rule → `train.json` / `val.json` (fixed 300) / `replay.json` |
| `code/train/train_lora.py`, `arc_format.py` | rsLoRA continuation of the NVARC checkpoint, NVARC prompt format and augmentation, forgetting monitor, resumable across providers, `--merge` |
| `code/colab/` | Colab notebook builder used for run r2 |
| `code/kaggle/nvarc-v8-origorder-skip.ipynb` | The competition notebook: public NVARC code + deadline-safe scaffold + model glob (`NVARC_MODEL_DIR`) |
| `code/selection/selection_lab.py` | Offline re-ranking study of NVARC candidates (Section 7 of the paper) |
| `code/induction/` | Test-time program-induction harness and verifier (negative result, Section 7) |
| `code/analysis/` | Scripts that compute every statistic and figure in the paper from `mega_scores.csv` |
| `code/infra/`, `code/training/`, `code/arc_eval.py`, `code/kaggle_scaffold.py` | Local scorer, dev/lockbox split, provider setup notes, SFT prep for the induction branch |

## Reproducing the pipeline

```bash
# 1. probe the pool (needs the ARC-AGI-2 JSON files in ../data)
cd code/sdg && python mega_probe.py --max-gb 15 --out mega_big --exclude mega_sample/challenges.json

# 2. score zero-shot on Kaggle: upload mega_big/ as a dataset, run nvarc-mega-score.ipynb → mega_scores.csv

# 3. band selection
python build_train_data.py --mega mega_sample --mega mega_big --arc ../data --out train_data \
       --eval-quantile 0.25 --eval-upper-quantile 0.95 --max-train 6000 --n-val 300

# 4. continuation (r1 settings)
cd ../train && python train_lora.py --model-dir MODEL --out runs/r1 --train ../sdg/train_data/train.json \
       --replay ../sdg/train_data/replay.json --val ../sdg/train_data/val.json \
       --arc-eval ../data/arc-agi_evaluation_challenges.json --arc-eval-solutions ../data/arc-agi_evaluation_solutions.json \
       --rank 128 --alpha 32 --lr 5e-5 --global-batch 32 --max-len 8192 --n-aug 1 --replay-frac 0.3 --eval-steps 200
python train_lora.py --merge --model-dir MODEL --out runs/r1      # → runs/r1/merged, upload as a Kaggle model

# 5. point the competition notebook's model glob at the merged model and submit
```

## Licence

Code in this repository: MIT-0 (see `LICENSE`). The paper text and figures: CC BY 4.0. Third-party components keep their own licences (NVARC code and checkpoint, Qwen3, ARC-Mega, ARC-AGI-2 data).

## Citation

```bibtex
@misc{mallichetty2026solverchooses,
  title  = {Let the Solver Choose Its Data: Likelihood-Band Selection of Synthetic ARC-AGI-2 Tasks, and Why Held-Out Loss Is the Wrong Signal},
  author = {Mallichetty, Harthik},
  year   = {2026},
  howpublished = {ARC Prize 2026 Paper Track, \url{https://github.com/harthikrm/arc-prize-2026-paper}}
}
```
