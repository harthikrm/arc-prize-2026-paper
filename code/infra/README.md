# ARC Prize 2026 — infra kit

Order of operations (laptop = your machine with `gcloud`; VM = the 4xL4 box):

| # | Where | Command | What it gives you |
|---|---|---|---|
| 0 | laptop | `PROJECT=<id> bash 00_gcp_quota_check.sh` | shows GPU quotas; request increases today |
| 1 | laptop | `PROJECT=<id> bash 01_create_l4_vm.sh` (`SPOT=1` for cheap) | g2-standard-48 = 4x L4, same as Kaggle scoring |
| - | laptop | `gcloud compute scp --zone Z --recurse ./infra ~/.kaggle/kaggle.json arc-l4x4:~/` | copy this kit + Kaggle token |
| 2 | VM | `bash ~/infra/02_vm_bootstrap.sh` | driver, docker, official Kaggle GPU image, competition data at /kaggle/input |
| 3 | VM | `python3 ~/infra/03_fetch_notebook.py <owner>/<slug>` | notebook + all attached models/datasets at Kaggle paths |
| 4 | VM | `bash ~/infra/04_rehearse.sh <slug> eval 20` then `... eval` | full offline rehearsal (no network, 12h clock), scored |

Other tools:
- `arc_eval.py` — Kaggle-exact scorer, failure taxonomy, `--compare` for complementarity.
- `kaggle_scaffold.py` — notebook skeleton: fallback-first submission, placeholder smoke mode, time budgets.

**Stop the VM when idle:** `gcloud compute instances stop arc-l4x4 --zone <zone>`
