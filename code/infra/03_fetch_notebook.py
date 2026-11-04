#!/usr/bin/env python3
"""
03_fetch_notebook.py — run ON THE VM. Pulls any Kaggle notebook AND everything attached to it
(models, datasets, other notebooks' outputs) into the same /kaggle/input/... paths Kaggle uses,
so the notebook runs unmodified inside the Kaggle docker image.

    python3 ~/infra/03_fetch_notebook.py <owner>/<notebook-slug>
    e.g. the NVARC team's 2026 baseline:  (copy the slug from the notebook URL on kaggle.com)

Result:
    /kaggle/notebooks/<slug>/<slug>.ipynb + kernel-metadata.json
    /kaggle/input/<dataset-slug>/...                              (dataset_sources)
    /kaggle/input/<model>/<framework>/<variation>/<version>/...   (model_sources)
    /kaggle/input/<notebook-slug>/...                             (kernel_sources = outputs)
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
NB_ROOT = Path("/kaggle/notebooks")


def run(cmd):
    print("   $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def place(src, dst):
    """Copy (resolving symlinks) so docker bind-mounts see real files."""
    dst = Path(dst)
    if dst.exists() and any(dst.iterdir()):
        print(f"   already present: {dst}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True, symlinks=False)
    print(f"   -> {dst}")


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit(__doc__)
    ref = sys.argv[1].strip().rstrip("/")
    slug = ref.split("/")[-1]
    nb_dir = NB_ROOT / slug
    nb_dir.mkdir(parents=True, exist_ok=True)

    print(f">> pulling notebook {ref} (with metadata)")
    run(["kaggle", "kernels", "pull", ref, "-p", str(nb_dir), "-m"])
    meta = json.loads((nb_dir / "kernel-metadata.json").read_text())
    for k in ("competition_sources", "dataset_sources", "model_sources", "kernel_sources"):
        print(f"   {k}: {meta.get(k)}")
    if meta.get("enable_internet"):
        print("   NOTE: notebook has internet enabled; the scoring rerun has NO internet. "
              "The rehearsal runs with --network none and will expose any hidden downloads.")

    import kagglehub  # noqa: E402  (installed by 02_vm_bootstrap.sh)

    for ds in meta.get("dataset_sources") or []:
        print(f">> dataset {ds}")
        place(kagglehub.dataset_download(ds), INPUT / ds.split("/")[-1])

    for m in meta.get("model_sources") or []:
        # owner/model/framework/variation/version  ->  /kaggle/input/model/framework/variation/version
        print(f">> model {m}")
        parts = m.split("/")
        if len(parts) != 5:
            print(f"   unexpected model handle format, skipping: {m}")
            continue
        _, model, framework, variation, version = parts
        place(kagglehub.model_download(m), INPUT / model / framework / variation / version)

    for kr in meta.get("kernel_sources") or []:
        print(f">> notebook output {kr}")
        dst = INPUT / kr.split("/")[-1]
        if dst.exists() and any(dst.iterdir()):
            print(f"   already present: {dst}")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        try:
            place(kagglehub.notebook_output_download(kr), dst)
        except Exception as e:  # older kagglehub: fall back to CLI
            print(f"   kagglehub failed ({e}); using kaggle CLI")
            run(["kaggle", "kernels", "output", kr, "-p", str(dst)])

    comps = meta.get("competition_sources") or []
    for c in comps:
        if not (INPUT / c).exists():
            print(f"   WARNING: competition {c} not at {INPUT / c}; run 02_vm_bootstrap.sh or download it")

    nbs = list(nb_dir.glob("*.ipynb")) + list(nb_dir.glob("*.py"))
    print(f"\nDone. Notebook file(s): {[str(p) for p in nbs]}")
    print(f"Disk used under /kaggle/input: ", end="", flush=True)
    os.system("du -sh /kaggle/input | cut -f1")
    print(f"Next: bash ~/infra/04_rehearse.sh {slug} eval")


if __name__ == "__main__":
    main()
