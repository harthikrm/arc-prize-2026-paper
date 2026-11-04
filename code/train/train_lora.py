#!/usr/bin/env python3
"""Continue-train NVARC's Qwen3-4B (sorokin/qwen3_4b_grids15_sft139) with LoRA on new ARC puzzles.

Built to hop between providers (Lightning / Modal / RunPod / GCP / Kaggle) on credits:
  * full-state checkpoints (adapter + optimizer + scheduler + RNG + step) every --save-minutes
  * auto-resume from the newest local checkpoint, or from a private Hugging Face repo (--hub-repo)
  * global batch fixed by --global-batch; gradient accumulation adapts to the number of GPUs
  * deterministic data order (index -> task + augmentation), so a resumed run is the same run
  * loss only on output grids, NVARC prompt format, NVARC-style augmentation

Single GPU:   python train_lora.py --model-dir MODEL --train data/train.json --out runs/r1
Multi GPU:    torchrun --nproc_per_node 8 train_lora.py ... (same args)
Merge:        python train_lora.py --merge --model-dir MODEL --out runs/r1   -> runs/r1/merged (Kaggle-ready)

--train / --replay / --val are JSON files in Kaggle layout {id: {"train": [...], "test": [...]}} with the test
outputs inside, or pass the solutions file with --train-solutions etc.
"""
import argparse
import glob
import json
import os
import random
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arc_format import AugmentedSet, encode_pairs, task_pairs  # noqa: E402


def load_tasks(path, sol_path=None, keys_path=None):
    if not path:
        return []
    ch = json.load(open(path))
    so = json.load(open(sol_path)) if sol_path else {}
    keep = set(json.load(open(keys_path))) if keys_path else None
    out = []
    for k, t in ch.items():
        if keep is not None and k not in keep and f"{k}_0" not in keep:
            continue
        pairs = task_pairs(t, so.get(k))
        if len(pairs) >= 2:
            out.append(pairs)
    return out


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train"); ap.add_argument("--train-solutions"); ap.add_argument("--train-keys")
    ap.add_argument("--replay"); ap.add_argument("--replay-solutions")
    ap.add_argument("--replay-frac", type=float, default=0.3, help="share of samples drawn from replay tasks")
    ap.add_argument("--val"); ap.add_argument("--val-solutions")
    ap.add_argument("--val-frac", type=float, default=0.02, help="held-out share of --train if --val not given")
    ap.add_argument("--n-aug", type=int, default=8, help="augmented samples per task per epoch")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--global-batch", type=int, default=32)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5, help="NVARC TTT uses 5e-5; conservative keeps the checkpoint's knowledge")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--alpha", type=int, default=32, help="rsLoRA scale alpha/sqrt(r); NVARC uses 32 with r=256")
    ap.add_argument("--arc-eval"); ap.add_argument("--arc-eval-solutions", help="public ARC eval: forgetting monitor (contaminated for NVARC, so a RISE in its loss is the signal, not the level)")
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--save-minutes", type=float, default=30)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--hub-repo", default=None, help="private HF repo for checkpoint hand-off between providers")
    ap.add_argument("--max-hours", type=float, default=0, help="stop cleanly (with a save) after this many hours")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge", action="store_true", help="merge newest checkpoint into the base model and exit")
    ap.add_argument("--dry-run", action="store_true", help="build data, print stats, no training")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="skip gradient checkpointing (~1.3x faster; needs ~85 GB at --max-len 8192, so H100/H200 only)")
    return ap.parse_args()


def newest_ckpt(out):
    c = sorted(glob.glob(os.path.join(out, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[-1]))
    return c[-1] if c else None


def hub_pull(repo, out):
    if not repo or newest_ckpt(out):
        return
    from huggingface_hub import list_repo_files, snapshot_download
    try:
        files = list_repo_files(repo)
    except Exception as e:
        print(f"[hub] nothing to pull from {repo}: {e}")
        return
    steps = sorted({int(f.split("/")[0].rsplit("-", 1)[-1]) for f in files if f.startswith("checkpoint-")})
    if steps:
        snapshot_download(repo, local_dir=out, allow_patterns=[f"checkpoint-{steps[-1]}/*"])
        print(f"[hub] pulled checkpoint-{steps[-1]} from {repo}")


def hub_push(repo, ckpt):
    if not repo or not ckpt:
        return
    from huggingface_hub import HfApi, create_repo, upload_folder
    create_repo(repo, private=True, exist_ok=True)
    upload_folder(repo_id=repo, folder_path=ckpt, path_in_repo=os.path.basename(ckpt))
    print(f"[hub] pushed {os.path.basename(ckpt)} -> {repo}", flush=True)
    try:                                             # keep only the newest checkpoint on the hub (each is ~3 GB)
        api = HfApi()
        old = sorted({f.split("/")[0] for f in api.list_repo_files(repo) if f.startswith("checkpoint-")},
                     key=lambda d: int(d.rsplit("-", 1)[-1]))[:-1]
        for d in old:
            api.delete_folder(path_in_repo=d, repo_id=repo)
            print(f"[hub] removed old {d}", flush=True)
    except Exception as e:
        print(f"[hub] cleanup skipped: {e}", flush=True)


def main():
    a = parse()
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments,
                              DataCollatorForSeq2Seq)

    rank0 = int(os.environ.get("RANK", "0")) == 0
    world = int(os.environ.get("WORLD_SIZE", "1"))
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if a.merge:
        ck = newest_ckpt(a.out)
        assert ck, f"no checkpoint in {a.out}"
        base = AutoModelForCausalLM.from_pretrained(a.model_dir, torch_dtype=torch.bfloat16)
        merged = PeftModel.from_pretrained(base, ck).merge_and_unload()
        dst = os.path.join(a.out, "merged")
        merged.save_pretrained(dst, safe_serialization=True)
        tok.save_pretrained(dst)
        for f in os.listdir(a.model_dir):            # keep any extra files NVARC's loader expects
            if not os.path.exists(os.path.join(dst, f)) and os.path.isfile(os.path.join(a.model_dir, f)):
                shutil.copy(os.path.join(a.model_dir, f), dst)
        print(f"merged {ck} -> {dst}")
        return

    # ---------------- data
    train_tasks = load_tasks(a.train, a.train_solutions, a.train_keys)
    rnd = random.Random(a.seed)
    rnd.shuffle(train_tasks)
    if a.val:
        val_tasks = load_tasks(a.val, a.val_solutions)
    else:
        nv = max(1, int(len(train_tasks) * a.val_frac))
        val_tasks, train_tasks = train_tasks[:nv], train_tasks[nv:]
    replay_tasks = load_tasks(a.replay, a.replay_solutions)
    fits = lambda t: encode_pairs(t[-2:], tok, a.max_len) is not None
    train_tasks = [t for t in train_tasks if fits(t)]
    val_tasks = [t for t in val_tasks if fits(t)][:400]
    replay_tasks = [t for t in replay_tasks if fits(t)]
    if replay_tasks and a.replay_frac > 0:   # mix: replay share of the per-epoch sample count
        n_rep = int(len(train_tasks) * a.replay_frac / (1 - a.replay_frac))
        rr = random.Random(a.seed + 1)
        train_tasks = train_tasks + [replay_tasks[rr.randrange(len(replay_tasks))] for _ in range(n_rep)]
        random.Random(a.seed + 2).shuffle(train_tasks)
    train_ds = AugmentedSet(train_tasks, tok, a.max_len, n_aug=max(1, round(a.n_aug * a.epochs)), seed=a.seed)
    val_ds = {"mega_heldout": AugmentedSet(val_tasks, tok, a.max_len, n_aug=1, seed=a.seed + 99)}
    if a.arc_eval:
        arc_tasks = [t for t in load_tasks(a.arc_eval, a.arc_eval_solutions) if fits(t)]
        val_ds["arc_eval"] = AugmentedSet(arc_tasks, tok, a.max_len, n_aug=1, seed=a.seed + 7)
    grad_acc = max(1, a.global_batch // (world * a.per_device_batch))
    steps = len(train_ds) // (grad_acc * world * a.per_device_batch)
    if rank0:
        lens = [len(train_ds[i]["input_ids"]) for i in range(0, len(train_ds), max(1, len(train_ds) // 300))]
        print(f"tasks: train {len(train_tasks)} (incl. replay) | val {len(val_tasks)} | samples {len(train_ds)} | "
              f"world {world} | grad_acc {grad_acc} | optimizer steps {steps} | "
              f"mean tokens/sample {sum(lens) / len(lens):.0f} | ~{sum(lens) / len(lens) * len(train_ds) / 1e6:.0f}M tokens", flush=True)
    if a.dry_run:
        return

    # ---------------- model
    flag = os.path.join(a.out, ".hub_pull_done")
    if rank0:
        os.makedirs(a.out, exist_ok=True)
        hub_pull(a.hub_repo, a.out)
        open(flag, "w").write("ok")
    else:                                            # other ranks wait for rank 0's checkpoint download
        while not os.path.exists(flag):
            time.sleep(5)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if not a.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, bias="none", use_rslora=True, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    if rank0:
        model.print_trainable_parameters()

    t_start = time.time()

    class TimedSave(TrainerCallback):
        last = time.time()

        def on_step_end(self, args, state, control, **kw):
            if time.time() - self.last > a.save_minutes * 60 or (state.global_step % a.eval_steps == 0):
                control.should_save = True               # also save right before each eval, so an eval crash loses nothing
                self.last = time.time()
            if a.max_hours and time.time() - t_start > a.max_hours * 3600:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kw):
            if rank0:
                hub_push(a.hub_repo, newest_ckpt(a.out))
            return control

    targs = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.per_device_batch, per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_acc, num_train_epochs=1, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_ratio=a.warmup, weight_decay=0.0, max_grad_norm=1.0, bf16=True, logging_steps=10,
        save_strategy="no", save_total_limit=2, eval_strategy="steps", eval_steps=a.eval_steps,
        prediction_loss_only=True, eval_accumulation_steps=1, eval_on_start=True,
        report_to="none", seed=a.seed, data_seed=a.seed, dataloader_num_workers=2,
        remove_unused_columns=False, group_by_length=True, length_column_name="length", ddp_find_unused_parameters=False)
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8)

    class LengthTrainer(Trainer):
        """group_by_length with precomputed (cell-count) lengths: avoids tokenizing the whole set up front."""
        def _get_train_sampler(self, *args, **kw):
            from transformers.trainer_pt_utils import LengthGroupedSampler
            return LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps,
                                        dataset=self.train_dataset, lengths=self.train_dataset.length)

        def _get_eval_sampler(self, *args, **kw):
            return torch.utils.data.SequentialSampler(args[0] if args else kw["eval_dataset"])

    trainer = LengthTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                            data_collator=collator, callbacks=[TimedSave()])
    ck = newest_ckpt(a.out)
    if rank0:
        print(f"resume from: {ck or 'scratch'}", flush=True)
    trainer.train(resume_from_checkpoint=ck)
    trainer.save_model(os.path.join(a.out, "final_adapter"))
    try:
        trainer._save_checkpoint(model, trial=None)  # full state, so a later run can still extend it
    except TypeError:
        trainer._save_checkpoint(model, None, None)
    except Exception as e:
        print(f"final full-state save failed ({e}); final_adapter is saved, last timed checkpoint stands")
    if rank0:
        hub_push(a.hub_repo, newest_ckpt(a.out))
        print(json.dumps(trainer.state.log_history[-5:], indent=1))
        print(f"done in {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
