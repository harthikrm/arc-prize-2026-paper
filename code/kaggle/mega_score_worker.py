"""Score NVARC's checkpoint (no test-time training) on a list of ARC test outputs.

One process per GPU. For each item: teacher-forced pass over (demos + test input + test output);
records the answer NLL and whether greedy decoding would reproduce the answer exactly
(= every answer token is the argmax given the correct prefix).

usage: python mega_score_worker.py RANK WORLD ITEMS_JSON OUT_JSONL END_TIME
"""
import json
import os
import sys
import time

rank, world = int(sys.argv[1]), int(sys.argv[2])
items_path, out_path, end_time = sys.argv[3], sys.argv[4], float(sys.argv[5])
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
sys.path[:] = [p for p in sys.path if "pip_install_unsloth" not in p]   # never import the utility-script torch

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

sys.path.insert(0, os.getcwd())
from arc_loader import QwenFormatter  # noqa: E402

MAXLEN = int(os.getenv("SCORE_MAXLEN", "8192"))
TOKEN_BUDGET = int(os.getenv("SCORE_TOKEN_BUDGET", "24576"))   # tokens per forward batch


def build(fmt, tok, it):
    """Token ids for prompt and answer; drop leading demos until it fits MAXLEN."""
    train = list(it["train"])
    while True:
        q = fmt.fmt_train(train) + fmt.fmt_query([{"input": it["test_input"]}])
        a = fmt.fmt_reply([it["test_output"]])
        qt, at = tok.encode(q), tok.encode(a)
        if len(qt) + len(at) <= MAXLEN:
            return qt, at, len(train)
        if len(train) <= 1:
            return None, None, 0
        train = train[1:]


def batches(prepped):
    """Group length-sorted items so that batch_size * max_len <= TOKEN_BUDGET."""
    prepped = sorted(prepped, key=lambda p: len(p[1]) + len(p[2]))
    cur, cur_max = [], 0
    for p in prepped:
        n = len(p[1]) + len(p[2])
        if cur and max(cur_max, n) * (len(cur) + 1) > TOKEN_BUDGET:
            yield cur
            cur, cur_max = [], 0
        cur.append(p)
        cur_max = max(cur_max, n)
    if cur:
        yield cur


@torch.no_grad()
def score_batch(model, batch, pad_id):
    seqs = [qt + at for _, qt, at, _ in batch]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long, device="cuda")
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    hidden = model.model(input_ids=ids).last_hidden_state          # right padding + causal mask: real tokens unaffected
    out = []
    for i, (it, qt, at, ntr) in enumerate(batch):
        q = len(qt)
        h = hidden[i, q - 1:q - 1 + len(at)]
        logp = model.lm_head(h).float().log_softmax(-1)
        tgt = torch.tensor(at, device=logp.device)
        tok_lp = logp[torch.arange(len(at), device=logp.device), tgt]
        correct = (logp.argmax(-1) == tgt)
        out.append({"key": it["key"], "source": it["source"], "n_ans_tokens": len(at), "n_train_used": ntr,
                    "nll_sum": float(-tok_lp.sum()), "nll_mean": float(-tok_lp.mean()),
                    "greedy_exact": bool(correct.all()), "frac_tokens_correct": float(correct.float().mean()),
                    "seq_prob": float(torch.exp(tok_lp.sum()))})
    return out


def main():
    md = os.environ["NVARC_MODEL_DIR"]
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(md, torch_dtype=torch.bfloat16, local_files_only=True,
                                                 attn_implementation="sdpa").cuda().eval()
    pad_id = int(os.getenv("PAD_ID", "13"))
    fmt = QwenFormatter(tokenizer=tok)
    items = json.load(open(items_path))[rank::world]
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["key"] for l in open(out_path) if l.strip()}
    prepped, skipped = [], 0
    for it in items:
        if it["key"] in done:
            continue
        qt, at, ntr = build(fmt, tok, it)
        if qt is None:
            skipped += 1
            continue
        prepped.append((it, qt, at, ntr))
    print(f"[rank {rank}] {len(prepped)} to score, {len(done)} already done, {skipped} too long", flush=True)
    t0, n = time.time(), 0
    with open(out_path, "a") as f:
        for b in batches(prepped):
            if time.time() > end_time:
                print(f"[rank {rank}] deadline reached after {n} items", flush=True)
                break
            try:
                res = score_batch(model, b, pad_id)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                res = []
                for single in b:                      # retry one at a time
                    try:
                        res += score_batch(model, [single], pad_id)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
            n += len(b)
            if n % 500 < len(b):
                print(f"[rank {rank}] {n}/{len(prepped)} items, {n / (time.time() - t0):.2f} items/s", flush=True)
    print(f"[rank {rank}] finished {n} items in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
