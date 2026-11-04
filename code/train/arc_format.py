"""NVARC-compatible ARC formatting + augmentation (no torch dependency, unit-testable).

Matches NVARC's QwenFormatter exactly:
  user turn      "<|im_start|>user\n{grid}<|im_end|><|im_start|>assistant\n"
  assistant turn "{grid}<|im_end|>"
grid = rows of digits joined by "\n". Loss is computed only on assistant turns (the outputs).
"""
import random

import numpy as np

USER_PRE = "<|im_start|>user\n"
USER_POST = "<|im_end|><|im_start|>assistant\n"
ASSIST_POST = "<|im_end|>"


def grid_str(g):
    return "\n".join("".join(str(int(c)) for c in row) for row in g)


def augment(pairs, rng):
    """Same family as NVARC: random D8 transform + full 10-colour permutation + example shuffle."""
    k, flip = rng.randrange(4), rng.random() < 0.5
    perm = list(range(10))
    rng.shuffle(perm)
    perm = np.array(perm)

    def tf(g):
        a = np.rot90(np.asarray(g), k)
        if flip:
            a = a.T
        return perm[a].tolist()

    out = [{"input": tf(p["input"]), "output": tf(p["output"])} for p in pairs]
    rng.shuffle(out)
    return out


def encode_pairs(pairs, tok, max_len):
    """Token ids + labels for a list of (input, output) pairs; the last pair is the 'challenge'.
    Leading demos are dropped until it fits max_len (keeps >= 2 pairs). Returns None if impossible."""
    pairs = list(pairs)
    while True:
        ids, labels = [], []
        for p in pairs:
            u = tok.encode(USER_PRE + grid_str(p["input"]) + USER_POST, add_special_tokens=False)
            a = tok.encode(grid_str(p["output"]) + ASSIST_POST, add_special_tokens=False)
            ids += u + a
            labels += [-100] * len(u) + a
        if len(ids) <= max_len:
            return ids, labels
        if len(pairs) <= 2:
            return None
        pairs = pairs[1:]


def task_pairs(task, solution=None):
    """All (input, output) pairs of a task: demos + test pairs with known outputs."""
    pairs = [{"input": p["input"], "output": p["output"]} for p in task["train"]]
    for i, t in enumerate(task.get("test", [])):
        out = t.get("output")
        if out is None and solution is not None and i < len(solution):
            out = solution[i]
        if out is not None:
            pairs.append({"input": t["input"], "output": out})
    return pairs


class AugmentedSet:
    """Map-style dataset: index -> (task, augmentation) deterministically, so resume is exact."""

    def __init__(self, tasks, tok, max_len, n_aug, seed):
        self.tasks, self.tok, self.max_len, self.n_aug, self.seed = tasks, tok, max_len, n_aug, seed

    def __len__(self):
        return len(self.tasks) * self.n_aug

    @property
    def length(self):                      # HF Trainer group_by_length reads dataset[length_column_name]
        if not hasattr(self, "_len_cache"):
            cells = [sum(len(p["input"]) * (len(p["input"][0]) + 1) + len(p["output"]) * (len(p["output"][0]) + 1) for p in t)
                     for t in self.tasks]
            self._len_cache = [min(cells[i % len(self.tasks)], self.max_len) for i in range(len(self))]
        return self._len_cache

    def __getitem__(self, i):
        if i == "length":
            return self.length
        if not 0 <= i < len(self):
            raise IndexError(i)                  # ends `for x in dataset` iteration (no endless wrap-around)
        return self._get(i)

    def _get(self, i):
        t = self.tasks[i % len(self.tasks)]
        for attempt in range(4):
            rng = random.Random(self.seed * 1_000_003 + i * 7 + attempt)
            enc = encode_pairs(augment(t, rng), self.tok, self.max_len)
            if enc is not None:
                ids, labels = enc
                return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}
        # unreachable for tasks that passed the length filter; return a minimal valid sample
        ids, labels = encode_pairs(t[-2:], self.tok, 10 ** 9)
        return {"input_ids": ids[: self.max_len], "labels": labels[: self.max_len], "attention_mask": [1] * min(len(ids), self.max_len)}
