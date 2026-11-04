"""Prompt templates for synthetic ARC puzzle generation.

Two prompt types:
  * AUTHOR      : from a seed concept, write the rule + an input generator + a transform.
  * REIMPLEMENT : from the rule text and a few example inputs only (no code), write the transform again.
                  Agreement between independent implementations is the self-consistency check
                  (same idea as NVARC's "20 output programs, keep the consistent ones", at 1/7 the cost).

Output is a sectioned text format, not JSON, because code inside JSON strings breaks constantly.
"""

ARC_RULES = """\
ARC grid conventions:
- A grid is a list of rows; each row is a list of integers 0-9 (colours). 0 is the background.
- Height and width are each between 1 and 30.
- A puzzle shows 2-5 (input, output) demonstration pairs and one or two test inputs; a solver must infer the
  rule from the demonstrations alone, so the rule must be unambiguous after seeing 3 random examples.
- Good puzzles are about objects, shapes, counting, symmetry, paths, containment and relations - not arithmetic
  on digit values, not text, not randomness in the output.
"""

AUTHOR_SYSTEM = "You design ARC-AGI style abstraction puzzles and implement them as small, correct Python programs."

AUTHOR_USER = """\
{arc_rules}
Design ONE new puzzle in this family:

  Family: {category}
  Brief: {description}

Requirements
1. Invent a specific, concrete rule in this family. Add one twist so it is not the most obvious version
   (different difficulty, an exception case, a second colour role). Difficulty target: {difficulty}.
2. Write `generate_input(rng)` that returns a fresh random input grid every call, using only the supplied
   `rng` (an instance of `random.Random`) for randomness. Vary grid size, object count, positions and colours
   between calls so no two inputs look alike. Every generated input must contain the information needed to
   apply the rule (markers, legends, etc.) and must NOT already be in its output state.
3. Write `transform(grid)` that applies the rule exactly and deterministically. It must work for every grid
   `generate_input` can produce, never raise, and return a valid grid (1-30 by 1-30, ints 0-9).
4. Both functions must be pure Python (standard library + numpy allowed, nothing else). No file, network,
   os, sys, subprocess or time access. No printing. Each function should run in well under 0.1 s.
5. `transform(grid)` must differ from `grid` for at least 90% of generated inputs, and outputs must not be
   constant across inputs.

Reply in EXACTLY this format and nothing else:

### NAME
<3-6 word name>

### RULE
<Precise natural-language description of the rule, 2-6 sentences, written so that another programmer could
implement transform() from this text alone, without seeing your code. State every colour role and every edge case.>

### GENERATE_INPUT
```python
import random
import numpy as np

def generate_input(rng: random.Random) -> list[list[int]]:
    ...
```

### TRANSFORM
```python
import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    ...
```
"""

REIMPLEMENT_SYSTEM = "You implement ARC-AGI style grid transformations exactly as specified."

REIMPLEMENT_USER = """\
{arc_rules}
Implement the following rule as a Python function. You see the rule text and some example INPUT grids
(no outputs). Implement exactly what the text says; do not invent extra behaviour.

### RULE
{rule}

### EXAMPLE INPUTS
{examples}

Constraints: pure Python (standard library + numpy only), deterministic, never raise on inputs like the
examples, return a list of lists of ints 0-9 with 1-30 rows and 1-30 columns. No printing.

Reply in EXACTLY this format and nothing else:

### TRANSFORM
```python
import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    ...
```
"""

DIFFICULTIES = {
    "easy": "easy (a single clear step; a person solves it in under a minute)",
    "medium": "medium (two interacting ideas, or one idea with an exception case)",
    "hard": "hard (three steps or a relation between objects that must be discovered; still solvable by a careful person)",
}


def grid_to_text(grid):
    return "\n".join(" ".join(str(int(c)) for c in row) for row in grid)


def author_messages(seed, difficulty="medium"):
    return [
        {"role": "system", "content": AUTHOR_SYSTEM},
        {"role": "user", "content": AUTHOR_USER.format(
            arc_rules=ARC_RULES, category=seed["category"], description=seed["description"],
            difficulty=DIFFICULTIES.get(difficulty, difficulty))},
    ]


def reimplement_messages(rule, example_inputs):
    ex = "\n\n".join(f"Example {i+1}:\n{grid_to_text(g)}" for i, g in enumerate(example_inputs))
    return [
        {"role": "system", "content": REIMPLEMENT_SYSTEM},
        {"role": "user", "content": REIMPLEMENT_USER.format(arc_rules=ARC_RULES, rule=rule, examples=ex)},
    ]


def parse_sections(text):
    """Parse '### HEADER' sections; code fences inside sections are stripped."""
    import re
    out = {}
    parts = re.split(r"^###\s+([A-Z_]+)\s*$", text, flags=re.M)
    # parts = [preamble, H1, body1, H2, body2, ...]
    for i in range(1, len(parts) - 1, 2):
        body = parts[i + 1].strip()
        m = re.search(r"```(?:python)?\s*\n(.*?)```", body, flags=re.S)
        if m:
            body = m.group(1).strip("\n")
        out[parts[i].strip()] = body
    return out
